#include "qshap.h"
#include <complex>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <Eigen/Dense>

// Compute a decision signature for a sample: for every internal node,
// record the backend-specific branch decision. The weight matrix depends on
// the sample's decision at ALL internal nodes (not just the path to the
// sample's leaf), because traversal_weight() explores both branches and
// uses x(split_feature) <= threshold to distribute weight at every node.
// Two samples with identical signatures produce identical weight matrices.
static std::vector<bool> compute_decision_signature(
    const Eigen::VectorXd &x,
    const TreeSummary &summary_tree)
{
    const int nc = summary_tree.node_count;
    std::vector<bool> sig(nc, false);
    for (int v = 0; v < nc; v++)
    {
        if (summary_tree.children_left(v) >= 0) // internal node
        {
            sig[v] = tree_goes_left(
                x(summary_tree.feature(v)), summary_tree.threshold(v),
                summary_tree.default_left(v), summary_tree.xgboost_split);
        }
    }
    return sig;
}

// Hash for vector<bool> to use as unordered_map key
struct VecBoolHash
{
    size_t operator()(const std::vector<bool> &v) const
    {
        size_t seed = v.size();
        for (size_t i = 0; i < v.size(); i++)
        {
            if (v[i])
                seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

Eigen::MatrixXd T2(
    const Eigen::MatrixXd &x,
    const Rcpp::List &tree_summary,
    const Eigen::MatrixXcd &store_v_invc,
    const Eigen::MatrixXcd &store_z,
    bool parallel)
{
    TreeSummary summary_tree = list_to_tree_summary(tree_summary);

    std::vector<double> init_prediction_vec;
    for (int i = 0; i < summary_tree.init_prediction.size(); i++)
    {
        if (summary_tree.children_left(i) < 0)
        {
            init_prediction_vec.push_back(summary_tree.init_prediction(i));
        }
    }
    Eigen::Map<Eigen::VectorXd> init_prediction(init_prediction_vec.data(), init_prediction_vec.size());

    const int n = x.rows();
    const int p = x.cols();
    Eigen::MatrixXd shap_value = Eigen::MatrixXd::Zero(n, p);

    // Group samples by decision signature: for every internal node,
    // record whether x[feature] <= threshold. Two samples with identical
    // signatures produce identical weight matrices, so T2_sample gives
    // identical results. We compute weight + T2_sample once per unique
    // group, then broadcast. Complexity reduces from O(n * L^2 * D^2)
    // to O(G * L^2 * D^2) where G = number of unique groups (<= min(n, 2^K)).
    std::unordered_map<std::vector<bool>, std::vector<int>, VecBoolHash> groups;
    groups.reserve(n);
    for (int i = 0; i < n; i++)
    {
        auto sig = compute_decision_signature(x.row(i), summary_tree);
        groups[std::move(sig)].push_back(i);
    }

    for (const auto &group : groups)
    {
        int representative = group.second[0];
        const auto w = weight(x.row(representative), summary_tree);
        T2_sample(representative, w.first, w.second, init_prediction,
                  store_v_invc, store_z, shap_value, summary_tree.feature_uniq);

        for (size_t k = 1; k < group.second.size(); k++)
        {
            shap_value.row(group.second[k]) = shap_value.row(representative);
        }
    }

    return shap_value;
}

// A much more optimized version of T2_sample
void T2_sample(
    int i,
    const Eigen::MatrixXd &w_matrix,
    const Eigen::MatrixXi &w_ind,
    const Eigen::VectorXd &init_prediction,
    const Eigen::MatrixXcd &store_v_invc,
    const Eigen::MatrixXcd &store_z,
    Eigen::MatrixXd &shap_value,
    const Eigen::VectorXi &feature_uniq)
{
    const int L = w_matrix.rows();
    const double eps2 = 1e-18; // (1e-9)^2

    // Reuse buffers across calls (thread-safe if you don't parallelize here)
    thread_local std::vector<int> union_feats;
    thread_local std::vector<std::complex<double>> pz;

    union_feats.clear();
    union_feats.reserve(feature_uniq.size()); // small
    // pz sized later per n12_c

    for (int l1 = 0; l1 < L; ++l1)
    {
        for (int l2 = l1; l2 < L; ++l2)
        {
            const double init_prod = init_prediction(l1) * init_prediction(l2);

            // ---- build union feature list (still same logic, but reuse vector) ----
            union_feats.clear();
            for (int j_idx = 0; j_idx < feature_uniq.size(); ++j_idx)
            {
                const int feat = feature_uniq(j_idx);
                if ((w_ind(l1, feat) + w_ind(l2, feat)) >= 1)
                    union_feats.push_back(feat);
            }
            const int n12 = (int)union_feats.size();
            if (n12 == 0)
                continue;

            const int n12_c = n12 / 2 + 1;

            // ---- NO COPIES: refer to stored rows ----
            // Important: use Ref to avoid allocation / copy
            const Eigen::Ref<const Eigen::VectorXcd> v_invc =
                store_v_invc.row(n12).head(n12_c);
            const Eigen::Ref<const Eigen::VectorXcd> z_roots =
                store_z.row(n12).head(n12_c);

            // ---- compute p_z_overall(k) for all k ----
            pz.assign(n12_c, std::complex<double>(1.0, 0.0)); // reuse capacity, but sets values

            for (int k = 0; k < n12_c; ++k)
            {
                std::complex<double> prod(1.0, 0.0);
                const std::complex<double> zk = z_roots(k);

                // product over features in union
                for (int idx = 0; idx < n12; ++idx)
                {
                    const int f = union_feats[idx];
                    const double a = w_matrix(l1, f);
                    const double b = w_matrix(l2, f);
                    prod *= (zk + (a * b));
                }
                pz[k] = prod;
            }

            // ---- for each feature j in union: compute dot(pz/denom, v_invc) ON THE FLY ----
            for (int idx = 0; idx < n12; ++idx)
            {
                const int j = union_feats[idx];
                const double a = w_matrix(l1, j);
                const double b = w_matrix(l2, j);
                const double w_factor = (a * b - 1.0);

                // Compute contribution_val = complex_dot_v2(tmp, v_invc, n12)
                // where tmp[k] = pz[k] / (z_roots[k] + a*b).
                std::complex<double> acc = pz[0] / (z_roots(0) + (a * b)) * v_invc(0);

                if (n12 % 2 == 0)
                {
                    // even: middle terms doubled except endpoints
                    for (int k = 1; k < n12_c - 1; ++k)
                    {
                        const std::complex<double> denom = z_roots(k) + (a * b);
                        // tiny denom guard (use norm to avoid sqrt)
                        if (std::norm(denom) < eps2)
                            continue;
                        acc += 2.0 * (pz[k] / denom) * v_invc(k);
                    }
                    // last term not doubled
                    {
                        const int k = n12_c - 1;
                        const std::complex<double> denom = z_roots(k) + (a * b);
                        if (std::norm(denom) >= eps2)
                            acc += (pz[k] / denom) * v_invc(k);
                    }
                }
                else
                {
                    // odd: all k>=1 doubled
                    for (int k = 1; k < n12_c; ++k)
                    {
                        const std::complex<double> denom = z_roots(k) + (a * b);
                        if (std::norm(denom) < eps2)
                            continue;
                        acc += 2.0 * (pz[k] / denom) * v_invc(k);
                    }
                }

                const double contribution_val = acc.real();
                const double final_contribution = w_factor * contribution_val * init_prod;

                shap_value(i, j) += (l1 == l2) ? final_contribution : (2.0 * final_contribution);
            }
        }
    }
}

Eigen::MatrixXd loss_treeshap(
    const Eigen::MatrixXd &x,
    const Eigen::VectorXd &y,
    const Rcpp::List &tree_summary,
    const Eigen::MatrixXcd &store_v_invc,
    const Eigen::MatrixXcd &store_z,
    const Eigen::MatrixXd &T0_x,
    double learning_rate)
{

    const int n_samples = x.rows();
    const int n_features = T0_x.cols();

    // Compute T2 once
    Eigen::MatrixXd T2_values = T2(x, tree_summary, store_v_invc, store_z, false);

    // Precompute scalars
    const double lr = learning_rate;
    const double lr2 = lr * lr;
    const double c = 2.0 * lr;

    // Robust lr==1 check to avoid scaling, as xgboost and lightgbm already adjusted for scaliing
    const bool lr_is_one = std::abs(lr - 1.0) <= 1e-12;

    // Allocate output
    Eigen::MatrixXd loss(n_samples, n_features);

    if (lr_is_one)
    {
        // loss = T2_values - 2 * (T0_x .* y)   (columnwise)
        loss.noalias() = T2_values;

        for (int j = 0; j < n_features; ++j)
        {
            loss.col(j).noalias() -= 2.0 * T0_x.col(j).cwiseProduct(y);
        }
    }
    else
    {
        // loss = lr^2 * T2_values - (2*lr) * (T0_x .* y)   (columnwise)
        loss.noalias() = T2_values;
        loss *= lr2;

        for (int j = 0; j < n_features; ++j)
        {
            loss.col(j).noalias() -= c * T0_x.col(j).cwiseProduct(y);
        }
    }

    return loss;
}

// ---------------------------------------------------------------------------
// Exact path-dependent TreeSHAP for a single tree.
//
// The previous implementation enumerated every subset of the unique split
// features and therefore stopped at 20 features.  The path recursion below is
// algebraically equivalent to that conditional-expectation game, including
// repeated features on a path, but runs in O(L * D^2) per sample (L leaves,
// maximum depth D) and does not depend exponentially on the total number of
// features used by the tree.
// ---------------------------------------------------------------------------

struct TreeShapPathElement
{
    int feature_index;
    double zero_fraction;
    double one_fraction;
    double pweight;
};

static void extend_tree_shap_path(
    std::vector<TreeShapPathElement> &path,
    double zero_fraction,
    double one_fraction,
    int feature_index)
{
    const int unique_depth = static_cast<int>(path.size());
    path.push_back({feature_index, zero_fraction, one_fraction,
                    unique_depth == 0 ? 1.0 : 0.0});

    for (int i = unique_depth - 1; i >= 0; --i)
    {
        path[i + 1].pweight +=
            one_fraction * path[i].pweight *
            static_cast<double>(i + 1) / static_cast<double>(unique_depth + 1);
        path[i].pweight =
            zero_fraction * path[i].pweight *
            static_cast<double>(unique_depth - i) /
            static_cast<double>(unique_depth + 1);
    }
}

static void unwind_tree_shap_path(
    std::vector<TreeShapPathElement> &path,
    int path_index)
{
    const int unique_depth = static_cast<int>(path.size()) - 1;
    const double one_fraction = path[path_index].one_fraction;
    const double zero_fraction = path[path_index].zero_fraction;
    double next_one_portion = path[unique_depth].pweight;

    for (int i = unique_depth - 1; i >= 0; --i)
    {
        if (one_fraction != 0.0)
        {
            const double tmp = path[i].pweight;
            path[i].pweight =
                next_one_portion * static_cast<double>(unique_depth + 1) /
                (static_cast<double>(i + 1) * one_fraction);
            next_one_portion =
                tmp - path[i].pweight * zero_fraction *
                          static_cast<double>(unique_depth - i) /
                          static_cast<double>(unique_depth + 1);
        }
        else
        {
            path[i].pweight =
                path[i].pweight * static_cast<double>(unique_depth + 1) /
                (zero_fraction * static_cast<double>(unique_depth - i));
        }
    }

    // The loop above has already unwound the permutation weights in place.
    // Only the feature/fraction metadata shifts left; shifting pweight as well
    // would overwrite those newly reconstructed weights for repeated splits.
    for (int i = path_index; i < unique_depth; ++i)
    {
        path[i].feature_index = path[i + 1].feature_index;
        path[i].zero_fraction = path[i + 1].zero_fraction;
        path[i].one_fraction = path[i + 1].one_fraction;
    }
    path.pop_back();
}

static double unwound_tree_shap_path_sum(
    const std::vector<TreeShapPathElement> &path,
    int path_index)
{
    const int unique_depth = static_cast<int>(path.size()) - 1;
    const double one_fraction = path[path_index].one_fraction;
    const double zero_fraction = path[path_index].zero_fraction;
    double next_one_portion = path[unique_depth].pweight;
    double total = 0.0;

    if (one_fraction != 0.0)
    {
        for (int i = unique_depth - 1; i >= 0; --i)
        {
            const double tmp =
                next_one_portion * static_cast<double>(unique_depth + 1) /
                (static_cast<double>(i + 1) * one_fraction);
            total += tmp;
            next_one_portion =
                path[i].pweight - tmp * zero_fraction *
                                      static_cast<double>(unique_depth - i) /
                                      static_cast<double>(unique_depth + 1);
        }
    }
    else
    {
        for (int i = unique_depth - 1; i >= 0; --i)
        {
            total +=
                path[i].pweight * static_cast<double>(unique_depth + 1) /
                (zero_fraction * static_cast<double>(unique_depth - i));
        }
    }

    return total;
}

static void tree_shap_recursive(
    int node,
    const Eigen::VectorXd &x,
    const Eigen::VectorXi &children_left,
    const Eigen::VectorXi &children_right,
    const Eigen::VectorXi &feature,
    const Eigen::VectorXd &threshold,
    const Eigen::VectorXd &value,
    const Eigen::VectorXd &n_node_samples,
    std::vector<TreeShapPathElement> path,
    double parent_zero_fraction,
    double parent_one_fraction,
    int parent_feature_index,
    Eigen::VectorXd &phi)
{
    // A branch with neither conditional nor unconditional probability cannot
    // contribute.  Skipping it also avoids an indeterminate 0/0 while
    // unwinding a genuinely zero-cover branch.
    if (parent_zero_fraction == 0.0 && parent_one_fraction == 0.0)
        return;

    extend_tree_shap_path(
        path, parent_zero_fraction, parent_one_fraction, parent_feature_index);

    const int left = children_left(node);
    if (left < 0)
    {
        const int unique_depth = static_cast<int>(path.size()) - 1;
        for (int i = 1; i <= unique_depth; ++i)
        {
            const double w = unwound_tree_shap_path_sum(path, i);
            const TreeShapPathElement &element = path[i];
            phi(element.feature_index) +=
                w * (element.one_fraction - element.zero_fraction) * value(node);
        }
        return;
    }

    const int right = children_right(node);
    const int split_feature = feature(node);
    const bool go_left = x(split_feature) <= threshold(node);
    const int hot = go_left ? left : right;
    const int cold = go_left ? right : left;

    const double left_cover = n_node_samples(left);
    const double right_cover = n_node_samples(right);
    const double total_cover = left_cover + right_cover;
    if (!std::isfinite(left_cover) || !std::isfinite(right_cover) ||
        left_cover < 0.0 || right_cover < 0.0 ||
        !std::isfinite(total_cover) || total_cover <= 0.0)
    {
        Rcpp::stop(
            "compute_treeshap: child covers at node %d must be finite, "
            "non-negative, and have a positive sum",
            node);
    }

    double incoming_zero_fraction = 1.0;
    double incoming_one_fraction = 1.0;
    for (int i = 0; i < static_cast<int>(path.size()); ++i)
    {
        if (path[i].feature_index == split_feature)
        {
            incoming_zero_fraction = path[i].zero_fraction;
            incoming_one_fraction = path[i].one_fraction;
            unwind_tree_shap_path(path, i);
            break;
        }
    }

    const double hot_cover = n_node_samples(hot);
    const double cold_cover = n_node_samples(cold);
    tree_shap_recursive(
        hot, x, children_left, children_right, feature, threshold, value,
        n_node_samples, path,
        incoming_zero_fraction * hot_cover / total_cover,
        incoming_one_fraction, split_feature, phi);
    tree_shap_recursive(
        cold, x, children_left, children_right, feature, threshold, value,
        n_node_samples, path,
        incoming_zero_fraction * cold_cover / total_cover,
        0.0, split_feature, phi);
}

// [[Rcpp::export]]
Eigen::MatrixXd compute_treeshap(
    const Eigen::MatrixXd &x,
    const Eigen::VectorXi &children_left,
    const Eigen::VectorXi &children_right,
    const Eigen::VectorXi &feature,
    const Eigen::VectorXd &threshold,
    const Eigen::VectorXd &value,
    const Eigen::VectorXd &n_node_samples)
{
    const int n = x.rows();
    const int p = x.cols();

    Eigen::MatrixXd shap_out = Eigen::MatrixXd::Zero(n, p);

    const int node_count = children_left.size();
    if (node_count == 0 || children_right.size() != node_count ||
        feature.size() != node_count || threshold.size() != node_count ||
        value.size() != node_count || n_node_samples.size() != node_count)
    {
        Rcpp::stop("compute_treeshap: tree arrays must have the same positive length");
    }

    for (int node = 0; node < node_count; ++node)
    {
        const int left = children_left(node);
        const int right = children_right(node);
        if ((left < 0) != (right < 0) || left >= node_count || right >= node_count)
            Rcpp::stop("compute_treeshap: invalid children at node %d", node);
        if (left >= 0 && (feature(node) < 0 || feature(node) >= p))
            Rcpp::stop("compute_treeshap: invalid split feature at node %d", node);
    }

    for (int i = 0; i < n; i++)
    {
        const Eigen::VectorXd xi = x.row(i);
        Eigen::VectorXd phi = Eigen::VectorXd::Zero(p);
        std::vector<TreeShapPathElement> path;
        tree_shap_recursive(
            0, xi, children_left, children_right, feature, threshold, value,
            n_node_samples, path, 1.0, 1.0, -1, phi);
        shap_out.row(i) = phi.transpose();
    }

    return shap_out;
}
