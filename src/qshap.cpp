#include "qshap.h"
#include <complex>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <Eigen/Dense>

// Compute a decision signature for a sample: for every internal node,
// record whether x[feature] <= threshold. The weight matrix depends on
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
            sig[v] = (x(summary_tree.feature(v)) <= summary_tree.threshold(v));
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
// TreeSHAP for a single tree (naïve subset enumeration).
// Efficient for trees where the number of unique split features M <= ~20.
// Complexity: O(n * M * 2^M * depth) per tree.
// ---------------------------------------------------------------------------

// Recursive conditional expectation: E[pred(x) | x_S]
// feat_mask: bitmask over features_used indicating which are in S.
static double cond_exp(
    int node,
    const Eigen::VectorXd &x,
    int feat_mask,
    const std::vector<int> &features_used,
    const Eigen::VectorXi &cl,
    const Eigen::VectorXi &cr,
    const Eigen::VectorXi &feat,
    const Eigen::VectorXd &thresh,
    const Eigen::VectorXd &val,
    const Eigen::VectorXd &nsamp)
{
    if (cl(node) < 0)
        return val(node); // leaf

    int f = feat(node);
    int left = cl(node);
    int right = cr(node);

    // Check if this feature is conditioned on (in S)
    bool conditioned = false;
    for (size_t fi = 0; fi < features_used.size(); fi++)
    {
        if (features_used[fi] == f && (feat_mask & (1 << fi)))
        {
            conditioned = true;
            break;
        }
    }

    if (conditioned)
    {
        if (x(f) <= thresh(node))
            return cond_exp(left, x, feat_mask, features_used, cl, cr, feat, thresh, val, nsamp);
        else
            return cond_exp(right, x, feat_mask, features_used, cl, cr, feat, thresh, val, nsamp);
    }
    else
    {
        double nl = nsamp(left);
        double nr = nsamp(right);
        double nt = nl + nr;
        return (nl / nt) * cond_exp(left, x, feat_mask, features_used, cl, cr, feat, thresh, val, nsamp) +
               (nr / nt) * cond_exp(right, x, feat_mask, features_used, cl, cr, feat, thresh, val, nsamp);
    }
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

    // Find unique features used in splits
    std::vector<int> features_used;
    for (int v = 0; v < children_left.size(); v++)
    {
        if (children_left(v) >= 0)
        {
            int f = feature(v);
            if (std::find(features_used.begin(), features_used.end(), f) == features_used.end())
                features_used.push_back(f);
        }
    }
    const int M = (int)features_used.size();

    if (M > 20)
        Rcpp::stop("compute_treeshap: tree uses %d unique features (max 20 for subset enumeration)", M);

    // Precompute Shapley weights: w(s) = s! * (M-s-1)! / M!
    std::vector<double> sw(M, 0.0);
    {
        std::vector<double> lf(M + 1, 0.0);
        for (int i = 1; i <= M; i++)
            lf[i] = lf[i - 1] + std::log((double)i);
        for (int s = 0; s < M; s++)
            sw[s] = std::exp(lf[s] + lf[M - s - 1] - lf[M]);
    }

    Eigen::MatrixXd shap_out = Eigen::MatrixXd::Zero(n, p);

    for (int i = 0; i < n; i++)
    {
        const Eigen::VectorXd xi = x.row(i);

        for (int fi = 0; fi < M; fi++)
        {
            int j = features_used[fi];
            double phi = 0.0;

            // Enumerate subsets of features_used \ {feature fi}
            // other_count = M - 1; 2^(M-1) subsets
            int other_count = M - 1;
            int total_subsets = 1 << other_count;

            for (int mask_other = 0; mask_other < total_subsets; mask_other++)
            {
                // Build full mask for S (without j) and S∪{j}
                // Map bits of mask_other to features_used, skipping fi
                int mask_S = 0;
                int bit = 0;
                for (int fi2 = 0; fi2 < M; fi2++)
                {
                    if (fi2 == fi)
                        continue;
                    if (mask_other & (1 << bit))
                        mask_S |= (1 << fi2);
                    bit++;
                }
                int mask_S_plus_j = mask_S | (1 << fi);

                int s = __builtin_popcount(mask_S);

                double v_with = cond_exp(0, xi, mask_S_plus_j, features_used,
                                         children_left, children_right, feature,
                                         threshold, value, n_node_samples);
                double v_without = cond_exp(0, xi, mask_S, features_used,
                                            children_left, children_right, feature,
                                            threshold, value, n_node_samples);

                phi += sw[s] * (v_with - v_without);
            }

            shap_out(i, j) = phi;
        }
    }

    return shap_out;
}