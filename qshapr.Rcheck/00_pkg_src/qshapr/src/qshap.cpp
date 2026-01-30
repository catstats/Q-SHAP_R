#include "qshap.h"
#include <complex>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

#include <chrono>

struct AccTimer
{
    double &acc;
    std::chrono::high_resolution_clock::time_point t0;
    AccTimer(double &a) : acc(a), t0(std::chrono::high_resolution_clock::now()) {}
    ~AccTimer()
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        acc += std::chrono::duration<double>(t1 - t0).count();
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

    Eigen::MatrixXd shap_value = Eigen::MatrixXd::Zero(x.rows(), x.cols());

    for (int i = 0; i < x.rows(); i++)
    {
        const auto w = weight(x.row(i), summary_tree);
        T2_sample(i, w.first, w.second, init_prediction, store_v_invc, store_z, shap_value, summary_tree.feature_uniq);
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
