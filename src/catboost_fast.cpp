#include "catboost_oblivious_core.h"
#include "catboost_fused_router.h"

// [[Rcpp::depends(RcppEigen)]]

namespace {

struct LeafContrib {
    std::vector<int> feature_ids;
    std::vector<double> T0;
    std::vector<double> T2;
};

void validate_oblivious_tree(const ObliviousTree &tree, int n_features) {
    if (tree.max_depth < 0 || tree.max_depth >= 30) {
        Rcpp::stop("Invalid CatBoost symmetric-tree depth.");
    }

    const int num_leaves = 1 << tree.max_depth;
    const int num_internal = num_leaves - 1;
    const int expected_nodes = 2 * num_leaves - 1;
    if (tree.node_count != expected_nodes ||
        tree.children_left.size() != expected_nodes ||
        tree.children_right.size() != expected_nodes ||
        tree.feature.size() != expected_nodes ||
        tree.threshold.size() != expected_nodes ||
        tree.n_node_samples.size() != expected_nodes ||
        tree.value.size() != expected_nodes) {
        Rcpp::stop(
            "Fast CatBoost Q-SHAP requires a complete symmetric numeric tree."
        );
    }

    for (int level = 0; level < tree.max_depth; ++level) {
        const int first = (1 << level) - 1;
        const int end = first + (1 << level);
        const int level_feature = tree.feature(first);
        const double level_threshold = tree.threshold(first);
        if (level_feature < 0 || level_feature >= n_features ||
            !std::isfinite(level_threshold)) {
            Rcpp::stop(
                "CatBoost split feature or threshold is invalid for the supplied x."
            );
        }

        for (int node = first; node < end; ++node) {
            if (tree.children_left(node) != 2 * node + 1 ||
                tree.children_right(node) != 2 * node + 2 ||
                tree.feature(node) != level_feature ||
                tree.threshold(node) != level_threshold) {
                Rcpp::stop(
                    "Fast CatBoost Q-SHAP requires one shared split per level."
                );
            }
        }
    }

    for (int node = num_internal; node < expected_nodes; ++node) {
        if (tree.children_left(node) >= 0 || tree.children_right(node) >= 0) {
            Rcpp::stop(
                "Fast CatBoost Q-SHAP requires a complete symmetric numeric tree."
            );
        }
    }
    for (int node = 0; node < expected_nodes; ++node) {
        if (!std::isfinite(tree.n_node_samples(node)) ||
            tree.n_node_samples(node) <= 0.0 ||
            !std::isfinite(tree.value(node))) {
            Rcpp::stop("Invalid CatBoost tree cover or value.");
        }
    }
}

}

// [[Rcpp::export]]
Rcpp::List catboost_qshap_r2_fast(const Rcpp::NumericMatrix &X,
                                  const Rcpp::NumericVector &y,
                                  const Rcpp::List &catboost_trees,
                                  double bias,
                                  bool compute_sd = true) {
    // Fast global CatBoost Q-SHAP backend. It groups by parsed leaf id,
    // never by raw prediction.
    // loss[i,j] = T2[i,j] - 2 * r_i^(k-1) * T0[i,j] because CatBoost JSON
    // leaf values are already scaled by the model learning rate.
    const int n = X.nrow();
    const int p = X.ncol();
    const int num_trees = catboost_trees.size();

    if (n <= 0) {
        Rcpp::stop("x must contain at least one row.");
    }
    if (y.size() != n) {
        Rcpp::stop("y must have one value per row of x.");
    }

    std::vector<ObliviousTree> trees;
    std::vector<qshap_catboost_core::TreeRouteSpec> route_specs;
    trees.reserve(static_cast<size_t>(num_trees));
    route_specs.reserve(static_cast<size_t>(num_trees));
    for (int tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
        trees.push_back(list_to_oblivious_tree(catboost_trees[tree_idx]));
        validate_oblivious_tree(trees.back(), p);
        const ObliviousTree &tree = trees.back();
        qshap_catboost_core::TreeRouteSpec route;
        route.level_feature.resize(static_cast<size_t>(tree.max_depth));
        route.level_border.resize(static_cast<size_t>(tree.max_depth));
        for (int level = 0; level < tree.max_depth; ++level) {
            const int node = (1 << level) - 1;
            route.level_feature[static_cast<size_t>(level)] = tree.feature(node);
            route.level_border[static_cast<size_t>(level)] =
                static_cast<float>(tree.threshold(node));
        }
        route_specs.push_back(std::move(route));
    }
    // catboost_qshap_matrix() has already replaced missing values with the
    // feature-specific infinities used by the parsed CatBoost metadata.
    const std::vector<int> nan_goes_right(static_cast<size_t>(p), 0);
    const qshap_catboost_core::QuantizedRoutingPlan routing =
        qshap_catboost_core::build_quantized_routing_plan(
            X.begin(), n, p, true, route_specs, nan_goes_right
        );

    Eigen::VectorXd cum_pred = Eigen::VectorXd::Constant(n, bias);
    Eigen::VectorXd loss_sum = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd loss_sumsq = Eigen::VectorXd::Zero(p);
    Eigen::MatrixXd loss_by_row;
    if (compute_sd) {
        // SD is defined from each row's loss after summing over all trees.
        // Keeping row totals preserves the cross-tree covariance terms that
        // are lost when per-tree squared losses are added independently.
        loss_by_row = Eigen::MatrixXd::Zero(n, p);
    }

    for (int tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
        const ObliviousTree &tree = trees[static_cast<size_t>(tree_idx)];
        const qshap_catboost_core::QuantizedTreeRoute &tree_route =
            routing.tree_routes[static_cast<size_t>(tree_idx)];
        const int depth = tree.max_depth;
        const int num_leaves = 1 << depth;
        const int num_internal = num_leaves - 1;

        std::vector<int> group_n(num_leaves, 0);
        std::vector<double> group_sum_r(num_leaves, 0.0);
        std::vector<int> sample_leaf;
        std::vector<double> sample_residual;
        if (compute_sd) {
            sample_leaf.resize(static_cast<size_t>(n));
            sample_residual.resize(n);
        }

        auto consume_leaf = [&](int i, int leaf) {
            const double residual = y[i] - cum_pred(i);
            group_n[leaf] += 1;
            group_sum_r[leaf] += residual;
            if (compute_sd) {
                sample_leaf[static_cast<size_t>(i)] = leaf;
                sample_residual[i] = residual;
            }
            cum_pred(i) += tree.value(num_internal + leaf);
        };
        qshap_catboost_core::route_tree_tiled(
            routing, tree_route, n, consume_leaf
        );

        std::vector<LeafContrib> leaf_cache(num_leaves);

        for (int leaf = 0; leaf < num_leaves; ++leaf) {
            if (group_n[leaf] == 0) continue;

            LeafContrib &leaf_contrib = leaf_cache[leaf];
            catboost_leaf_T0_T2_compact(
                tree,
                leaf,
                p,
                leaf_contrib.feature_ids,
                leaf_contrib.T0,
                leaf_contrib.T2
            );

            const double m = static_cast<double>(group_n[leaf]);
            const double sum_r = group_sum_r[leaf];

            for (int idx = 0; idx < static_cast<int>(leaf_contrib.feature_ids.size()); ++idx) {
                const int feature_id = leaf_contrib.feature_ids[idx];
                const double a = leaf_contrib.T2[idx];
                const double b = 2.0 * leaf_contrib.T0[idx];
                if (a == 0.0 && b == 0.0) continue;
                loss_sum(feature_id) += m * a - b * sum_r;
            }
        }

        if (compute_sd) {
            for (int i = 0; i < n; ++i) {
                const LeafContrib &leaf_contrib = leaf_cache[sample_leaf[i]];
                const double residual = sample_residual[i];
                for (int idx = 0;
                     idx < static_cast<int>(leaf_contrib.feature_ids.size());
                     ++idx) {
                    const int feature_id = leaf_contrib.feature_ids[idx];
                    const double row_loss =
                        leaf_contrib.T2[idx] - 2.0 * leaf_contrib.T0[idx] * residual;
                    loss_by_row(i, feature_id) += row_loss;
                }
            }
        }
    }

    double y_sum = 0.0;
    for (int i = 0; i < n; ++i) y_sum += y[i];
    const double y_mean = y_sum / static_cast<double>(n);
    double SST = 0.0;
    for (int i = 0; i < n; ++i) {
        const double centered = y[i] - y_mean;
        SST += centered * centered;
    }
    if (SST <= 0.0) {
        Rcpp::stop("Cannot compute R2 decomposition when y has zero variance.");
    }
    Eigen::VectorXd rsq = -loss_sum / SST;

    Rcpp::List out = Rcpp::List::create(
        Rcpp::Named("rsq") = rsq,
        Rcpp::Named("loss_sum") = loss_sum,
        Rcpp::Named("n") = n,
        Rcpp::Named("SST") = SST
    );

    if (compute_sd) {
        loss_sumsq = loss_by_row.array().square().colwise().sum().transpose();
        Eigen::VectorXd sd_rsq(p);
        if (n > 1) {
            for (int j = 0; j < p; ++j) {
                const double var = std::max(
                    (loss_sumsq(j) - (loss_sum(j) * loss_sum(j)) / n) / (n - 1),
                    0.0
                );
                sd_rsq(j) = std::sqrt(static_cast<double>(n) * var) / SST;
            }
        } else {
            sd_rsq.setConstant(NA_REAL);
        }
        out["loss_sumsq"] = loss_sumsq;
        out["sd_rsq"] = sd_rsq;
    } else {
        out["loss_sumsq"] = R_NilValue;
        out["sd_rsq"] = R_NilValue;
    }

    return out;
}

// [[Rcpp::export]]
Rcpp::List catboost_qshap_r2_leaf_cached_nogroup(const Eigen::MatrixXd &X,
                                                 const Eigen::VectorXd &y,
                                                 const Rcpp::List &catboost_trees,
                                                 double bias) {
    // Baseline for benchmarking the leaf/path residual grouping theorem.
    // It still caches T0/T2 once per reached leaf, but it does NOT aggregate
    // residuals by leaf before updating loss_sum.
    const int n = X.rows();
    const int p = X.cols();
    const int num_trees = catboost_trees.size();

    Eigen::VectorXd cum_pred = Eigen::VectorXd::Constant(n, bias);
    Eigen::VectorXd loss_sum = Eigen::VectorXd::Zero(p);

    for (int tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
        ObliviousTree tree = list_to_oblivious_tree(catboost_trees[tree_idx]);
        const int depth = tree.max_depth;
        const int num_leaves = 1 << depth;
        const int num_internal = num_leaves - 1;
        if (tree.node_count != (2 * num_leaves - 1)) {
            Rcpp::stop("Leaf-cached CatBoost Q-SHAP baseline supports symmetric numeric trees only.");
        }

        std::vector<int> level_feature(depth);
        std::vector<double> level_threshold(depth);
        for (int t = 0; t < depth; ++t) {
            const int node = (1 << t) - 1;
            level_feature[t] = tree.feature(node);
            level_threshold[t] = tree.threshold(node);
        }

        std::vector<char> leaf_ready(num_leaves, 0);
        std::vector<LeafContrib> leaf_cache(num_leaves);

        for (int i = 0; i < n; ++i) {
            const int leaf = route_oblivious_leaf(X, i, level_feature, level_threshold);
            if (!leaf_ready[leaf]) {
                catboost_leaf_T0_T2_compact(
                    tree,
                    leaf,
                    p,
                    leaf_cache[leaf].feature_ids,
                    leaf_cache[leaf].T0,
                    leaf_cache[leaf].T2
                );
                leaf_ready[leaf] = 1;
            }

            const double residual = y(i) - cum_pred(i);
            const LeafContrib &leaf_contrib = leaf_cache[leaf];
            for (int idx = 0; idx < static_cast<int>(leaf_contrib.feature_ids.size()); ++idx) {
                const int feature_id = leaf_contrib.feature_ids[idx];
                const double a = leaf_contrib.T2[idx];
                const double b = 2.0 * leaf_contrib.T0[idx];
                if (a == 0.0 && b == 0.0) continue;
                loss_sum(feature_id) += a - b * residual;
            }

            cum_pred(i) += tree.value(num_internal + leaf);
        }
    }

    const double y_mean = y.mean();
    const double SST = (y.array() - y_mean).square().sum();
    if (SST <= 0.0) {
        Rcpp::stop("Cannot compute R2 decomposition when y has zero variance.");
    }
    Eigen::VectorXd rsq = -loss_sum / SST;

    return Rcpp::List::create(
        Rcpp::Named("rsq") = rsq,
        Rcpp::Named("loss_sum") = loss_sum,
        Rcpp::Named("n") = n,
        Rcpp::Named("SST") = SST
    );
}

// [[Rcpp::export]]
Eigen::MatrixXd catboost_t0_grouped_fast(const Eigen::MatrixXd &X,
                                         const Rcpp::List &catboost_trees) {
    // T0-only benchmark backend for CatBoost symmetric trees.
    const int n = X.rows();
    const int p = X.cols();
    const int num_trees = catboost_trees.size();
    Eigen::MatrixXd T0 = Eigen::MatrixXd::Zero(n, p);

    for (int tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
        ObliviousTree tree = list_to_oblivious_tree(catboost_trees[tree_idx]);
        const int depth = tree.max_depth;
        const int num_leaves = 1 << depth;
        if (tree.node_count != (2 * num_leaves - 1)) {
            Rcpp::stop("Fast CatBoost T0 currently supports regression with symmetric numeric trees.");
        }

        std::vector<int> level_feature(depth);
        std::vector<double> level_threshold(depth);
        for (int t = 0; t < depth; ++t) {
            const int node = (1 << t) - 1;
            level_feature[t] = tree.feature(node);
            level_threshold[t] = tree.threshold(node);
        }

        std::vector<std::vector<int>> groups(num_leaves);
        for (int i = 0; i < n; ++i) {
            const int leaf = route_oblivious_leaf(X, i, level_feature, level_threshold);
            groups[leaf].push_back(i);
        }

        std::vector<int> feature_ids;
        std::vector<double> T0_leaf;
        std::vector<double> T2_leaf;
        for (int leaf = 0; leaf < num_leaves; ++leaf) {
            if (groups[leaf].empty()) continue;
            catboost_leaf_T0_T2_compact(tree, leaf, p, feature_ids, T0_leaf, T2_leaf);
            for (int idx : groups[leaf]) {
                for (int contrib_idx = 0; contrib_idx < static_cast<int>(feature_ids.size()); ++contrib_idx) {
                    const int feature_id = feature_ids[contrib_idx];
                    const double value = T0_leaf[contrib_idx];
                    if (value != 0.0) {
                        T0(idx, feature_id) += value;
                    }
                }
            }
        }
    }

    return T0;
}
