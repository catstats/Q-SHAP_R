#ifndef QSHAP_CATBOOST_OBLIVIOUS_CORE_HPP
#define QSHAP_CATBOOST_OBLIVIOUS_CORE_HPP

#include <RcppEigen.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <vector>

struct ObliviousTree {
    Eigen::VectorXi children_left;
    Eigen::VectorXi children_right;
    Eigen::VectorXi feature;
    Eigen::VectorXd threshold;
    Eigen::VectorXd n_node_samples;
    Eigen::VectorXd value;
    int max_depth;
    int node_count;
};

inline ObliviousTree list_to_oblivious_tree(const Rcpp::List &tree) {
    ObliviousTree out;
    out.children_left = Rcpp::as<Eigen::VectorXi>(tree["children_left"]);
    out.children_right = Rcpp::as<Eigen::VectorXi>(tree["children_right"]);
    out.feature = Rcpp::as<Eigen::VectorXi>(tree["feature"]);
    out.threshold = Rcpp::as<Eigen::VectorXd>(tree["threshold"]);
    out.n_node_samples = Rcpp::as<Eigen::VectorXd>(tree["n_node_samples"]);
    out.value = Rcpp::as<Eigen::VectorXd>(tree["value"]);
    out.max_depth = Rcpp::as<int>(tree["max_depth"]);
    out.node_count = Rcpp::as<int>(tree["node_count"]);
    return out;
}

inline int popcount_int(int x) {
    int count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
}

inline double choose_double(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k == 0 || k == n) return 1.0;
    k = std::min(k, n - k);
    double out = 1.0;
    for (int i = 1; i <= k; ++i) {
        out *= static_cast<double>(n - k + i);
        out /= static_cast<double>(i);
    }
    return out;
}

inline std::vector<int> path_bits_from_leaf(int leaf_id, int depth) {
    std::vector<int> bits(depth, 0);
    for (int t = 0; t < depth; ++t) {
        bits[t] = (leaf_id >> (depth - 1 - t)) & 1;
    }
    return bits;
}

inline int route_oblivious_leaf(const Eigen::MatrixXd &x,
                                int row,
                                const std::vector<int> &level_feature,
                                const std::vector<double> &level_threshold) {
    int leaf = 0;
    const int depth = static_cast<int>(level_feature.size());
    for (int t = 0; t < depth; ++t) {
        const int f = level_feature[t];
        const int right = x(row, f) > level_threshold[t] ? 1 : 0;
        leaf = leaf * 2 + right;
    }
    return leaf;
}

inline std::vector<int> suffix_unique_features(const std::vector<int> &level_feature,
                                               int start_level) {
    std::vector<int> out;
    for (int t = static_cast<int>(level_feature.size()) - 1; t >= start_level; --t) {
        const int f = level_feature[t];
        if (std::find(out.begin(), out.end(), f) == out.end()) {
            out.push_back(f);
        }
    }
    return out;
}

inline std::unordered_map<int, int> feature_positions(const std::vector<int> &features) {
    std::unordered_map<int, int> pos;
    pos.reserve(features.size());
    for (int i = 0; i < static_cast<int>(features.size()); ++i) {
        pos[features[i]] = i;
    }
    return pos;
}

inline std::vector<int> project_masks(const std::vector<int> &curr_features,
                                      const std::vector<int> &next_features) {
    const int curr_k = static_cast<int>(curr_features.size());
    const int curr_n = 1 << curr_k;
    auto curr_pos = feature_positions(curr_features);
    std::vector<int> projection(curr_n, 0);

    for (int mask = 0; mask < curr_n; ++mask) {
        int projected = 0;
        for (int j = 0; j < static_cast<int>(next_features.size()); ++j) {
            const auto it = curr_pos.find(next_features[j]);
            if (it != curr_pos.end() && (mask & (1 << it->second))) {
                projected |= (1 << j);
            }
        }
        projection[mask] = projected;
    }

    return projection;
}

// For CatBoost symmetric trees, same tree leaf/path => same per-tree T0 and T2.
// Compact output contains only raw features used by the tree path/symmetric levels.
// T0_values[pos] is T_{0,ij}^{(k)} and T2_values[pos] is T_{2,ij}^{(k)}
// for feature_ids[pos].
inline void catboost_leaf_T0_T2_compact(const ObliviousTree &tree,
                                        int leaf_id,
                                        int n_features,
                                        std::vector<int> &feature_ids,
                                        std::vector<double> &T0_values,
                                        std::vector<double> &T2_values) {
    const int depth = tree.max_depth;
    feature_ids.clear();
    T0_values.clear();
    T2_values.clear();
    if (depth == 0) return;

    const int num_leaves = 1 << depth;
    const int num_internal = num_leaves - 1;

    std::vector<int> level_feature(depth);
    std::vector<double> level_threshold(depth);
    for (int t = 0; t < depth; ++t) {
        const int node = (1 << t) - 1;
        level_feature[t] = tree.feature(node);
        level_threshold[t] = tree.threshold(node);
    }

    std::vector<std::vector<int>> suffix_features(depth + 1);
    suffix_features[depth] = std::vector<int>();
    for (int t = depth - 1; t >= 0; --t) {
        suffix_features[t] = suffix_features[t + 1];
        const int f = level_feature[t];
        if (std::find(suffix_features[t].begin(), suffix_features[t].end(), f) ==
            suffix_features[t].end()) {
            suffix_features[t].push_back(f);
        }
    }

    std::vector<int> hot_bits = path_bits_from_leaf(leaf_id, depth);

    std::vector<std::vector<double>> dp_next(num_leaves, std::vector<double>(1));
    for (int leaf = 0; leaf < num_leaves; ++leaf) {
        dp_next[leaf][0] = tree.value(num_internal + leaf);
    }

    for (int t = depth - 1; t >= 0; --t) {
        const std::vector<int> &curr_features = suffix_features[t];
        const std::vector<int> &next_features = suffix_features[t + 1];
        const int curr_k = static_cast<int>(curr_features.size());
        const int curr_n = 1 << curr_k;
        const int nodes_this_level = 1 << t;

        auto curr_pos = feature_positions(curr_features);
        const int split_pos = curr_pos[level_feature[t]];
        const std::vector<int> projection = project_masks(curr_features, next_features);

        std::vector<std::vector<double>> dp_curr(
            nodes_this_level, std::vector<double>(curr_n, 0.0));

        for (int node_pos = 0; node_pos < nodes_this_level; ++node_pos) {
            const int bfs_node = (1 << t) - 1 + node_pos;
            const int left_bfs = tree.children_left(bfs_node);
            const int right_bfs = tree.children_right(bfs_node);
            const int left_pos = 2 * node_pos;
            const int right_pos = 2 * node_pos + 1;

            const double parent_n = std::max(tree.n_node_samples(bfs_node), 1e-300);
            const double p_left = tree.n_node_samples(left_bfs) / parent_n;
            const double p_right = tree.n_node_samples(right_bfs) / parent_n;

            for (int mask = 0; mask < curr_n; ++mask) {
                const int next_mask = projection[mask];
                const bool observed = (mask & (1 << split_pos)) != 0;
                if (observed) {
                    dp_curr[node_pos][mask] =
                        hot_bits[t] ? dp_next[right_pos][next_mask]
                                    : dp_next[left_pos][next_mask];
                } else {
                    dp_curr[node_pos][mask] =
                        p_left * dp_next[left_pos][next_mask] +
                        p_right * dp_next[right_pos][next_mask];
                }
            }
        }

        dp_next.swap(dp_curr);
    }

    const std::vector<int> &root_features = suffix_features[0];
    const int k = static_cast<int>(root_features.size());
    if (k == 0) return;

    std::vector<double> tmp_T0(k, 0.0);
    std::vector<double> tmp_T2(k, 0.0);
    const std::vector<double> &m = dp_next[0];
    for (int mask = 0; mask < (1 << k); ++mask) {
        const int s = popcount_int(mask);
        for (int bit = 0; bit < k; ++bit) {
            if (mask & (1 << bit)) continue;
            const int with_feature = mask | (1 << bit);
            const double weight = 1.0 / (static_cast<double>(k) * choose_double(k - 1, s));
            const int feature_id = root_features[bit];
            if (feature_id >= 0 && feature_id < n_features) {
                tmp_T0[bit] += weight * (m[with_feature] - m[mask]);
                tmp_T2[bit] += weight *
                    (m[with_feature] * m[with_feature] - m[mask] * m[mask]);
            }
        }
    }

    feature_ids.reserve(k);
    T0_values.reserve(k);
    T2_values.reserve(k);
    for (int bit = 0; bit < k; ++bit) {
        const int feature_id = root_features[bit];
        if (feature_id >= 0 && feature_id < n_features &&
            (tmp_T0[bit] != 0.0 || tmp_T2[bit] != 0.0)) {
            feature_ids.push_back(feature_id);
            T0_values.push_back(tmp_T0[bit]);
            T2_values.push_back(tmp_T2[bit]);
        }
    }
}

// Dense compatibility wrapper for callers that need feature-aligned vectors.
inline void catboost_leaf_T0_T2(const ObliviousTree &tree,
                                int leaf_id,
                                int n_features,
                                Eigen::VectorXd &T0_leaf,
                                Eigen::VectorXd &T2_leaf) {
    T0_leaf = Eigen::VectorXd::Zero(n_features);
    T2_leaf = Eigen::VectorXd::Zero(n_features);

    std::vector<int> feature_ids;
    std::vector<double> T0_values;
    std::vector<double> T2_values;
    catboost_leaf_T0_T2_compact(tree, leaf_id, n_features, feature_ids, T0_values, T2_values);

    for (int idx = 0; idx < static_cast<int>(feature_ids.size()); ++idx) {
        const int feature_id = feature_ids[idx];
        T0_leaf(feature_id) = T0_values[idx];
        T2_leaf(feature_id) = T2_values[idx];
    }
}

#endif
