#include "general_tree_ol2d.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <unordered_map>
#include <vector>

// [[Rcpp::depends(RcppEigen)]]

namespace {

using cd = std::complex<double>;

struct ProductNode {
    int left = -1;
    int right = -1;
    int feature = -1;
    double threshold = 0.0;
    int default_left = 0;
    bool xgboost_split = false;
    double edge_prob = 1.0;
    double leaf_pred = 0.0;
};

struct ProductTree {
    std::vector<int> children_left;
    std::vector<int> children_right;
    std::vector<int> feature;
    std::vector<int> repeated_parent;
    std::vector<int> edge_heights;
    std::vector<double> weights;
    std::vector<double> leaf_pred;
    std::vector<double> threshold;
    std::vector<int> default_left;
    std::vector<char> xgboost_split;
    int max_depth = 0;
    int poly_depth = 0;
    int num_nodes = 0;
    int n_features = 0;
};

struct RootCache {
    int maxd = 0;
    std::vector<cd> base;
    std::vector<cd> offset;
    std::vector<cd> norm;
};

struct VecBoolHash {
    size_t operator()(const std::vector<bool> &v) const {
        size_t seed = v.size();
        for (size_t i = 0; i < v.size(); ++i) {
            if (v[i]) {
                seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
        }
        return seed;
    }
};

inline bool is_leaf(const TreeSummary &tree, int node) {
    return tree.children_left(node) < 0;
}

inline double branch_prob_from_inverse_weight(double inverse_weight) {
    if (!std::isfinite(inverse_weight) || inverse_weight <= 0.0) {
        return 0.0;
    }
    return 1.0 / inverse_weight;
}

int build_product_rec(const TreeSummary &tree,
                      int u,
                      int v,
                      std::vector<ProductNode> &nodes) {
    const int id = static_cast<int>(nodes.size());
    nodes.push_back(ProductNode());

    const bool u_leaf = is_leaf(tree, u);
    const bool v_leaf = is_leaf(tree, v);
    if (u_leaf && v_leaf) {
        nodes[id].leaf_pred = tree.init_prediction(u) * tree.init_prediction(v);
        return id;
    }

    const bool choose_u = !u_leaf;
    const int split_node = choose_u ? u : v;
    nodes[id].feature = tree.feature(split_node);
    nodes[id].threshold = tree.threshold(split_node);
    nodes[id].default_left = tree.default_left(split_node);
    nodes[id].xgboost_split = tree.xgboost_split;

    if (choose_u) {
        const int left_u = tree.children_left(u);
        const int right_u = tree.children_right(u);
        const int left_id = build_product_rec(tree, left_u, v, nodes);
        const int right_id = build_product_rec(tree, right_u, v, nodes);
        nodes[id].left = left_id;
        nodes[id].right = right_id;
        nodes[left_id].edge_prob = branch_prob_from_inverse_weight(tree.sample_weight(left_u));
        nodes[right_id].edge_prob = branch_prob_from_inverse_weight(tree.sample_weight(right_u));
    } else {
        const int left_v = tree.children_left(v);
        const int right_v = tree.children_right(v);
        const int left_id = build_product_rec(tree, u, left_v, nodes);
        const int right_id = build_product_rec(tree, u, right_v, nodes);
        nodes[id].left = left_id;
        nodes[id].right = right_id;
        nodes[left_id].edge_prob = branch_prob_from_inverse_weight(tree.sample_weight(left_v));
        nodes[right_id].edge_prob = branch_prob_from_inverse_weight(tree.sample_weight(right_v));
    }

    return id;
}

int max_path_depth(const std::vector<ProductNode> &nodes, int node = 0) {
    if (nodes[node].left < 0) {
        return 0;
    }
    return 1 + std::max(max_path_depth(nodes, nodes[node].left),
                        max_path_depth(nodes, nodes[node].right));
}

int annotate_rec(const std::vector<ProductNode> &nodes,
                 int node,
                 int incoming_feature,
                 std::unordered_map<int, int> seen,
                 ProductTree &tree) {
    if (incoming_feature >= 0) {
        double weight = nodes[node].edge_prob;
        const auto it = seen.find(incoming_feature);
        if (it != seen.end()) {
            tree.repeated_parent[node] = it->second;
            weight *= tree.weights[it->second];
        }
        tree.weights[node] = weight;
        seen[incoming_feature] = node;
    }

    if (nodes[node].left < 0) {
        tree.edge_heights[node] = static_cast<int>(seen.size());
        return tree.edge_heights[node];
    }

    const int left_height = annotate_rec(
        nodes, nodes[node].left, nodes[node].feature, seen, tree);
    const int right_height = annotate_rec(
        nodes, nodes[node].right, nodes[node].feature, seen, tree);
    tree.edge_heights[node] = std::max(left_height, right_height);
    return tree.edge_heights[node];
}

ProductTree make_product_tree(const TreeSummary &summary_tree, int n_features) {
    std::vector<ProductNode> nodes;
    nodes.reserve(static_cast<size_t>(summary_tree.node_count) *
                  static_cast<size_t>(summary_tree.node_count));
    build_product_rec(summary_tree, 0, 0, nodes);

    ProductTree tree;
    tree.num_nodes = static_cast<int>(nodes.size());
    tree.n_features = n_features;
    tree.max_depth = max_path_depth(nodes);
    tree.children_left.resize(tree.num_nodes);
    tree.children_right.resize(tree.num_nodes);
    tree.feature.resize(tree.num_nodes);
    tree.repeated_parent.assign(tree.num_nodes, -1);
    tree.edge_heights.assign(tree.num_nodes, 0);
    tree.weights.assign(tree.num_nodes, 1.0);
    tree.leaf_pred.assign(tree.num_nodes, 0.0);
    tree.threshold.resize(tree.num_nodes);
    tree.default_left.resize(tree.num_nodes);
    tree.xgboost_split.resize(tree.num_nodes);

    for (int i = 0; i < tree.num_nodes; ++i) {
        tree.children_left[i] = nodes[i].left;
        tree.children_right[i] = nodes[i].right;
        tree.feature[i] = nodes[i].feature;
        tree.threshold[i] = nodes[i].threshold;
        tree.default_left[i] = nodes[i].default_left;
        tree.xgboost_split[i] = nodes[i].xgboost_split ? 1 : 0;
        tree.leaf_pred[i] = nodes[i].leaf_pred;
    }

    std::unordered_map<int, int> seen;
    annotate_rec(nodes, 0, -1, seen, tree);
    tree.poly_depth = tree.edge_heights[0];
    return tree;
}

RootCache precompute_complex_root_cache(int maxd) {
    RootCache cache;
    cache.maxd = maxd;
    if (maxd <= 0) {
        return cache;
    }

    cache.base.assign(maxd, cd(0.0, 0.0));
    for (int k = 0; k < maxd; ++k) {
        cache.base[k] = std::polar(1.0, 2.0 * M_PI * k / maxd);
    }

    cache.offset.assign(static_cast<size_t>(maxd + 1) * maxd, cd(0.0, 0.0));
    for (int degree = 0; degree <= maxd; ++degree) {
        for (int k = 0; k < maxd; ++k) {
            cache.offset[static_cast<size_t>(degree) * maxd + k] =
                std::pow(cache.base[k] + cd(1.0, 0.0), degree);
        }
    }

    cache.norm.assign(static_cast<size_t>(maxd + 1) * maxd, cd(0.0, 0.0));
    for (int degree = 1; degree <= maxd; ++degree) {
        Eigen::MatrixXcd vander(degree, degree);
        for (int row = 0; row < degree; ++row) {
            const int power = degree - 1 - row;
            for (int col = 0; col < degree; ++col) {
                vander(row, col) = std::pow(cache.base[col], power);
            }
        }

        const Eigen::VectorXd inv_binom = inv_binom_coef(degree - 1);
        Eigen::VectorXcd rhs(degree);
        for (int i = 0; i < degree; ++i) {
            rhs(i) = cd(inv_binom(i), 0.0);
        }

        const Eigen::VectorXcd solved = vander.fullPivLu().solve(rhs);
        for (int i = 0; i < degree; ++i) {
            cache.norm[static_cast<size_t>(degree) * maxd + i] = solved(i);
        }
    }

    return cache;
}

inline cd safe_div_or_zero(cd numerator, cd denominator) {
    if (std::norm(denominator) < 1e-24) {
        return cd(0.0, 0.0);
    }
    return numerator / denominator;
}

double psi_roots(const std::vector<cd> &E,
                 int e_offset,
                 const RootCache &cache,
                 int offset_degree,
                 double q,
                 int degree) {
    if (degree <= 0) {
        return 0.0;
    }

    cd res(0.0, 0.0);
    const cd qc(q, 0.0);
    const int maxd = cache.maxd;
    for (int k = 0; k < degree; ++k) {
        const cd numerator =
            E[e_offset + k] *
            cache.offset[static_cast<size_t>(offset_degree) * maxd + k];
        const cd denom = cache.base[k] + qc;
        res += safe_div_or_zero(numerator, denom) *
               cache.norm[static_cast<size_t>(degree) * maxd + k];
    }
    return (res / static_cast<double>(degree)).real();
}

void inference_rec(const ProductTree &tree,
                   const RootCache &cache,
                   const Eigen::VectorXd &x,
                   std::vector<char> &activation,
                   std::vector<double> &value,
                   std::vector<cd> &C,
                   std::vector<cd> &E,
                   int node = 0,
                   int edge_feature = -1,
                   int depth = 0) {
    const int repeated_parent = tree.repeated_parent[node];
    double s = 0.0;
    if (repeated_parent >= 0) {
        activation[node] = activation[node] & activation[repeated_parent];
        if (activation[repeated_parent]) {
            s = 1.0 / tree.weights[repeated_parent];
        }
    }

    const int maxd = cache.maxd;
    const int curE = depth * maxd;
    const int childE = (depth + 1) * maxd;
    const int curC = depth * maxd;
    double q = 0.0;

    if (edge_feature >= 0) {
        if (activation[node]) {
            q = 1.0 / tree.weights[node];
        }
        const int prevC = (depth - 1) * maxd;
        for (int k = 0; k < maxd; ++k) {
            C[curC + k] = C[prevC + k] * (cache.base[k] + q);
        }
        if (repeated_parent >= 0) {
            for (int k = 0; k < maxd; ++k) {
                C[curC + k] =
                    safe_div_or_zero(C[curC + k], cache.base[k] + s);
            }
        }
    }

    const int left = tree.children_left[node];
    const int right = tree.children_right[node];
    int offset_degree = 0;

    if (left >= 0) {
        const int f = tree.feature[node];
        if (tree_goes_left(x(f), tree.threshold[node],
                           tree.default_left[node],
                           tree.xgboost_split[node] != 0)) {
            activation[left] = 1;
            activation[right] = 0;
        } else {
            activation[left] = 0;
            activation[right] = 1;
        }

        inference_rec(tree, cache, x, activation, value, C, E,
                      left, f, depth + 1);
        offset_degree = tree.edge_heights[node] - tree.edge_heights[left];
        for (int k = 0; k < maxd; ++k) {
            E[childE + k] *=
                cache.offset[static_cast<size_t>(offset_degree) * maxd + k];
            E[curE + k] = E[childE + k];
        }

        inference_rec(tree, cache, x, activation, value, C, E,
                      right, f, depth + 1);
        offset_degree = tree.edge_heights[node] - tree.edge_heights[right];
        for (int k = 0; k < maxd; ++k) {
            E[childE + k] *=
                cache.offset[static_cast<size_t>(offset_degree) * maxd + k];
            E[curE + k] += E[childE + k];
        }
    } else {
        for (int k = 0; k < maxd; ++k) {
            E[curE + k] = C[curC + k] * tree.leaf_pred[node];
        }
    }

    if (edge_feature >= 0) {
        if (repeated_parent >= 0 && !activation[repeated_parent]) {
            return;
        }

        const int normal_d = tree.edge_heights[node];
        value[edge_feature] +=
            (q - 1.0) * psi_roots(E, curE, cache, 0, q, normal_d);

        if (repeated_parent >= 0) {
            offset_degree =
                tree.edge_heights[repeated_parent] - tree.edge_heights[node];
            const int parent_d = tree.edge_heights[repeated_parent];
            value[edge_feature] -=
                (s - 1.0) *
                psi_roots(E, curE, cache, offset_degree, s, parent_d);
        }
    }
}

Eigen::VectorXd product_tree_t2_sample(const ProductTree &tree,
                                       const RootCache &cache,
                                       const Eigen::VectorXd &x) {
    Eigen::VectorXd out = Eigen::VectorXd::Zero(tree.n_features);
    if (cache.maxd <= 0) {
        return out;
    }

    std::vector<double> value(tree.n_features, 0.0);
    std::vector<char> activation(tree.num_nodes, 0);
    std::vector<cd> C(static_cast<size_t>(tree.max_depth + 2) * cache.maxd,
                      cd(1.0, 0.0));
    std::vector<cd> E(static_cast<size_t>(tree.max_depth + 2) * cache.maxd,
                      cd(0.0, 0.0));

    inference_rec(tree, cache, x, activation, value, C, E);

    for (int j = 0; j < tree.n_features; ++j) {
        out(j) = value[j];
    }
    return out;
}

std::vector<bool> decision_signature(const Eigen::VectorXd &x,
                                     const TreeSummary &tree) {
    std::vector<bool> sig(tree.node_count, false);
    for (int node = 0; node < tree.node_count; ++node) {
        if (tree.children_left(node) >= 0) {
            sig[node] = tree_goes_left(
                x(tree.feature(node)), tree.threshold(node),
                tree.default_left(node), tree.xgboost_split);
        }
    }
    return sig;
}

} // namespace

// [[Rcpp::export]]
Eigen::MatrixXd T2_ol2d(const Eigen::MatrixXd &x,
                        const Rcpp::List &tree_summary,
                        const Eigen::MatrixXcd &store_v_invc,
                        const Eigen::MatrixXcd &store_z,
                        bool parallel = false) {
    (void)store_v_invc;
    (void)store_z;
    (void)parallel;

    const TreeSummary summary_tree = list_to_tree_summary(tree_summary);
    const int n = x.rows();
    const int p = x.cols();
    Eigen::MatrixXd shap_value = Eigen::MatrixXd::Zero(n, p);

    const ProductTree product_tree = make_product_tree(summary_tree, p);
    if (product_tree.poly_depth <= 0) {
        return shap_value;
    }
    const RootCache root_cache = precompute_complex_root_cache(product_tree.poly_depth);

    std::unordered_map<std::vector<bool>, std::vector<int>, VecBoolHash> groups;
    groups.reserve(n);
    for (int i = 0; i < n; ++i) {
        groups[decision_signature(x.row(i), summary_tree)].push_back(i);
    }

    for (const auto &entry : groups) {
        const int representative = entry.second.front();
        shap_value.row(representative) =
            product_tree_t2_sample(product_tree, root_cache, x.row(representative));
        for (size_t k = 1; k < entry.second.size(); ++k) {
            shap_value.row(entry.second[k]) = shap_value.row(representative);
        }
    }

    return shap_value;
}
