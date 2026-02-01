#ifndef QSHAPR_UTILS_H
#define QSHAPR_UTILS_H

#include <RcppEigen.h>
#include <Rcpp.h>
#include <Eigen/Dense>
#include <complex>
#include <cmath>

// [[Rcpp::depends(RcppEigen)]]

struct TreeSummary {
    Eigen::VectorXi children_left;
    Eigen::VectorXi children_right;
    Eigen::VectorXi feature;
    Eigen::VectorXi feature_uniq;
    Eigen::VectorXf threshold;           // Changed from VectorXd to VectorXf for memory efficiency
    int max_depth;
    Eigen::VectorXf sample_weight;       // Changed from VectorXd to VectorXf for memory efficiency
    Eigen::VectorXf init_prediction;     // Changed from VectorXd to VectorXf for memory efficiency
    int node_count;
};

struct SimpleTree {
    Eigen::VectorXi children_left;
    Eigen::VectorXi children_right;
    Eigen::VectorXi feature;
    Eigen::VectorXf threshold;           // Changed from VectorXd to VectorXf for memory efficiency
    int max_depth;
    Eigen::VectorXf n_node_samples;      // Changed from VectorXd to VectorXf for memory efficiency
    Eigen::VectorXf value;               // Changed from VectorXd to VectorXf for memory efficiency
    int node_count;
};

TreeSummary list_to_tree_summary(const Rcpp::List& tree_summary_list);

Eigen::VectorXd inv_binom_coef(int d);

Eigen::MatrixXcd complex_v_invc_degree(int d);

// [[Rcpp::export]]
Eigen::MatrixXcd store_complex_v_invc(int d);

// [[Rcpp::export]]
Eigen::MatrixXcd store_complex_root(int d);

double complex_dot_v2(const Eigen::VectorXcd& p, const Eigen::VectorXcd& v_invc, int d);

void traversal_weight(
    const Eigen::VectorXd& x,
    int v,
    Eigen::VectorXd& w,
    const Eigen::VectorXi& children_left,
    const Eigen::VectorXi& children_right,
    const Eigen::VectorXi& feature,
    const Eigen::VectorXf& threshold,        // Changed from VectorXd to VectorXf
    const Eigen::VectorXf& sample_weight,    // Changed from VectorXd to VectorXf
    const Eigen::VectorXi& leaf_ind,
    Eigen::MatrixXd& w_res,
    Eigen::MatrixXi& w_ind,
    int depth,
    Eigen::VectorXi& met_feature
);

std::pair<Eigen::MatrixXd, Eigen::MatrixXi> weight(
    const Eigen::VectorXd& x,
    const TreeSummary& summary_tree
);

#endif
