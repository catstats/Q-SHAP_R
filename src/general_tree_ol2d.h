#ifndef QSHAP_GENERAL_TREE_OL2D_HPP
#define QSHAP_GENERAL_TREE_OL2D_HPP

#include <RcppEigen.h>

// Experimental general-tree O(L^2 D)
// product-tree Q-SHAP quadratic backend. The stable T2/T2_sample
// implementation remains the default; use this path for comparison.

Eigen::MatrixXd T2_ol2d(
    const Eigen::MatrixXd& x,
    const Rcpp::List& tree_summary,
    const Eigen::MatrixXcd& store_v_invc,
    const Eigen::MatrixXcd& store_z,
    bool parallel
);

#endif
