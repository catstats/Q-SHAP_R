#' Calculating Feature-Specific R-Squared Values for Boosting Trees
#'
#' The qshap package computes feature-specific R-squared values using Shapley
#' decomposition of the total R-squared for boosting trees built in \pkg{xgboost} and \pkg{lightgbm}.
#' It supports parallel computing.
#' 
#'
#' @details
#' The package provides fast computation of feature importance through Shapley
#' values for tree ensemble models. Main functions include:
#'
#' \itemize{
#'   \item \code{gazer()}: Create a Q-SHAP explainer from a trained model
#'   \item \code{rsq()}: Calculate feature-specific R-squared values
#'   \item \code{loss()}: Calculate feature-specific loss contributions
#'   \item \code{plot()}: Visualize R-squared values
#' }
#'
#' The method uses polynomial-time complexity for Shapley value calculation and
#' includes built-in support for multi-core processing.
#'
#' @author
#' Steven He, Zhongli Jiang, Min Zhang, Dabao Zhang
#'
#' @references
#' Zhongli Jiang, Min Zhang, and Dabao Zhang. 2025. Fast 
#' calculation of feature contributions in boosting trees. In 
#' Proceedings of the Forty-First Conference on Uncertainty in Artificial Intelligence (UAI '25), Vol. 286. JMLR.org, Article 
#' 82, 1859–1875.
#'
#' @examples
#' library(xgboost)
#' set.seed(42)
#' n <- 100
#' p <- 100
#' X <- matrix(rnorm(n * p), nrow = n, ncol = p)
#' y <- X[, 1] - X[, 2] + rnorm(n, sd = 0.2)
#' model <- xgboost(X, y, nrounds = 15, max_depth = 2, verbose = 0)
#' explainer <- gazer(model)
#' phi_rsq <- rsq(explainer, X, y)
#'
#' @docType package
#' @name qshap
#' @keywords internal
"_PACKAGE"