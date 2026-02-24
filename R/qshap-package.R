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
#' \dontrun{
#' # Example with XGBoost
#' library(xgboost)
#' library(qshap)
#'
#' # Generate sample data
#' set.seed(42)
#' n <- 500
#' X <- matrix(rnorm(n * 5), nrow = n, ncol = 5)
#' colnames(X) <- paste0("Feature_", 1:5)
#' y <- 2 * X[,1] + 1.5 * X[,2] - 0.8 * X[,3] + rnorm(n, 0, 0.5)
#'
#' # Train model
#' model <- xgboost(data = X, label = y, nrounds = 50, max_depth = 3, verbose = 0)
#'
#' # Create explainer and calculate R-squared values
#' explainer <- gazer(model)
#' phi_rsq <- rsq(explainer, X, y)
#'
#' # Visualize
#' vis$rsq(phi_rsq, label = colnames(X))
#' }
#'
#' @docType package
#' @name qshap
#' @keywords internal
"_PACKAGE"