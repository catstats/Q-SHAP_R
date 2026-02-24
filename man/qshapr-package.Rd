\name{qshapr-package}
\alias{qshapr-package}
\alias{qshapr}
\docType{package}
\title{
Feature-Specific R-Squared Values for Boosting Trees
}
\description{
The qshapr package computes feature-specific R-squared values using Shapley 
decomposition of the total R-squared for Boosting Trees in polynomial time. 
It supports XGBoost and LightGBM models with parallel processing capabilities.
}
\details{
The package provides fast computation of feature importance through Shapley 
values for tree ensemble models. Main functions include:

\itemize{
  \item \code{gazer()}: Create a Q-SHAP explainer from a trained model
  \item \code{rsq()}: Calculate feature-specific R-squared values
  \item \code{qshap_loss()}: Calculate feature-specific loss contributions
  \item \code{vis$rsq()}: Visualize R-squared values
  \item \code{vis$elbow()}: Create elbow plot of top features
  \item \code{vis$cumu()}: Plot cumulative explained variance
}

The method uses polynomial time complexity for Shapley value calculation and 
includes built-in support for multi-core processing.
}
\author{
Zhongli Jiang, Min Zhang, Dabao Zhang

Maintainer: Zhongli Jiang <zhongli.jiang@example.com>
}
\references{
Jiang, Z., Zhang, M., & Zhang, D. (2024). Fast Calculation of Feature 
Contributions in Boosting Trees. In The 41st Conference on Uncertainty in 
Artificial Intelligence.
}
\keyword{package}

\examples{
\dontrun{
# Example with XGBoost
library(xgboost)
library(qshapr)

# Generate sample data
set.seed(42)
n <- 500
X <- matrix(rnorm(n * 5), nrow = n, ncol = 5)
colnames(X) <- paste0("Feature_", 1:5)
y <- 2 * X[,1] + 1.5 * X[,2] - 0.8 * X[,3] + rnorm(n, 0, 0.5)

# Train model
model <- xgboost(data = X, label = y, nrounds = 50, max_depth = 3, verbose = 0)

# Create explainer and calculate R-squared values
explainer <- gazer(model)
phi_rsq <- rsq(explainer, X, y)

# Visualize
vis$rsq(phi_rsq, label = colnames(X))
}
}
