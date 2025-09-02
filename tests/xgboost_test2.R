library(xgboost)
library(qshapr)

# Create synthetic data (same as LightGBM test)
set.seed(0)
n_samples <- 100
X <- matrix(runif(n_samples * 3), n_samples, 3)
y <- X[,1] + 2 * X[,2] + 0.5 * X[,3] + rnorm(n_samples, sd=0.1)

# Train XGBoost model with optimal parameters
model <- xgboost(
  data = X,
  label = y,
  nrounds = 1,
  max_depth = 3,   # Optimal depth for Q-SHAP accuracy
  eta = 1.0,       # Higher learning rate
  reg_lambda = 0,  # No regularization
  verbose = 0
)

# Calculate true model R²
ypred <- predict(model, X)
true_rsq <- 1 - sum((y - ypred)^2) / sum((y - mean(y))^2)

# Calculate Q-SHAP R² contributions
explainer <- qshapr::create_tree_explainer(model)
rsq_contributions <- qshapr::qshap_rsq(explainer, X, y)

# Show the two numbers that should match
print(paste("Q-SHAP R² sum:", sum(rsq_contributions)))
print(paste("True Model R²:", true_rsq))