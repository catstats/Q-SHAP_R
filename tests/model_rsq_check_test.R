# Test script for model R² validation check
# This test verifies that the qshap_rsq function correctly validates
# the sum of Q-SHAP R² contributions against the model's actual R²

library(lightgbm)
library(qshapr)

cat("=== Model R² Validation Check Test ===\n\n")

# Create synthetic data
set.seed(42)
n_samples <- 100
X <- matrix(runif(n_samples * 3), n_samples, 3)
y <- X[,1] + 2 * X[,2] + 0.5 * X[,3] + rnorm(n_samples, sd=0.1)

# Train LightGBM model
dtrain <- lgb.Dataset(data = X, label = y)
model <- lgb.train(
  params = list(
    objective = "regression", 
    num_leaves = 15, 
    min_data_in_leaf = 5,
    verbose = -1
  ),
  data = dtrain,
  nrounds = 10,
  verbose = -1
)

cat("Test 1: Default behavior with validation enabled\n")
cat("------------------------------------------------\n")
explainer <- create_tree_explainer(model)

# This should pass validation without warning
rsq_contributions <- qshap_rsq(explainer, X, y)

# Calculate true model R² for comparison
ypred <- predict(model, X)
sst <- sum((y - mean(y))^2)
sse <- sum((y - ypred)^2)
true_rsq <- 1 - sse/sst

cat("Model R²:", true_rsq, "\n")
cat("Q-SHAP R² sum:", sum(rsq_contributions), "\n")
cat("Difference:", abs(sum(rsq_contributions) - true_rsq), "\n\n")

cat("Test 2: Validation disabled\n")
cat("---------------------------\n")
# This should work without validation
rsq_no_check <- qshap_rsq(explainer, X, y, check_model_rsq = FALSE)
cat("Q-SHAP R² sum (no check):", sum(rsq_no_check), "\n\n")

cat("Test 3: Custom tolerance\n")
cat("------------------------\n")
# This should work with a looser tolerance
rsq_loose <- qshap_rsq(explainer, X, y, check_model_rsq = TRUE, tolerance = 1e-4)
cat("Q-SHAP R² sum (loose tolerance):", sum(rsq_loose), "\n\n")

cat("=== Test completed successfully ===\n")
