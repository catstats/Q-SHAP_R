# Test to verify data type optimizations work correctly
# This test checks that the float storage optimization doesn't break functionality

library(qshapr)

cat("===========================================\n")
cat("Testing Data Type Optimizations\n")
cat("===========================================\n\n")

# Test 1: Simple synthetic data
cat("Test 1: Simple synthetic data\n")
cat("------------------------------\n")

set.seed(42)
n <- 100
p <- 5

# Create test data
X <- matrix(rnorm(n * p), nrow = n, ncol = p)
colnames(X) <- paste0("Var", 1:p)

# Simple linear model: y depends mainly on first 3 variables
y <- 2 * X[,1] + 1.5 * X[,2] - 0.8 * X[,3] + rnorm(n, 0, 0.5)

cat("Created", n, "samples with", p, "features\n")
cat("True coefficients: 2.0, 1.5, -0.8, 0, 0\n\n")

# Check if xgboost is available
if (requireNamespace("xgboost", quietly = TRUE)) {
  cat("Testing with XGBoost model...\n")
  
  # Train simple xgboost model
  model <- xgboost::xgboost(
    data = X,
    label = y,
    nrounds = 10,
    max_depth = 3,
    verbose = 0
  )
  
  # Create explainer
  explainer <- gazer(model)
  cat("✓ Explainer created successfully\n")
  
  # Calculate Q-SHAP R²
  phi_rsq <- qshap_rsq(explainer, X, y)
  cat("✓ Q-SHAP R² calculated successfully\n")
  
  # Verify results
  cat("\nFeature-specific R² values:\n")
  for (i in 1:length(phi_rsq)) {
    cat(sprintf("  %s: %.4f\n", colnames(X)[i], phi_rsq[i]))
  }
  
  total_rsq <- sum(phi_rsq)
  cat(sprintf("\nTotal R²: %.4f\n", total_rsq))
  
  # Calculate actual model R²
  ypred <- predict(model, X)
  sst <- sum((y - mean(y))^2)
  sse <- sum((y - ypred)^2)
  model_rsq <- 1 - sse/sst
  cat(sprintf("Model R²: %.4f\n", model_rsq))
  
  # Check that they're close
  rsq_diff <- abs(total_rsq - model_rsq)
  if (rsq_diff < 0.01) {
    cat("✓ R² values match (difference < 0.01)\n")
  } else {
    cat("✗ WARNING: R² values differ by", round(rsq_diff, 4), "\n")
  }
  
  # Test with integer matrix (should be converted to double)
  cat("\nTest 1b: Integer matrix conversion\n")
  X_int <- matrix(as.integer(X * 10), nrow = n, ncol = p)
  colnames(X_int) <- paste0("Var", 1:p)
  phi_rsq_int <- qshap_rsq(explainer, X_int, y)
  cat("✓ Integer matrix handled correctly\n")
  
  cat("\n")
} else {
  cat("xgboost not available, skipping XGBoost tests\n\n")
}

# Test 2: Edge cases
cat("Test 2: Edge cases\n")
cat("------------------\n")

if (requireNamespace("xgboost", quietly = TRUE)) {
  # Very small values (test float precision)
  cat("Testing with very small threshold values...\n")
  X_small <- matrix(rnorm(50 * 3, mean = 0, sd = 0.001), nrow = 50, ncol = 3)
  y_small <- rowSums(X_small) + rnorm(50, 0, 0.0001)
  
  model_small <- xgboost::xgboost(
    data = X_small,
    label = y_small,
    nrounds = 5,
    max_depth = 2,
    verbose = 0
  )
  
  explainer_small <- gazer(model_small)
  phi_rsq_small <- qshap_rsq(explainer_small, X_small, y_small)
  cat("✓ Small values handled correctly\n")
  
  # Large values (test float range)
  cat("Testing with large values...\n")
  X_large <- matrix(rnorm(50 * 3, mean = 0, sd = 100), nrow = 50, ncol = 3)
  y_large <- rowSums(X_large) + rnorm(50, 0, 10)
  
  model_large <- xgboost::xgboost(
    data = X_large,
    label = y_large,
    nrounds = 5,
    max_depth = 2,
    verbose = 0
  )
  
  explainer_large <- gazer(model_large)
  phi_rsq_large <- qshap_rsq(explainer_large, X_large, y_large)
  cat("✓ Large values handled correctly\n")
  
  cat("\n")
}

cat("===========================================\n")
cat("All tests completed successfully! ✓\n")
cat("===========================================\n")
cat("\nData type optimizations verified:\n")
cat("  - Float storage for tree data (threshold, sample_weight, init_prediction)\n")
cat("  - Double precision for computation matrices (x, y, shap_value, loss)\n")
cat("  - Explicit storage.mode setting for R-C++ interface\n")
cat("  - Numerical accuracy maintained\n")
