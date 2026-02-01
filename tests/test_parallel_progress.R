# Test script for parallel processing with progress bar
# This script tests the new progress bar functionality for parallel computing

library(qshapr)

# Suppress library loading messages in tests
suppressPackageStartupMessages({
  if (requireNamespace("xgboost", quietly = TRUE)) {
    library(xgboost)
  } else {
    stop("xgboost package is required for this test")
  }
})

cat("========================================\n")
cat("Testing Parallel Processing Progress Bar\n")
cat("========================================\n\n")

# Generate test data
set.seed(42)
n <- 200
p <- 8
X <- matrix(rnorm(n * p), nrow = n, ncol = p)
colnames(X) <- paste0("Feature_", 1:p)
y <- 2 * X[,1] + 1.5 * X[,2] - 0.8 * X[,3] + rnorm(n, 0, 0.5)

cat("Test data created:\n")
cat(sprintf("  - %d samples\n", n))
cat(sprintf("  - %d features\n", p))
cat("\n")

# Train a simple XGBoost model
cat("Training XGBoost model...\n")
model <- xgboost(
  data = X, 
  label = y, 
  nrounds = 30,  # More trees to make progress visible
  max_depth = 3, 
  verbose = 0
)
cat("Model trained successfully\n\n")

# Create explainer
cat("Creating Q-SHAP explainer...\n")
explainer <- gazer(model)
print(explainer)
cat("\n")

# Test 1: Serial processing (baseline)
cat("Test 1: Serial processing (ncore = 1)\n")
cat("======================================\n")
time_serial <- system.time({
  phi_rsq_serial <- qshap_rsq(explainer, X, y, ncore = 1)
})
cat(sprintf("\nCompleted in %.2f seconds\n", time_serial[3]))
cat("Feature-specific R²:\n")
print(round(phi_rsq_serial, 4))
cat(sprintf("Total R²: %.4f\n\n", sum(phi_rsq_serial)))

# Test 2: Parallel processing with 2 cores
cat("Test 2: Parallel processing (ncore = 2)\n")
cat("========================================\n")
time_parallel_2 <- system.time({
  phi_rsq_parallel_2 <- qshap_rsq(explainer, X, y, ncore = 2)
})
cat(sprintf("\nCompleted in %.2f seconds\n", time_parallel_2[3]))
cat("Feature-specific R²:\n")
print(round(phi_rsq_parallel_2, 4))
cat(sprintf("Total R²: %.4f\n\n", sum(phi_rsq_parallel_2)))

# Test 3: Parallel processing with 4 cores (or all available)
cat("Test 3: Parallel processing (ncore = 4)\n")
cat("========================================\n")
time_parallel_4 <- system.time({
  phi_rsq_parallel_4 <- qshap_rsq(explainer, X, y, ncore = 4)
})
cat(sprintf("\nCompleted in %.2f seconds\n", time_parallel_4[3]))
cat("Feature-specific R²:\n")
print(round(phi_rsq_parallel_4, 4))
cat(sprintf("Total R²: %.4f\n\n", sum(phi_rsq_parallel_4)))

# Verify results are consistent
cat("Verification:\n")
cat("=============\n")
max_diff_2 <- max(abs(phi_rsq_serial - phi_rsq_parallel_2))
max_diff_4 <- max(abs(phi_rsq_serial - phi_rsq_parallel_4))
cat(sprintf("Max difference (serial vs parallel 2 cores): %.10f\n", max_diff_2))
cat(sprintf("Max difference (serial vs parallel 4 cores): %.10f\n", max_diff_4))

tolerance <- 1e-10
if (max_diff_2 < tolerance && max_diff_4 < tolerance) {
  cat("✓ Results are consistent across serial and parallel processing!\n")
} else {
  cat("✗ Warning: Results differ between serial and parallel processing\n")
}

cat("\n========================================\n")
cat("All tests completed successfully!\n")
cat("========================================\n")
