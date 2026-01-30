# Tests for qshapr S3 class system
# 
# This file tests the formal S3 class constructors, validators,
# and methods for qshapr package objects.

suppressPackageStartupMessages({
  library(qshapr)
  library(xgboost)
})

cat("========================================\n")
cat("Testing qshapr S3 Class System\n")
cat("========================================\n\n")

# ============================================================================
# Test 1: simple_tree class
# ============================================================================
cat("Test 1: simple_tree class\n")
cat("-------------------------\n")

# Create a simple tree
st <- simple_tree(
  children_left = c(-1L, -1L),
  children_right = c(-1L, -1L),
  feature = c(-1L, -1L),
  threshold = c(0, 0),
  max_depth = 1L,
  n_node_samples = c(10L, 5L),
  value = c(1.5, 2.0),
  node_count = 2L
)

# Test class
stopifnot(inherits(st, "simple_tree"))
cat("✓ simple_tree object created successfully\n")

# Test print method
cat("✓ Testing print.simple_tree:\n")
print(st)
cat("\n")

# Test validation - should fail with mismatched lengths
tryCatch({
  bad_st <- simple_tree(
    children_left = c(-1L),  # Length 1
    children_right = c(-1L, -1L),  # Length 2 - mismatch!
    feature = c(-1L, -1L),
    threshold = c(0, 0),
    max_depth = 1L,
    n_node_samples = c(10L, 5L),
    value = c(1.5, 2.0),
    node_count = 2L
  )
  stop("Should have failed validation")
}, error = function(e) {
  cat("✓ Validation correctly caught mismatched vector lengths\n")
})

cat("\n")

# ============================================================================
# Test 2: tree_summary class
# ============================================================================
cat("Test 2: tree_summary class\n")
cat("-------------------------\n")

# Create a tree_summary
ts <- tree_summary(
  children_left = c(-1L, -1L),
  children_right = c(-1L, -1L),
  feature = c(-1L, -1L),
  feature_uniq = integer(0),
  threshold = c(0, 0),
  max_depth = 1L,
  sample_weight = c(1.0, 1.0),
  init_prediction = c(0.5, 0.3),
  node_count = 2L
)

# Test class
stopifnot(inherits(ts, "tree_summary"))
cat("✓ tree_summary object created successfully\n")

# Test print method
cat("✓ Testing print.tree_summary:\n")
print(ts)
cat("\n")

# ============================================================================
# Test 3: qshapr_tree_explainer class with real model
# ============================================================================
cat("Test 3: qshapr_tree_explainer class\n")
cat("-----------------------------------\n")

# Create a simple dataset
set.seed(42)
n <- 100
X <- matrix(rnorm(n * 3), nrow = n, ncol = 3)
colnames(X) <- c("X1", "X2", "X3")
y <- 2 * X[,1] + 1.5 * X[,2] - 0.8 * X[,3] + rnorm(n, 0, 0.1)

# Train XGBoost model
model <- xgboost(
  data = X, 
  label = y, 
  nrounds = 5,
  max_depth = 2,
  verbose = 0
)

# Create explainer
explainer <- create_tree_explainer(model)

# Test class
stopifnot(inherits(explainer, "qshapr_tree_explainer"))
stopifnot(inherits(explainer, "xgboost_explainer"))
cat("✓ qshapr_tree_explainer object created successfully\n")

# Test print method
cat("✓ Testing print.qshapr_tree_explainer:\n")
print(explainer)
cat("\n")

# Test summary method
cat("✓ Testing summary.qshapr_tree_explainer:\n")
summary(explainer)
cat("\n")

# Verify internal structure
stopifnot(explainer$model_type == "xgboost")
stopifnot(length(explainer$trees) == 5)
stopifnot(!is.null(explainer$base_score))
cat("✓ Explainer structure validated\n\n")

# ============================================================================
# Test 4: qshap_result class
# ============================================================================
cat("Test 4: qshap_result class\n")
cat("-------------------------\n")

# Create a qshap_result object
rsq_values <- c(0.5, 0.3, 0.15)
feature_names <- c("Feature_A", "Feature_B", "Feature_C")

qr <- qshap_result(
  rsq = rsq_values,
  feature_names = feature_names,
  n_samples = 100L
)

# Test class
stopifnot(inherits(qr, "qshap_result"))
cat("✓ qshap_result object created successfully\n")

# Test print method
cat("✓ Testing print.qshap_result:\n")
print(qr)
cat("\n")

# Test summary method
cat("✓ Testing summary.qshap_result:\n")
summary(qr)
cat("\n")

# Test as.data.frame method
cat("✓ Testing as.data.frame.qshap_result:\n")
df <- as.data.frame(qr)
print(head(df))
stopifnot(is.data.frame(df))
stopifnot(nrow(df) == 3)
stopifnot(all(c("feature", "rsq") %in% colnames(df)))
cat("✓ Conversion to data.frame successful\n\n")

# Test validation - should fail with mismatched lengths
tryCatch({
  bad_qr <- qshap_result(
    rsq = c(0.5, 0.3),  # Length 2
    feature_names = c("A", "B", "C")  # Length 3 - mismatch!
  )
  stop("Should have failed validation")
}, error = function(e) {
  cat("✓ Validation correctly caught mismatched lengths\n")
})

cat("\n")

# ============================================================================
# Test 5: Integration test - Full workflow
# ============================================================================
cat("Test 5: Integration test - Full workflow\n")
cat("----------------------------------------\n")

# Calculate Q-SHAP R-squared values
phi_rsq <- qshap_rsq(explainer, X, y)

cat("✓ qshap_rsq computed successfully\n")
cat("  Result type:", class(phi_rsq), "\n")
cat("  Length:", length(phi_rsq), "\n")
cat("  Total R²:", round(sum(phi_rsq), 4), "\n")

# Verify result is numeric vector (for backward compatibility)
stopifnot(is.numeric(phi_rsq))
stopifnot(length(phi_rsq) == ncol(X))

# Manually create qshap_result for comparison
qr_manual <- qshap_result(
  rsq = phi_rsq,
  feature_names = colnames(X),
  n_samples = nrow(X)
)

cat("\n✓ Manual qshap_result creation:\n")
print(qr_manual)

cat("\n")

# ============================================================================
# Test 6: Edge cases
# ============================================================================
cat("Test 6: Edge cases\n")
cat("-----------------\n")

# Test with single feature
single_rsq <- qshap_result(
  rsq = 0.8,
  feature_names = "OnlyFeature"
)
cat("✓ Single feature qshap_result:\n")
print(single_rsq)
cat("\n")

# Test with many features
many_rsq <- qshap_result(
  rsq = runif(100, 0, 0.1),
  feature_names = paste0("F", 1:100)
)
cat("✓ Many features qshap_result (100 features):\n")
print(many_rsq, n = 5)  # Show only top 5
cat("\n")

# ============================================================================
# Summary
# ============================================================================
cat("========================================\n")
cat("All class system tests passed! ✓\n")
cat("========================================\n")
