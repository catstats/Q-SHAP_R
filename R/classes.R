#' S3 Class Constructors and Methods for qshap
#'
#' This file contains formal S3 class definitions, constructors, validators,
#' and methods for the qshap package objects.
#'
#' @name qshap-classes
#' @keywords internal
NULL

# ============================================================================
# simple_tree class
# ============================================================================

#' Constructor for simple_tree class
#' 
#' Creates a simple_tree object with validation
#' 
#' @param children_left Integer vector of left child indices (-1 for leaf nodes)
#' @param children_right Integer vector of right child indices (-1 for leaf nodes)
#' @param feature Integer vector of feature indices used for splitting (-1 for leaf nodes)
#' @param threshold Numeric vector of threshold values for splits
#' @param max_depth Integer maximum depth of the tree
#' @param n_node_samples Integer vector of sample counts at each node
#' @param value Numeric vector of node values
#' @param node_count Integer total number of nodes in the tree
#' 
#' @return An object of class \code{simple_tree}
#' @keywords internal
new_simple_tree <- function(children_left,
                            children_right,
                            feature,
                            threshold,
                            max_depth,
                            n_node_samples,
                            value,
                            node_count) {
  structure(
    list(
      children_left   = as.integer(children_left),
      children_right  = as.integer(children_right),
      feature         = as.integer(feature),
      threshold       = as.numeric(threshold),
      max_depth       = as.integer(max_depth),
      n_node_samples  = as.integer(n_node_samples),
      value           = as.numeric(value),
      node_count      = as.integer(node_count)
    ),
    class = "simple_tree"
  )
}

#' Validator for simple_tree class
#' 
#' @param x A simple_tree object
#' @return The validated object (invisibly) or stops with an error
#' @keywords internal
validate_simple_tree <- function(x) {
  if (!inherits(x, "simple_tree")) {
    stop("Object must be of class 'simple_tree'", call. = FALSE)
  }
  
  # Check that all required fields are present
  required_fields <- c("children_left", "children_right", "feature", "threshold",
                       "max_depth", "n_node_samples", "value", "node_count")
  missing <- setdiff(required_fields, names(x))
  if (length(missing) > 0) {
    stop("Missing required fields: ", paste(missing, collapse = ", "), call. = FALSE)
  }
  
  # Check that vector lengths match node_count
  if (x$node_count > 0) {
    vec_lengths <- c(
      length(x$children_left),
      length(x$children_right),
      length(x$feature),
      length(x$threshold),
      length(x$n_node_samples),
      length(x$value)
    )
    if (!all(vec_lengths == x$node_count)) {
      stop("All node vectors must have length equal to node_count", call. = FALSE)
    }
  }
  
  # Check max_depth is positive
  if (x$max_depth < 0) {
    stop("max_depth must be non-negative", call. = FALSE)
  }
  
  invisible(x)
}

#' User-friendly constructor for simple_tree
#' 
#' @inheritParams new_simple_tree
#' @return A validated simple_tree object
#' @keywords internal
simple_tree <- function(children_left,
                        children_right,
                        feature,
                        threshold,
                        max_depth,
                        n_node_samples,
                        value,
                        node_count) {
  x <- new_simple_tree(
    children_left, children_right, feature, threshold,
    max_depth, n_node_samples, value, node_count
  )
  validate_simple_tree(x)
  x
}

#' Print method for simple_tree
#' 
#' @param x A simple_tree object
#' @param ... Additional arguments (currently unused)
#' @export
print.simple_tree <- function(x, ...) {
  cat("<simple_tree>\n")
  cat("  Nodes:", x$node_count, "\n")
  cat("  Max depth:", x$max_depth, "\n")
  cat("  Leaf nodes:", sum(x$children_left == -1), "\n")
  cat("  Split nodes:", sum(x$children_left != -1), "\n")
  invisible(x)
}

# ============================================================================
# tree_summary class
# ============================================================================

#' Constructor for tree_summary class
#' 
#' Creates a tree_summary object with validation
#' 
#' @param children_left Integer vector of left child indices
#' @param children_right Integer vector of right child indices
#' @param feature Integer vector of feature indices
#' @param feature_uniq Integer vector of unique feature indices used in tree
#' @param threshold Numeric vector of threshold values
#' @param max_depth Integer maximum depth
#' @param sample_weight Numeric vector of sample weights per node
#' @param init_prediction Numeric vector of initial predictions per node
#' @param node_count Integer total number of nodes
#' 
#' @return An object of class \code{tree_summary}
#' @keywords internal
new_tree_summary <- function(children_left,
                             children_right,
                             feature,
                             feature_uniq,
                             threshold,
                             max_depth,
                             sample_weight,
                             init_prediction,
                             node_count) {
  structure(
    list(
      children_left   = as.integer(children_left),
      children_right  = as.integer(children_right),
      feature         = as.integer(feature),
      feature_uniq    = as.integer(feature_uniq),
      threshold       = as.numeric(threshold),
      max_depth       = as.integer(max_depth),
      sample_weight   = as.numeric(sample_weight),
      init_prediction = as.numeric(init_prediction),
      node_count      = as.integer(node_count)
    ),
    class = "tree_summary"
  )
}

#' Validator for tree_summary class
#' 
#' @param x A tree_summary object
#' @return The validated object (invisibly) or stops with an error
#' @keywords internal
validate_tree_summary <- function(x) {
  if (!inherits(x, "tree_summary")) {
    stop("Object must be of class 'tree_summary'", call. = FALSE)
  }
  
  # Check required fields
  required_fields <- c("children_left", "children_right", "feature", "feature_uniq",
                       "threshold", "max_depth", "sample_weight", "init_prediction", "node_count")
  missing <- setdiff(required_fields, names(x))
  if (length(missing) > 0) {
    stop("Missing required fields: ", paste(missing, collapse = ", "), call. = FALSE)
  }
  
  # Check vector lengths
  if (x$node_count > 0) {
    vec_lengths <- c(
      length(x$children_left),
      length(x$children_right),
      length(x$feature),
      length(x$threshold),
      length(x$sample_weight),
      length(x$init_prediction)
    )
    if (!all(vec_lengths == x$node_count)) {
      stop("All node vectors must have length equal to node_count", call. = FALSE)
    }
  }
  
  invisible(x)
}

#' User-friendly constructor for tree_summary
#' 
#' @inheritParams new_tree_summary
#' @return A validated tree_summary object
#' @keywords internal
tree_summary <- function(children_left,
                         children_right,
                         feature,
                         feature_uniq,
                         threshold,
                         max_depth,
                         sample_weight,
                         init_prediction,
                         node_count) {
  x <- new_tree_summary(
    children_left, children_right, feature, feature_uniq,
    threshold, max_depth, sample_weight, init_prediction, node_count
  )
  validate_tree_summary(x)
  x
}

#' Print method for tree_summary
#' 
#' @param x A tree_summary object
#' @param ... Additional arguments (currently unused)
#' @export
print.tree_summary <- function(x, ...) {
  cat("<tree_summary>\n")
  cat("  Nodes:", x$node_count, "\n")
  cat("  Max depth:", x$max_depth, "\n")
  cat("  Unique features:", length(x$feature_uniq), "\n")
  cat("  Total init prediction:", round(sum(x$init_prediction), 4), "\n")
  invisible(x)
}

# ============================================================================
# qshap_tree_explainer class
# ============================================================================

#' Constructor for qshap_tree_explainer class
#' 
#' Creates a qshap_tree_explainer object
#' 
#' @param model The original tree model object
#' @param model_type Character string indicating model type ("xgboost" or "lightgbm")
#' @param max_depth Integer maximum tree depth
#' @param base_score Numeric base score (for XGBoost)
#' @param trees List of tree objects
#' @param store_v_invc Precomputed complex values for SHAP computation
#' @param store_z Precomputed root values for SHAP computation
#' 
#' @return An object of class \code{qshap_tree_explainer}
#' @keywords internal
new_qshap_tree_explainer <- function(model,
                                      model_type,
                                      max_depth,
                                      base_score = NULL,
                                      trees,
                                      store_v_invc,
                                      store_z) {
  obj <- list(
    model = model,
    model_type = as.character(model_type),
    max_depth = as.integer(max_depth),
    trees = trees,
    store_v_invc = store_v_invc,
    store_z = store_z
  )
  
  # Add base_score only for XGBoost
  if (!is.null(base_score)) {
    obj$base_score <- as.numeric(base_score)
  }
  
  # Set appropriate classes
  if (model_type == "xgboost") {
    class(obj) <- c("qshap_tree_explainer", "xgboost_explainer")
  } else if (model_type == "lightgbm") {
    class(obj) <- c("qshap_tree_explainer", "lightgbm_explainer")
  } else {
    class(obj) <- "qshap_tree_explainer"
  }
  
  obj
}

#' Validator for qshap_tree_explainer
#' 
#' @param x A qshap_tree_explainer object
#' @return The validated object (invisibly) or stops with an error
#' @keywords internal
validate_qshap_tree_explainer <- function(x) {
  if (!inherits(x, "qshap_tree_explainer")) {
    stop("Object must be of class 'qshap_tree_explainer'", call. = FALSE)
  }
  
  # Check required fields
  required_fields <- c("model", "model_type", "max_depth", "trees", "store_v_invc", "store_z")
  missing <- setdiff(required_fields, names(x))
  if (length(missing) > 0) {
    stop("Missing required fields: ", paste(missing, collapse = ", "), call. = FALSE)
  }
  
  # Validate model_type
  if (!x$model_type %in% c("xgboost", "lightgbm")) {
    stop("model_type must be 'xgboost' or 'lightgbm'", call. = FALSE)
  }
  
  # Check base_score for XGBoost
  if (x$model_type == "xgboost" && is.null(x$base_score)) {
    stop("base_score is required for XGBoost models", call. = FALSE)
  }
  
  # Check trees is a list
  if (!is.list(x$trees)) {
    stop("trees must be a list", call. = FALSE)
  }
  
  # Check max_depth is positive
  if (x$max_depth <= 0) {
    stop("max_depth must be positive", call. = FALSE)
  }
  
  invisible(x)
}

#' Print method for qshap_tree_explainer
#' 
#' @param x A qshap_tree_explainer object
#' @param ... Additional arguments (currently unused)
#' @export
print.qshap_tree_explainer <- function(x, ...) {
  cat("<qshap_tree_explainer>\n")
  cat("  Model type:", x$model_type, "\n")
  cat("  Number of trees:", length(x$trees), "\n")
  cat("  Max depth:", x$max_depth, "\n")
  if (!is.null(x$base_score)) {
    cat("  Base score:", round(x$base_score, 4), "\n")
  }
  invisible(x)
}

#' Summary method for qshap_tree_explainer
#' 
#' Provides detailed summary information about the explainer
#' 
#' @param object A qshap_tree_explainer object
#' @param ... Additional arguments (currently unused)
#' @export
summary.qshap_tree_explainer <- function(object, ...) {
  cat("Q-SHAP Tree Explainer Summary\n")
  cat("=============================\n\n")
  
  cat("Model Information:\n")
  cat("  Type:", object$model_type, "\n")
  cat("  Number of trees:", length(object$trees), "\n")
  cat("  Maximum tree depth:", object$max_depth, "\n")
  
  if (!is.null(object$base_score)) {
    cat("  Base score:", round(object$base_score, 6), "\n")
  }
  
  cat("\nTree Structure Statistics:\n")
  if (length(object$trees) > 0) {
    # Get node counts from first few trees
    node_counts <- sapply(object$trees[1:min(5, length(object$trees))], 
                          function(t) t$node_count)
    cat("  Average nodes per tree (first 5):", round(mean(node_counts), 1), "\n")
    cat("  Node count range (first 5):", min(node_counts), "-", max(node_counts), "\n")
  }
  
  cat("\nComputation Resources:\n")
  cat("  Precomputed matrices initialized: Yes\n")
  cat("  Store dimension (v_invc):", 
      if(is.matrix(object$store_v_invc)) nrow(object$store_v_invc) else "N/A", "\n")
  
  invisible(object)
}

# ============================================================================
# qshap_result class
# ============================================================================

#' Constructor for qshap_result class
#' 
#' Creates a qshap_result object to store Q-SHAP R-squared results
#' 
#' @param rsq Numeric vector of feature-specific R-squared values
#' @param feature_names Character vector of feature names (optional)
#' @param total_rsq Numeric total R-squared (sum of feature-specific values)
#' @param n_samples Integer number of samples used
#' @param n_features Integer number of features
#' @param loss Optional loss matrix (n_samples x n_features)
#' 
#' @return An object of class \code{qshap_result}
#' @keywords internal
new_qshap_result <- function(rsq, 
                             feature_names = NULL,
                             total_rsq = NULL,
                             n_samples = NULL,
                             n_features = NULL,
                             loss = NULL) {
  if (is.null(total_rsq)) {
    total_rsq <- sum(rsq, na.rm = TRUE)
  }
  if (is.null(n_features)) {
    n_features <- length(rsq)
  }
  if (is.null(feature_names)) {
    feature_names <- paste0("Feature_", seq_along(rsq))
  }
  
  structure(
    list(
      rsq = as.numeric(rsq),
      feature_names = as.character(feature_names),
      total_rsq = as.numeric(total_rsq),
      n_samples = if (!is.null(n_samples)) as.integer(n_samples) else NULL,
      n_features = as.integer(n_features),
      loss = loss
    ),
    class = "qshap_result"
  )
}

#' Validator for qshap_result
#' 
#' @param x A qshap_result object
#' @return The validated object (invisibly) or stops with an error
#' @keywords internal
validate_qshap_result <- function(x) {
  if (!inherits(x, "qshap_result")) {
    stop("Object must be of class 'qshap_result'", call. = FALSE)
  }
  
  required_fields <- c("rsq", "feature_names", "total_rsq", "n_features")
  missing <- setdiff(required_fields, names(x))
  if (length(missing) > 0) {
    stop("Missing required fields: ", paste(missing, collapse = ", "), call. = FALSE)
  }
  
  # Check lengths match
  if (length(x$rsq) != length(x$feature_names)) {
    stop("Length of rsq must match length of feature_names", call. = FALSE)
  }
  
  if (length(x$rsq) != x$n_features) {
    stop("Length of rsq must equal n_features", call. = FALSE)
  }
  
  invisible(x)
}

#' User-friendly constructor for qshap_result
#' 
#' @inheritParams new_qshap_result
#' @return A validated qshap_result object
#' @export
qshap_result <- function(rsq, 
                         feature_names = NULL,
                         total_rsq = NULL,
                         n_samples = NULL,
                         n_features = NULL,
                         loss = NULL) {
  x <- new_qshap_result(rsq, feature_names, total_rsq, n_samples, n_features, loss)
  validate_qshap_result(x)
  x
}

#' Print method for qshap_result
#' 
#' @param x A qshap_result object
#' @param n Integer number of top features to display (default: 10)
#' @param ... Additional arguments (currently unused)
#' @export
print.qshap_result <- function(x, n = 10, ...) {
  cat("<qshap_result>\n")
  cat("  Total R^2:", round(x$total_rsq, 4), "\n")
  cat("  Number of features:", x$n_features, "\n")
  if (!is.null(x$n_samples)) {
    cat("  Number of samples:", x$n_samples, "\n")
  }
  
  cat("\nTop", min(n, x$n_features), "features by R^2:\n")
  
  # Create a data frame and sort by R^2
  df <- data.frame(
    Feature = x$feature_names,
    R_squared = x$rsq,
    stringsAsFactors = FALSE
  )
  df <- df[order(df$R_squared, decreasing = TRUE), ]
  df <- utils::head(df, n)
  
  # Print as a formatted table
  print(df, row.names = FALSE, digits = 4)
  
  if (x$n_features > n) {
    cat("... and", x$n_features - n, "more features\n")
  }
  
  invisible(x)
}

#' Summary method for qshap_result
#' 
#' @param object A qshap_result object
#' @param ... Additional arguments (currently unused)
#' @export
summary.qshap_result <- function(object, ...) {
  cat("Q-SHAP Results Summary\n")
  cat("======================\n\n")
  
  cat("Overall Statistics:\n")
  cat("  Total R^2:", round(object$total_rsq, 6), "\n")
  cat("  Number of features:", object$n_features, "\n")
  if (!is.null(object$n_samples)) {
    cat("  Number of samples:", object$n_samples, "\n")
  }
  
  cat("\nR^2 Distribution:\n")
  cat("  Min:", round(min(object$rsq, na.rm = TRUE), 6), "\n")
  cat("  Q1:", round(stats::quantile(object$rsq, 0.25, na.rm = TRUE), 6), "\n")
  cat("  Median:", round(stats::median(object$rsq, na.rm = TRUE), 6), "\n")
  cat("  Mean:", round(mean(object$rsq, na.rm = TRUE), 6), "\n")
  cat("  Q3:", round(stats::quantile(object$rsq, 0.75, na.rm = TRUE), 6), "\n")
  cat("  Max:", round(max(object$rsq, na.rm = TRUE), 6), "\n")
  
  # Count significant features
  sig_features <- sum(object$rsq > 0.01, na.rm = TRUE)
  cat("\nSignificant Features (R^2 > 0.01):", sig_features, "\n")
  
  invisible(object)
}

#' Coercion method to data.frame for qshap_result
#' 
#' @param x A qshap_result object
#' @param row.names Not used
#' @param optional Not used
#' @param ... Additional arguments (currently unused)
#' @export
as.data.frame.qshap_result <- function(x, row.names = NULL, optional = FALSE, ...) {
  df <- data.frame(
    feature = x$feature_names,
    rsq = x$rsq,
    stringsAsFactors = FALSE
  )
  
  # Sort by R^2 descending
  df <- df[order(df$rsq, decreasing = TRUE), ]
  rownames(df) <- NULL
  
  df
}
