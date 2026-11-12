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
#' @param n_node_samples Numeric vector of node covers or weights, such as
#'   sample counts or summed Hessians
#' @param value Numeric vector of node values
#' @param node_count Integer total number of nodes in the tree
#' @param default_left Logical vector indicating the missing-value direction.
#' @param xgboost_split Logical; use XGBoost float32 strict-less routing.
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
                            node_count,
                            default_left = NULL,
                            xgboost_split = FALSE) {
  if (is.null(default_left)) {
    default_left <- rep.int(FALSE, node_count)
  }
  structure(
    list(
      children_left   = as.integer(children_left),
      children_right  = as.integer(children_right),
      feature         = as.integer(feature),
      threshold       = as.numeric(threshold),
      max_depth       = as.integer(max_depth),
      n_node_samples  = as.numeric(n_node_samples),
      value           = as.numeric(value),
      node_count      = as.integer(node_count),
      default_left    = as.logical(default_left),
      xgboost_split   = isTRUE(xgboost_split)
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
                       "max_depth", "n_node_samples", "value", "node_count",
                       "default_left", "xgboost_split")
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
      length(x$value),
      length(x[["default_left"]])
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
                        node_count,
                        default_left = NULL,
                        xgboost_split = FALSE) {
  x <- new_simple_tree(
    children_left, children_right, feature, threshold,
    max_depth, n_node_samples, value, node_count,
    default_left, xgboost_split
  )
  validate_simple_tree(x)
  x
}

#' Print method for simple_tree
#' 
#' @param x A simple_tree object
#' @param ... Additional arguments (currently unused)
#' @return The input \code{x} is returned invisibly. Called primarily for its
#'   side effect of printing a summary of the \code{simple_tree} object to the
#'   console.
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
#' @param default_left Logical vector indicating the missing-value direction.
#' @param xgboost_split Logical; use XGBoost float32 strict-less routing.
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
                             node_count,
                             default_left = NULL,
                             xgboost_split = FALSE) {
  if (is.null(default_left)) {
    default_left <- rep.int(FALSE, node_count)
  }
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
      node_count      = as.integer(node_count),
      default_left    = as.logical(default_left),
      xgboost_split   = isTRUE(xgboost_split)
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
                       "threshold", "max_depth", "sample_weight", "init_prediction",
                       "node_count", "default_left", "xgboost_split")
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
      length(x$init_prediction),
      length(x[["default_left"]])
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
                         node_count,
                         default_left = NULL,
                         xgboost_split = FALSE) {
  x <- new_tree_summary(
    children_left, children_right, feature, feature_uniq,
    threshold, max_depth, sample_weight, init_prediction, node_count,
    default_left, xgboost_split
  )
  validate_tree_summary(x)
  x
}

#' Print method for tree_summary
#' 
#' @param x A tree_summary object
#' @param ... Additional arguments (currently unused)
#' @return The input \code{x} is returned invisibly. Called primarily for its
#'   side effect of printing a summary of the \code{tree_summary} object to the
#'   console.
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
#' @param model_type Character string indicating the model backend:
#'   \code{"xgboost"}, \code{"lightgbm"}, or \code{"catboost"}.
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
  
  # Backend routing is recorded in model_type; methods use this shared class.
  class(obj) <- "qshap_tree_explainer"
  
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
  if (!x$model_type %in% c("xgboost", "lightgbm", "catboost")) {
    stop("model_type must be 'xgboost', 'lightgbm', or 'catboost'", call. = FALSE)
  }
  
  # Check base_score for XGBoost and CatBoost
  if (x$model_type %in% c("xgboost", "catboost") && is.null(x$base_score)) {
    stop("base_score is required for XGBoost and CatBoost models", call. = FALSE)
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
#' @return The input \code{x} is returned invisibly. Called primarily for its
#'   side effect of printing a summary of the \code{qshap_tree_explainer} object
#'   to the console.
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
#' Computes a separate, programmatically accessible model summary. Printing is
#' handled by \code{print.summary.qshap_tree_explainer()}.
#' 
#' @param object A qshap_tree_explainer object
#' @param ... Additional arguments (currently unused)
#' @return A list of class \code{summary.qshap_tree_explainer}, containing
#'   \code{model_type}, \code{n_trees}, \code{max_depth}, \code{base_score},
#'   \code{node_counts} for all trees, \code{node_count_summary} (minimum,
#'   mean and maximum), \code{precomputed_initialized}, and
#'   \code{precomputed_dimensions} (dimensions of \code{v_invc} and \code{z}).
#'   The original model and cached numerical arrays are not included. Summary
#'   values retain their numerical precision; an empty ensemble has missing
#'   node-count summary statistics.
#' @export
summary.qshap_tree_explainer <- function(object, ...) {
  node_counts <- vapply(object$trees, function(tree) {
    as.numeric(tree$node_count)
  }, numeric(1))
  node_count_summary <- c(min = NA_real_, mean = NA_real_, max = NA_real_)
  if (length(node_counts)) {
    node_count_summary[] <- c(min(node_counts), mean(node_counts), max(node_counts))
  }

  structure(list(
    model_type = object$model_type,
    n_trees = length(object$trees),
    max_depth = object$max_depth,
    base_score = object$base_score,
    node_counts = node_counts,
    node_count_summary = node_count_summary,
    precomputed_initialized = !is.null(object$store_v_invc) && !is.null(object$store_z),
    precomputed_dimensions = list(
      v_invc = dim(object$store_v_invc), z = dim(object$store_z)
    )
  ), class = "summary.qshap_tree_explainer")
}

#' Print a Q-SHAP explainer summary
#'
#' @param x A \code{summary.qshap_tree_explainer} object.
#' @param digits Integer number of significant digits to display, from 1 to 22.
#' @param ... Additional arguments (currently unused).
#' @return The summary object \code{x}, invisibly.
#' @export
print.summary.qshap_tree_explainer <- function(x, digits = 6L, ...) {
  validate_summary_digits(digits)
  display <- function(value) format(signif(value, digits), trim = TRUE)
  cat("Q-SHAP Tree Explainer Summary\n")
  cat("=============================\n\n")

  cat("Model Information:\n")
  cat("  Type:", x$model_type, "\n")
  cat("  Number of trees:", x$n_trees, "\n")
  cat("  Maximum tree depth:", x$max_depth, "\n")
  if (!is.null(x$base_score)) {
    cat("  Base score:", display(x$base_score), "\n")
  }

  cat("\nTree Structure Statistics:\n")
  if (x$n_trees > 0L) {
    cat("  Average nodes per tree (all trees):", display(x$node_count_summary[["mean"]]), "\n")
    cat("  Node count range (all trees):", display(x$node_count_summary[["min"]]),
        "-", display(x$node_count_summary[["max"]]), "\n")
  } else {
    cat("  No trees available.\n")
  }

  cat("\nComputation Resources:\n")
  cat("  Precomputed matrices initialized:",
      if (x$precomputed_initialized) "Yes" else "No", "\n")
  for (name in names(x$precomputed_dimensions)) {
    dimensions <- x$precomputed_dimensions[[name]]
    cat("  Store dimensions (", name, "): ",
        if (is.null(dimensions)) "N/A" else paste(dimensions, collapse = " x "),
        "\n", sep = "")
  }
  invisible(x)
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
#' @param loss Optional raw observation-level squared-loss contribution matrix
#'   (n_samples x n_features).
#' @param local_rsq Optional matrix of observation-level contributions to the
#'   global R-squared decomposition (n_samples x n_features). Its column sums
#'   equal the feature-specific values in \code{rsq}.
#' 
#' @return An object of class \code{qshap_result}
#' @keywords internal
new_qshap_result <- function(rsq, 
                             feature_names = NULL,
                             total_rsq = NULL,
                             n_samples = NULL,
                             n_features = NULL,
                             loss = NULL,
                             local_rsq = NULL) {
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
      loss = loss,
      local_rsq = local_rsq
    ),
    class = c("qshap_result", "qshap_rsq")
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

  for (field in c("loss", "local_rsq")) {
    value <- x[[field]]
    if (!is.null(value)) {
      numeric_values <- is.numeric(value) ||
        (is.data.frame(value) && all(vapply(value, is.numeric, logical(1))))
      if (length(dim(value)) != 2L || !numeric_values) {
        stop(field, " must be a numeric matrix or data frame", call. = FALSE)
      }
      if (ncol(value) != x$n_features) {
        stop(field, " must have n_features columns", call. = FALSE)
      }
      if (!is.null(x$n_samples) && nrow(value) != x$n_samples) {
        stop(field, " must have n_samples rows", call. = FALSE)
      }
    }
  }

  if (!is.null(x$loss) && !is.null(x$local_rsq) &&
      !identical(dim(x$loss), dim(x$local_rsq))) {
    stop("loss and local_rsq must have identical dimensions", call. = FALSE)
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
                         loss = NULL,
                         local_rsq = NULL) {
  x <- new_qshap_result(
    rsq = rsq,
    feature_names = feature_names,
    total_rsq = total_rsq,
    n_samples = n_samples,
    n_features = n_features,
    loss = loss,
    local_rsq = local_rsq
  )
  validate_qshap_result(x)
  x
}

#' Print method for qshap_result
#' 
#' @param x A qshap_result object
#' @param n Integer number of top features to display (default: 10)
#' @param ... Additional arguments (currently unused)
#' @return The input \code{x} is returned invisibly. Called primarily for its
#'   side effect of printing a summary of the \code{qshap_result} object to the
#'   console.
#' @export
print.qshap_result <- function(x, n = 10, ...) {
  cat("<qshap_result>\n")
  cat("  Total R^2:", round(x$total_rsq, 4), "\n")
  cat("  Number of features:", x$n_features, "\n")
  if (!is.null(x$n_samples)) {
    cat("  Number of samples:", x$n_samples, "\n")
  }
  if (!is.null(x$loss)) {
    cat("  Raw loss contributions:", paste(dim(x$loss), collapse = " x "), "\n")
  }
  if (!is.null(x$local_rsq)) {
    cat("  Local R^2-scale contributions:",
        paste(dim(x$local_rsq), collapse = " x "), "\n")
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
#' Computes a separate summary without printing. Use its \code{features} table
#' or \code{rsq_distribution} vector for further analysis, and \code{print()}
#' to control display precision and the number of displayed features.
#'
#' @param object A qshap_result object
#' @param ... Additional arguments (currently unused)
#' @return A list of class \code{summary.qshap_result}, containing
#'   \code{total_rsq}, \code{n_samples}, \code{n_features}, a complete
#'   \code{features} data frame (columns \code{feature} and \code{rsq}, sorted
#'   in decreasing contribution order), \code{rsq_distribution} (\code{min},
#'   \code{q1}, \code{median}, \code{mean}, \code{q3}, \code{max}),
#'   \code{n_missing}, \code{threshold} (0.01), \code{n_above_threshold},
#'   \code{loss_dim}, and \code{local_rsq_dim}. Missing contributions are
#'   excluded from distribution statistics and threshold counts. Empty or
#'   entirely missing contributions yield missing distribution statistics.
#'   Numerical values are not rounded. Local matrices themselves are not copied
#'   into the summary. The threshold count is descriptive, not a significance test.
#' @examples
#' result <- qshap_result(c(0.30, 0.15, -0.05),
#'                        feature_names = c("a", "b", "c"), n_samples = 20)
#' s <- summary(result)
#' s$total_rsq
#' s$features
#' s$rsq_distribution
#' print(s, n = 2, digits = 4)
#' @export
summary.qshap_result <- function(object, ...) {
  rsq <- object$rsq
  observed <- rsq[!is.na(rsq)]
  distribution <- c(min = NA_real_, q1 = NA_real_, median = NA_real_,
                    mean = NA_real_, q3 = NA_real_, max = NA_real_)
  if (length(observed)) {
    distribution[] <- c(
      min(observed), stats::quantile(observed, 0.25, names = FALSE),
      stats::median(observed), mean(observed),
      stats::quantile(observed, 0.75, names = FALSE), max(observed)
    )
  }
  structure(list(
    total_rsq = object$total_rsq,
    n_samples = object$n_samples,
    n_features = object$n_features,
    features = as.data.frame(object),
    rsq_distribution = distribution,
    n_missing = sum(is.na(rsq)),
    threshold = 0.01,
    n_above_threshold = sum(rsq > 0.01, na.rm = TRUE),
    loss_dim = dim(object$loss),
    local_rsq_dim = dim(object$local_rsq)
  ), class = "summary.qshap_result")
}

#' Print a Q-SHAP result summary
#'
#' @param x A \code{summary.qshap_result} object.
#' @param n Non-negative integer number of top features to display (default: 10).
#'   This changes display only; the summary retains every feature.
#' @param digits Integer number of significant digits to display, from 1 to 22.
#' @param ... Additional arguments passed to the feature table's print method.
#' @return The summary object \code{x}, invisibly.
#' @export
print.summary.qshap_result <- function(x, n = 10L, digits = 6L, ...) {
  validate_summary_digits(digits)
  if (!is.numeric(n) || length(n) != 1L || is.na(n) ||
      !is.finite(n) || n < 0 || n != floor(n)) {
    stop("n must be a non-negative integer", call. = FALSE)
  }
  display <- function(value) format(signif(value, digits), trim = TRUE)
  cat("Q-SHAP Results Summary\n")
  cat("======================\n\n")

  cat("Overall Statistics:\n")
  cat("  Total R^2:", display(x$total_rsq), "\n")
  cat("  Number of features:", x$n_features, "\n")
  if (!is.null(x$n_samples)) {
    cat("  Number of samples:", x$n_samples, "\n")
  }
  if (!is.null(x$loss_dim)) {
    cat("  Raw loss contributions:", paste(x$loss_dim, collapse = " x "), "\n")
  }
  if (!is.null(x$local_rsq_dim)) {
    cat("  Local R^2-scale contributions:",
        paste(x$local_rsq_dim, collapse = " x "), "\n")
  }

  cat("\nR^2 Distribution:\n")
  labels <- c("Min", "Q1", "Median", "Mean", "Q3", "Max")
  for (i in seq_along(labels)) {
    cat("  ", labels[i], ": ", display(x$rsq_distribution[i]), "\n", sep = "")
  }
  if (x$n_missing > 0L) cat("  Missing contributions:", x$n_missing, "\n")
  cat("\nFeatures with R^2 >", x$threshold, ":", x$n_above_threshold, "\n")

  n_show <- min(n, nrow(x$features))
  if (n_show > 0L) {
    cat("\nTop", n_show, "features by R^2:\n")
    print(x$features[seq_len(n_show), , drop = FALSE],
          row.names = FALSE, digits = digits, ...)
    if (nrow(x$features) > n_show) {
      cat("... and", nrow(x$features) - n_show, "more features\n")
    }
  }
  invisible(x)
}

validate_summary_digits <- function(digits) {
  if (!is.numeric(digits) || length(digits) != 1L || is.na(digits) ||
      !is.finite(digits) || digits != floor(digits) || digits < 1 || digits > 22) {
    stop("digits must be an integer between 1 and 22", call. = FALSE)
  }
  invisible(digits)
}

#' Coercion method to data.frame for qshap_result
#' 
#' @param x A qshap_result object
#' @param row.names Not used
#' @param optional Not used
#' @param ... Additional arguments (currently unused)
#' @return A \code{data.frame} with columns \code{feature} (character) and
#'   \code{rsq} (numeric), sorted by \code{rsq} in decreasing order.
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
