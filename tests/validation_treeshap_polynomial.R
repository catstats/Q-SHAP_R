library(qshap)

tree_prediction <- function(tree, x) {
  node <- 0L
  while (tree$children_left[node + 1L] >= 0L) {
    feature <- tree$feature[node + 1L] + 1L
    if (x[feature] <= tree$threshold[node + 1L]) {
      node <- tree$children_left[node + 1L]
    } else {
      node <- tree$children_right[node + 1L]
    }
  }
  tree$value[node + 1L]
}

conditional_expectation <- function(tree, x, conditioned, node = 0L) {
  left <- tree$children_left[node + 1L]
  if (left < 0L) {
    return(tree$value[node + 1L])
  }

  right <- tree$children_right[node + 1L]
  feature <- tree$feature[node + 1L]
  if (feature %in% conditioned) {
    next_node <- if (x[feature + 1L] <= tree$threshold[node + 1L]) {
      left
    } else {
      right
    }
    return(conditional_expectation(tree, x, conditioned, next_node))
  }

  left_cover <- tree$n_node_samples[left + 1L]
  right_cover <- tree$n_node_samples[right + 1L]
  total_cover <- left_cover + right_cover
  left_cover / total_cover *
    conditional_expectation(tree, x, conditioned, left) +
    right_cover / total_cover *
    conditional_expectation(tree, x, conditioned, right)
}

# Deliberately exponential reference oracle, retained only for tiny test trees.
subset_treeshap_reference <- function(tree, X) {
  features <- sort(unique(tree$feature[tree$children_left >= 0L]))
  out <- matrix(0.0, nrow = nrow(X), ncol = ncol(X))
  if (!length(features)) {
    return(out)
  }

  m <- length(features)
  for (row in seq_len(nrow(X))) {
    for (feature in features) {
      others <- setdiff(features, feature)
      for (mask in 0:(2^length(others) - 1L)) {
        included <- if (length(others)) {
          as.logical(intToBits(mask)[seq_along(others)])
        } else {
          logical()
        }
        subset <- others[included]
        subset_size <- length(subset)
        weight <- factorial(subset_size) *
          factorial(m - subset_size - 1L) / factorial(m)
        out[row, feature + 1L] <- out[row, feature + 1L] + weight * (
          conditional_expectation(tree, X[row, ], c(subset, feature)) -
            conditional_expectation(tree, X[row, ], subset)
        )
      }
    }
  }
  out
}

compute_polynomial_treeshap <- function(tree, X) {
  qshap:::compute_treeshap(
    X,
    tree$children_left,
    tree$children_right,
    tree$feature,
    tree$threshold,
    tree$value,
    tree$n_node_samples
  )
}

make_complete_tree <- function(depth, p) {
  n_internal <- 2L^depth - 1L
  n_leaves <- 2L^depth
  node_count <- n_internal + n_leaves
  children_left <- rep.int(-1L, node_count)
  children_right <- rep.int(-1L, node_count)
  feature <- rep.int(-1L, node_count)
  threshold <- numeric(node_count)
  value <- numeric(node_count)
  cover <- numeric(node_count)

  for (node in 0:(n_internal - 1L)) {
    children_left[node + 1L] <- 2L * node + 1L
    children_right[node + 1L] <- 2L * node + 2L
    # Sampling with replacement deliberately creates repeated path features.
    feature[node + 1L] <- sample.int(p, 1L) - 1L
    threshold[node + 1L] <- stats::runif(1L, -0.75, 0.75)
  }

  leaf_nodes <- (n_internal + 1L):node_count
  value[leaf_nodes] <- stats::rnorm(n_leaves)
  cover[leaf_nodes] <- sample.int(40L, n_leaves, replace = TRUE)
  for (node in rev(0:(n_internal - 1L))) {
    left <- children_left[node + 1L] + 1L
    right <- children_right[node + 1L] + 1L
    cover[node + 1L] <- cover[left] + cover[right]
    value[node + 1L] <-
      (cover[left] * value[left] + cover[right] * value[right]) /
      cover[node + 1L]
  }

  qshap:::simple_tree(
    children_left, children_right, feature, threshold, depth,
    cover, value, node_count
  )
}

make_irregular_tree <- function(max_depth, p) {
  make_node <- function(depth) {
    if (depth >= max_depth || (depth > 0L && stats::runif(1L) < 0.35)) {
      return(list(
        leaf = TRUE,
        value = stats::rnorm(1L),
        cover = sample.int(40L, 1L)
      ))
    }

    left <- make_node(depth + 1L)
    right <- make_node(depth + 1L)
    cover <- left$cover + right$cover
    list(
      leaf = FALSE,
      feature = sample.int(p, 1L) - 1L,
      threshold = stats::runif(1L, -0.75, 0.75),
      left = left,
      right = right,
      cover = cover,
      value = (left$cover * left$value + right$cover * right$value) / cover
    )
  }

  nodes <- list()
  depths <- integer()
  append_node <- function(node, depth) {
    index <- length(nodes)
    nodes[[index + 1L]] <<- node
    depths[index + 1L] <<- depth
    if (isTRUE(node$leaf)) {
      return(index)
    }
    left <- append_node(node$left, depth + 1L)
    right <- append_node(node$right, depth + 1L)
    stored <- nodes[[index + 1L]]
    stored$left_index <- left
    stored$right_index <- right
    nodes[[index + 1L]] <<- stored
    index
  }
  append_node(make_node(0L), 0L)

  node_count <- length(nodes)
  children_left <- rep.int(-1L, node_count)
  children_right <- rep.int(-1L, node_count)
  feature <- rep.int(-1L, node_count)
  threshold <- numeric(node_count)
  value <- numeric(node_count)
  cover <- numeric(node_count)
  for (index in seq_len(node_count)) {
    node <- nodes[[index]]
    value[index] <- node$value
    cover[index] <- node$cover
    if (!isTRUE(node$leaf)) {
      children_left[index] <- node$left_index
      children_right[index] <- node$right_index
      feature[index] <- node$feature
      threshold[index] <- node$threshold
    }
  }

  qshap:::simple_tree(
    children_left, children_right, feature, threshold, max(depths),
    cover, value, node_count
  )
}

check_against_subset_reference <- function(tree, X, tolerance = 2e-12) {
  polynomial <- compute_polynomial_treeshap(tree, X)
  reference <- subset_treeshap_reference(tree, X)
  error <- max(abs(polynomial - reference))
  if (!is.finite(error) || error > tolerance) {
    stop(
      sprintf("Polynomial/subset TreeSHAP mismatch: %.6g", error),
      call. = FALSE
    )
  }

  predictions <- apply(X, 1L, function(x) tree_prediction(tree, x))
  efficiency_error <- max(abs(rowSums(polynomial) -
    (predictions - tree$value[1L])))
  if (!is.finite(efficiency_error) || efficiency_error > tolerance) {
    stop(
      sprintf("TreeSHAP efficiency mismatch: %.6g", efficiency_error),
      call. = FALSE
    )
  }
}

set.seed(20260805)

# Random small complete trees exercise the exact oracle, including repeated
# features at different depths and features that occur only on some paths.
for (depth in 1:4) {
  for (trial in 1:4) {
    tree <- make_complete_tree(depth, p = 5L)
    X <- matrix(stats::rnorm(8L * 5L), nrow = 8L, ncol = 5L)
    check_against_subset_reference(tree, X)
  }
}

# Random unbalanced topologies exercise the same oracle beyond complete trees.
for (trial in 1:20) {
  tree <- make_irregular_tree(max_depth = 5L, p = 4L)
  X <- matrix(stats::rnorm(8L * 4L), nrow = 8L, ncol = 4L)
  check_against_subset_reference(tree, X)
}

# An explicitly irregular tree repeats feature 0 on its right branch.
irregular_tree <- qshap:::simple_tree(
  children_left = c(1L, 2L, -1L, -1L, 5L, 7L, -1L, -1L, -1L),
  children_right = c(4L, 3L, -1L, -1L, 6L, 8L, -1L, -1L, -1L),
  feature = c(0L, 1L, -1L, -1L, 0L, 2L, -1L, -1L, -1L),
  threshold = c(0.1, -0.2, 0, 0, 0.7, -0.4, 0, 0, 0),
  max_depth = 3L,
  n_node_samples = c(35, 12, 5, 7, 23, 12, 11, 3, 9),
  value = c(
    0.885714285714286, -0.125, -1, 0.5, 1.41304347826087,
    0.875, 2, -0.4, 1.3
  ),
  node_count = 9L
)
check_against_subset_reference(
  irregular_tree,
  matrix(stats::rnorm(30L), nrow = 10L, ncol = 3L)
)

# A 21-feature comb tree proves the production kernel has no 20-feature cap.
n_unique <- 21L
node_count <- 2L * n_unique + 1L
children_left <- rep.int(-1L, node_count)
children_right <- rep.int(-1L, node_count)
feature <- rep.int(-1L, node_count)
threshold <- numeric(node_count)
value <- numeric(node_count)
cover <- numeric(node_count)
for (depth in 0:(n_unique - 1L)) {
  node <- 2L * depth
  children_left[node + 1L] <- node + 1L
  children_right[node + 1L] <- node + 2L
  feature[node + 1L] <- depth
  threshold[node + 1L] <- (depth - 10) / 20
}
leaf_nodes <- c(seq.int(2L, 2L * n_unique, by = 2L), node_count)
cover[leaf_nodes] <- seq_along(leaf_nodes) + 1
value[leaf_nodes] <- seq(-1.2, 1.4, length.out = length(leaf_nodes))
for (depth in rev(0:(n_unique - 1L))) {
  node <- 2L * depth + 1L
  left <- children_left[node] + 1L
  right <- children_right[node] + 1L
  cover[node] <- cover[left] + cover[right]
  value[node] <- (cover[left] * value[left] + cover[right] * value[right]) /
    cover[node]
}
wide_tree <- qshap:::simple_tree(
  children_left, children_right, feature, threshold, n_unique,
  cover, value, node_count
)
X_wide <- matrix(stats::rnorm(6L * n_unique), nrow = 6L, ncol = n_unique)
wide_shap <- compute_polynomial_treeshap(wide_tree, X_wide)
stopifnot(all(is.finite(wide_shap)))
stopifnot(max(abs(
  rowSums(wide_shap) -
    (apply(X_wide, 1L, function(x) tree_prediction(wide_tree, x)) -
      wide_tree$value[1L])
)) < 2e-12)

# Real-model CatBoost scalability checks live in
# validation_catboost_fast.R. That optional validation script is kept out of
# the source package because the R CatBoost package is not distributed on
# CRAN.

cat("Polynomial TreeSHAP scalability validation passed.\n")
