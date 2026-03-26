#' @importFrom stats predict
NULL

# Loss implementation for CatBoost model
#' @keywords internal
qshap_loss_catboost <- function(explainer, x, y, y_mean_ori = NULL) {
  if (!requireNamespace("catboost", quietly = TRUE)) {
    stop("catboost package is required for CatBoost support. ",
         "Install from: https://catboost.ai/docs/en/concepts/r-installation")
  }

  model <- explainer$model
  store_v_invc <- explainer$store_v_invc
  store_z <- explainer$store_z
  cb_trees <- explainer$trees
  bias <- explainer$base_score  # CatBoost bias extracted in gazer

  num_tree <- length(cb_trees)
  loss <- matrix(0, nrow = nrow(x), ncol = ncol(x))

  if (!is.matrix(x)) {
    x <- as.matrix(x)
  }

  # Create CatBoost pool for predictions
  pool <- catboost::catboost.load_pool(data = x)

  pb <- progress::progress_bar$new(
    format = "Progress [:bar] :current/:total (:percent)",
    total = num_tree,
    clear = FALSE,
    width = 60
  )

  for (i in seq_len(num_tree)) {
    pb$tick()

    # Compute residual before tree i
    if (i == 1) {
      local_res <- y - bias
    } else {
      pred_partial <- catboost::catboost.predict(model, pool, ntree_end = i - 1)
      local_res <- y - pred_partial
    }

    # Compute per-tree SHAP via our C++ TreeSHAP (since CatBoost R
    # doesn't expose per-tree SHAP values)
    tree_i <- cb_trees[[i]]
    T0_x_tree <- compute_treeshap(
      x,
      tree_i$children_left,
      tree_i$children_right,
      tree_i$feature,
      tree_i$threshold,
      tree_i$value,
      tree_i$n_node_samples
    )

    summary_tree <- summarize_tree(tree_i)

    current_tree_loss <- loss_treeshap(
      x, local_res, summary_tree, store_v_invc, store_z, T0_x_tree, 1.0
    )

    if (i == 1) {
      loss <- current_tree_loss
    } else {
      loss <- loss + current_tree_loss
    }
  }

  loss
}


# Convert CatBoost oblivious tree JSON to simple_tree format
#' @keywords internal
catboost_oblivious_to_simple <- function(tree_json, scale = 1.0) {
  splits <- tree_json$splits
  leaf_values <- as.numeric(tree_json$leaf_values) * scale
  leaf_weights <- as.numeric(tree_json$leaf_weights)

  # CatBoost oblivious trees can have empty leaves (weight = 0).
  # Replace with a tiny positive value to avoid Inf in sample_weight
  # and zero their value so they don't contribute to predictions.
  empty_mask <- leaf_weights == 0
  if (any(empty_mask)) {
    leaf_weights[empty_mask] <- 1
    leaf_values[empty_mask] <- 0.0
  }

  k <- length(splits)  # tree depth

  if (k == 0) {
    # Single leaf tree (stump)
    return(simple_tree(
      children_left  = -1L,
      children_right = -1L,
      feature        = -1L,
      threshold      = 0.0,
      max_depth      = 0L,
      n_node_samples = leaf_weights[1],
      value          = leaf_values[1],
      node_count     = 1L
    ))
  }

  # CatBoost stores splits bottom-up: splits[1] = deepest, splits[k] = root
  # Reverse to get top-down order
  splits_topdown <- rev(splits)

  # Build a complete binary tree with 2^(k+1) - 1 nodes
  total_nodes <- 2^(k + 1) - 1
  num_internal <- 2^k - 1
  num_leaves <- 2^k

  children_left  <- integer(total_nodes)
  children_right <- integer(total_nodes)
  feature        <- integer(total_nodes)
  threshold      <- numeric(total_nodes)
  value          <- numeric(total_nodes)
  n_node_samples <- numeric(total_nodes)

  # Fill internal nodes (BFS order)
  for (v in 0:(num_internal - 1)) {
    d <- floor(log2(v + 1))  # depth of node v in BFS
    children_left[v + 1]  <- 2L * v + 1L   # 0-based
    children_right[v + 1] <- 2L * v + 2L   # 0-based

    split_info <- splits_topdown[[d + 1]]
    # Handle different CatBoost JSON field names
    feat_idx <- split_info$float_feature_index
    if (is.null(feat_idx)) feat_idx <- split_info$split_index
    if (is.null(feat_idx)) stop("Cannot find feature index in CatBoost split info")

    border <- split_info$border
    if (is.null(border)) border <- split_info$threshold
    if (is.null(border)) stop("Cannot find threshold/border in CatBoost split info")

    feature[v + 1]   <- as.integer(feat_idx)
    threshold[v + 1] <- as.numeric(border)
  }

  # Fill leaf nodes
  # CatBoost leaf j maps to BFS leaf index (num_internal + j)
  for (j in 0:(num_leaves - 1)) {
    leaf_bfs <- num_internal + j  # 0-based BFS index
    children_left[leaf_bfs + 1]  <- -1L
    children_right[leaf_bfs + 1] <- -1L
    feature[leaf_bfs + 1]        <- -1L
    threshold[leaf_bfs + 1]      <- 0.0
    value[leaf_bfs + 1]          <- leaf_values[j + 1]
    n_node_samples[leaf_bfs + 1] <- leaf_weights[j + 1]
  }

  # Compute internal node sample counts and values bottom-up
  for (v in (num_internal - 1):0) {
    left_child  <- 2 * v + 1
    right_child <- 2 * v + 2
    nl <- n_node_samples[left_child + 1]
    nr <- n_node_samples[right_child + 1]
    n_node_samples[v + 1] <- nl + nr
    value[v + 1] <- (nl * value[left_child + 1] + nr * value[right_child + 1]) / (nl + nr)
  }

  simple_tree(
    children_left  = children_left,
    children_right = children_right,
    feature        = feature,
    threshold      = threshold,
    max_depth      = as.integer(k),
    n_node_samples = n_node_samples,
    value          = value,
    node_count     = as.integer(total_nodes)
  )
}


# Formats a CatBoost model into a list of simple_tree objects
#' @keywords internal
catboost_formatter <- function(model, max_depth = NULL) {
  if (!requireNamespace("catboost", quietly = TRUE)) {
    stop("catboost package is required for CatBoost support. ",
         "Install from: https://catboost.ai/docs/en/concepts/r-installation")
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("jsonlite package is required for CatBoost tree parsing")
  }

  # Export model to JSON for tree extraction
  tmp <- tempfile(fileext = ".json")
  on.exit(unlink(tmp), add = TRUE)
  catboost::catboost.save_model(model, tmp, file_format = "json")
  model_json <- jsonlite::fromJSON(tmp, simplifyVector = FALSE)

  # Extract scale and bias
  scale <- 1.0
  bias <- 0.0
  if (!is.null(model_json$scale_and_bias)) {
    sb <- model_json$scale_and_bias
    if (is.list(sb) && length(sb) >= 2) {
      scale <- as.numeric(sb[[1]])
      bias <- as.numeric(sb[[2]])
    }
  }

  # Extract trees — handle oblivious trees
  trees_data <- model_json$oblivious_trees
  if (is.null(trees_data)) {
    stop("Could not find oblivious_trees in CatBoost JSON. ",
         "Only symmetric/oblivious trees are currently supported. ",
         "Train with grow_policy='SymmetricTree' (default).")
  }

  num_trees <- length(trees_data)
  result <- vector("list", num_trees)
  actual_max_depth <- 0L

  for (i in seq_len(num_trees)) {
    result[[i]] <- catboost_oblivious_to_simple(trees_data[[i]], scale = scale)
    actual_max_depth <- max(actual_max_depth, result[[i]]$max_depth)
  }

  # Return trees and metadata
  attr(result, "bias") <- bias
  attr(result, "scale") <- scale
  attr(result, "max_depth") <- actual_max_depth

  result
}
