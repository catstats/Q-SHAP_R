#' @importFrom stats predict
NULL

# Find the leaf index for each sample in a CatBoost oblivious tree.
# Vectorized: all nodes at depth d share the same feature/threshold,
# so we compute decisions for all n samples at each depth simultaneously.
# Returns 0-based leaf index within [0, 2^k - 1].
#' @keywords internal
find_leaf_indices_oblivious <- function(x, tree) {
  k <- tree$max_depth
  if (k == 0L) return(rep(0L, nrow(x)))

  # For oblivious tree in BFS order, nodes at depth d start at index 2^d - 1.
  # All nodes at depth d use the same feature and threshold (the first node at that depth).
  leaf_idx <- rep(0L, nrow(x))
  for (d in 0:(k - 1L)) {
    node_at_depth <- 2^d - 1L  # 0-based BFS index of first node at depth d
    f <- tree$feature[node_at_depth + 1L] + 1L  # R 1-based column
    thr <- tree$threshold[node_at_depth + 1L]
    # If x <= threshold: go left (bit = 0), else go right (bit = 1)
    goes_right <- as.integer(x[, f] > thr)
    leaf_idx <- leaf_idx * 2L + goes_right
  }
  leaf_idx
}


# Loss implementation for CatBoost model
# Optimized: for oblivious trees, samples in the same leaf produce identical
# TreeSHAP values and identical T2 results (same decision signature).
# We compute once per unique leaf and broadcast to all samples in that leaf.
#' @keywords internal
qshap_loss_catboost <- function(explainer, x, y, y_mean_ori = NULL) {
  store_v_invc <- explainer$store_v_invc
  store_z <- explainer$store_z
  cb_trees <- explainer$trees
  bias <- explainer$base_score  # CatBoost bias extracted in gazer
  tree_summaries <- explainer$tree_summaries
  if (is.null(tree_summaries)) {
    tree_summaries <- lapply(cb_trees, summarize_tree)
  }

  num_tree <- length(cb_trees)
  n <- nrow(x)
  p <- ncol(x)
  loss <- matrix(0, nrow = n, ncol = p)

  if (!is.matrix(x)) {
    x <- as.matrix(x)
  }

  pb <- if (interactive()) progress::progress_bar$new(
    format = "Progress [:bar] :current/:total (:percent)",
    total = num_tree,
    clear = FALSE,
    width = 60
  ) else NULL

  # Track cumulative prediction incrementally using tree leaf values
  # (avoids calling catboost.predict entirely)
  cum_pred <- rep(bias, n)

  for (i in seq_len(num_tree)) {
    if (!is.null(pb)) pb$tick()

    # Residual before tree i
    local_res <- y - cum_pred

    tree_i <- cb_trees[[i]]
    summary_tree <- tree_summaries[[i]]

    # --- Leaf-based grouping optimization for oblivious trees ---
    # For oblivious trees, all nodes at the same depth use the same
    # feature and threshold. Therefore, two samples in the same leaf
    # have identical decisions at EVERY internal node, producing
    # identical TreeSHAP values and identical T2 weight matrices.
    # We only need to compute TreeSHAP and loss_treeshap once per
    # unique leaf (at most 2^depth groups), then broadcast.

    leaf_ids <- find_leaf_indices_oblivious(x, tree_i)

    # Use match() for fast grouping: maps each sample to its group index
    unique_leaves <- unique(leaf_ids)
    n_groups <- length(unique_leaves)
    leaf_to_group <- match(leaf_ids, unique_leaves)  # 1-based group index

    # One representative per group (first occurrence)
    rep_indices <- integer(n_groups)
    seen <- logical(n_groups)
    for (idx in seq_len(n)) {
      g <- leaf_to_group[idx]
      if (!seen[g]) {
        rep_indices[g] <- idx
        seen[g] <- TRUE
      }
    }

    x_reps <- x[rep_indices, , drop = FALSE]

    # Compute TreeSHAP only for representatives (2^k instead of n)
    T0_reps <- compute_treeshap(
      x_reps,
      tree_i$children_left,
      tree_i$children_right,
      tree_i$feature,
      tree_i$threshold,
      tree_i$value,
      tree_i$n_node_samples
    )

    # Compute T2 only for representatives (2^k instead of n)
    T2_reps <- T2(x_reps, summary_tree, store_v_invc, store_z, FALSE)

    # Broadcast T0 and T2 to all n samples via leaf group mapping
    T0_all <- T0_reps[leaf_to_group, , drop = FALSE]
    T2_all <- T2_reps[leaf_to_group, , drop = FALSE]

    # Compute loss: loss = T2 - 2 * T0 * residual (vectorized, O(n*p))
    current_tree_loss <- T2_all - 2.0 * T0_all * local_res

    if (i == 1) {
      loss <- current_tree_loss
    } else {
      loss <- loss + current_tree_loss
    }

    # Update cumulative prediction using tree's leaf values
    # leaf_ids are 0-based leaf indices; BFS index = num_internal + leaf_id
    num_internal <- 2^(tree_i$max_depth) - 1L
    # tree_i$value is 0-based indexed, leaf value at BFS position (num_internal + leaf_id)
    tree_preds <- tree_i$value[num_internal + leaf_ids + 1L]
    cum_pred <- cum_pred + tree_preds
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
  catboost_save_model <- tryCatch(
    getExportedValue("catboost", "catboost.save_model"),
    error = function(e) NULL
  )
  if (is.null(catboost_save_model)) {
    stop("The optional catboost package is required for CatBoost support. ",
         "It is not distributed on CRAN; install it from the official ",
         "CatBoost R instructions: ",
         "https://catboost.ai/docs/en/concepts/r-installation",
         call. = FALSE)
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("jsonlite package is required for CatBoost tree parsing")
  }

  # Export model to JSON for tree extraction while keeping CatBoost a
  # runtime-only optional backend.
  tmp <- tempfile(fileext = ".json")
  on.exit(unlink(tmp), add = TRUE)
  catboost_save_model(model, tmp, file_format = "json")
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
