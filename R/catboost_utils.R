#' @importFrom stats predict
NULL

# CatBoost quantizes numeric inputs and split borders to IEEE-754 float32.
# Round-trip through four-byte binary values while returning ordinary doubles
# that can be passed to the existing Rcpp backends.
#' @keywords internal
catboost_float32 <- function(x) {
  if (!length(x)) return(as.numeric(x))
  bytes <- writeBin(
    as.numeric(x),
    raw(),
    size = 4L,
    endian = .Platform$endian
  )
  readBin(
    bytes,
    what = numeric(),
    n = length(x),
    size = 4L,
    endian = .Platform$endian
  )
}


catboost_float32_matrix <- function(x) {
  if (!is.matrix(x)) x <- as.matrix(x)
  if (!is.numeric(x)) {
    stop("CatBoost currently requires a numeric matrix.", call. = FALSE)
  }
  dimensions <- dim(x)
  dimnames_x <- dimnames(x)
  x <- catboost_float32(x)
  dim(x) <- dimensions
  dimnames(x) <- dimnames_x
  x
}


# CatBoost may export leaves that received no training rows.  Q-SHAP's path
# probabilities require strictly positive cover, so use a negligible value
# relative to the observed tree cover instead of pretending that an empty leaf
# contains one sample.  Leaf predictions themselves are never changed.
#' @keywords internal
catboost_cover_floor <- function(weights) {
  weights <- suppressWarnings(as.numeric(weights))
  positive <- weights[is.finite(weights) & weights > 0]
  if (!length(positive)) return(1e-12)

  total <- sum(positive)
  if (!is.finite(total) || total <= 0) total <- max(positive)
  floor_value <- total * 1e-12
  if (!is.finite(floor_value) || floor_value <= 0) {
    floor_value <- .Machine$double.xmin
  }
  floor_value
}


# CatBoost JSON encodes numeric splits as the predicate x > border.  The
# false branch is stored in `left` and the true branch in `right`, so the
# shared simple-tree representation routes finite values left with x <= border.
#' @keywords internal
catboost_split_info <- function(split, float_features, context = "split",
                                round_border = TRUE) {
  split_type <- split[["split_type"]]
  if (length(split_type) != 1L || !identical(as.character(split_type), "FloatFeature")) {
    stop(
      "Unsupported CatBoost ", context, ": only numeric FloatFeature splits ",
      "are supported; found '",
      if (length(split_type)) as.character(split_type)[1L] else "missing split_type",
      "'.",
      call. = FALSE
    )
  }

  float_index <- suppressWarnings(as.numeric(split[["float_feature_index"]]))
  if (length(float_index) != 1L || is.na(float_index) || !is.finite(float_index) ||
      float_index < 0 || float_index != floor(float_index)) {
    stop("Invalid CatBoost ", context, " float_feature_index.", call. = FALSE)
  }
  float_index <- as.integer(float_index)

  border <- suppressWarnings(as.numeric(split[["border"]]))
  if (length(border) != 1L || is.na(border) || !is.finite(border)) {
    stop("Invalid CatBoost ", context, " border.", call. = FALSE)
  }

  if (length(float_features) < float_index + 1L) {
    stop(
      "CatBoost ", context, " references float feature ", float_index,
      ", but features_info does not contain it.",
      call. = FALSE
    )
  }
  feature_meta <- float_features[[float_index + 1L]]
  flat_index <- feature_meta[["flat_feature_index"]]
  if (is.null(flat_index)) {
    stop(
      "CatBoost ", context, " feature metadata does not provide ",
      "flat_feature_index; refusing to guess an input column.",
      call. = FALSE
    )
  }
  flat_index <- suppressWarnings(as.numeric(flat_index))
  if (length(flat_index) != 1L || is.na(flat_index) || !is.finite(flat_index) ||
      flat_index < 0 || flat_index != floor(flat_index)) {
    stop("Invalid CatBoost flat feature index for ", context, ".", call. = FALSE)
  }

  nan_treatment <- feature_meta[["nan_value_treatment"]]
  nan_treatment <- if (length(nan_treatment)) as.character(nan_treatment)[1L] else "AsIs"
  if (!nan_treatment %in% c("AsFalse", "AsTrue", "AsIs")) {
    stop(
      "Unsupported CatBoost nan_value_treatment '", nan_treatment,
      "' for ", context, ".",
      call. = FALSE
    )
  }

  list(
    feature = as.integer(flat_index),
    threshold = if (isTRUE(round_border)) catboost_float32(border) else border,
    # CatBoost sends AsFalse and AsIs to the false/left branch, and AsTrue to
    # the true/right branch.
    default_left = !identical(nan_treatment, "AsTrue")
  )
}


# Determine whether a CatBoost explainer can use the symmetric-tree backend.
# Explainers created by older qshap versions have no marker because those
# versions could only parse oblivious trees, so a missing marker is symmetric.
#' @keywords internal
catboost_uses_oblivious_trees <- function(explainer) {
  tree_type <- attr(explainer$trees, "catboost_tree_type", exact = TRUE)
  is.null(tree_type) || identical(tree_type, "oblivious")
}


# Prepare CatBoost inputs for every Q-SHAP backend.  Float32 rounding matches
# CatBoost's quantization boundary.  Missing values are replaced feature-wise
# by infinities that reproduce nan_value_treatment, allowing the shared T0/T2
# traversal to use ordinary numeric comparisons.
#' @keywords internal
catboost_qshap_matrix <- function(x, trees = NULL, float32 = TRUE) {
  if (isTRUE(float32)) {
    x <- catboost_float32_matrix(x)
  } else {
    if (!is.matrix(x)) x <- as.matrix(x)
    if (!is.numeric(x)) {
      stop("CatBoost currently requires a numeric matrix.", call. = FALSE)
    }
    # RcppEigen's MatrixXd requires double storage. The fast C++ router casts
    # only the split columns to float32 as it scans them, avoiding a full
    # float32 round-trip copy of large inputs.
    if (!is.double(x)) storage.mode(x) <- "double"
  }
  if (!is.null(trees)) {
    split_features <- unlist(lapply(trees, function(tree) {
      tree$feature[tree$children_left >= 0L]
    }), use.names = FALSE)
    if (length(split_features) && max(split_features) >= ncol(x)) {
      stop(
        "The CatBoost model references feature ", max(split_features),
        ", but x has only ", ncol(x), " columns.",
        call. = FALSE
      )
    }
  }
  if (anyNA(x)) {
    default_left <- attr(trees, "catboost_feature_default_left", exact = TRUE)
    if (is.null(default_left)) {
      stop(
        "This CatBoost explainer does not contain missing-value metadata; ",
        "recreate it with gazer() before explaining inputs with missing values.",
        call. = FALSE
      )
    }
    if (length(default_left) < ncol(x)) {
      # Trailing ignored columns may be absent from features_info and are not
      # referenced by any parsed split, so either direction is equivalent.
      default_left <- c(
        default_left,
        rep.int(TRUE, ncol(x) - length(default_left))
      )
    }
    for (j in seq_len(ncol(x))) {
      missing <- is.na(x[, j])
      if (any(missing)) {
        x[missing, j] <- if (isTRUE(default_left[j])) -Inf else Inf
      }
    }
  }
  x
}


# Predict one parsed CatBoost simple_tree.  This works for complete symmetric
# trees and for the recursive Depthwise/Lossguide layout.
#' @keywords internal
catboost_predict_simple_tree <- function(x, tree, float32_prepared = FALSE) {
  if (!isTRUE(float32_prepared)) x <- catboost_float32_matrix(x)

  n <- nrow(x)
  out <- numeric(n)
  default_left <- tree[["default_left"]]
  if (is.null(default_left)) default_left <- rep.int(FALSE, tree$node_count)

  for (i in seq_len(n)) {
    node <- 0L
    steps <- 0L
    repeat {
      steps <- steps + 1L
      if (steps > tree$node_count) {
        stop("Malformed CatBoost tree: traversal did not reach a leaf.", call. = FALSE)
      }
      pos <- node + 1L
      left <- tree$children_left[pos]
      if (left < 0L) {
        out[i] <- tree$value[pos]
        break
      }

      feature <- tree$feature[pos] + 1L
      if (feature < 1L || feature > ncol(x)) {
        stop(
          "Parsed CatBoost tree references feature ", feature - 1L,
          ", but x has only ", ncol(x), " columns.",
          call. = FALSE
        )
      }
      value <- x[i, feature]
      goes_left <- if (is.na(value)) {
        isTRUE(default_left[pos])
      } else {
        value <= tree$threshold[pos]
      }
      node <- if (goes_left) left else tree$children_right[pos]
    }
  }
  out
}


# Predict an ensemble reconstructed from CatBoost JSON.
#' @keywords internal
catboost_predict_from_trees <- function(x, trees, bias = NULL) {
  x <- catboost_float32_matrix(x)
  if (is.null(bias)) bias <- attr(trees, "bias", exact = TRUE)
  if (is.null(bias)) bias <- 0.0
  pred <- rep(as.numeric(bias), nrow(x))
  for (tree in trees) {
    pred <- pred + catboost_predict_simple_tree(x, tree, float32_prepared = TRUE)
  }
  pred
}


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
  if (!catboost_uses_oblivious_trees(explainer)) {
    return(qshap_loss_catboost_general(explainer, x, y, y_mean_ori))
  }

  x <- catboost_qshap_matrix(x, explainer$trees)
  y <- as.numeric(y)
  if (length(y) != nrow(x)) {
    stop("y must have one value per row of x.", call. = FALSE)
  }
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


# Generic CatBoost loss implementation for recursive Depthwise/Lossguide trees.
# The shared T0 and T2 implementations already support arbitrary binary tree
# layouts; unlike the symmetric backend, this path does not assume 2^depth
# leaves or one shared split per level.
#' @keywords internal
qshap_loss_catboost_general <- function(explainer, x, y, y_mean_ori = NULL) {
  x <- catboost_qshap_matrix(x, explainer$trees)
  y <- as.numeric(y)
  if (length(y) != nrow(x)) {
    stop("y must have one value per row of x.", call. = FALSE)
  }

  cb_trees <- explainer$trees
  tree_summaries <- explainer$tree_summaries
  if (is.null(tree_summaries)) {
    tree_summaries <- lapply(cb_trees, summarize_tree)
  }

  n <- nrow(x)
  p <- ncol(x)
  loss <- matrix(0.0, nrow = n, ncol = p)
  cum_pred <- rep(explainer$base_score, n)

  pb <- if (interactive()) progress::progress_bar$new(
    format = "Progress [:bar] :current/:total (:percent)",
    total = length(cb_trees),
    clear = FALSE,
    width = 60
  ) else NULL

  for (i in seq_along(cb_trees)) {
    if (!is.null(pb)) pb$tick()

    tree_i <- cb_trees[[i]]
    local_res <- y - cum_pred
    T0_x <- compute_treeshap(
      x,
      tree_i$children_left,
      tree_i$children_right,
      tree_i$feature,
      tree_i$threshold,
      tree_i$value,
      tree_i$n_node_samples
    )
    T2_x <- T2(
      x,
      tree_summaries[[i]],
      explainer$store_v_invc,
      explainer$store_z,
      FALSE
    )
    loss <- loss + T2_x - 2.0 * T0_x * local_res
    cum_pred <- cum_pred + catboost_predict_simple_tree(
      x, tree_i, float32_prepared = TRUE
    )
  }

  loss
}


# Convert CatBoost oblivious tree JSON to simple_tree format
#' @keywords internal
catboost_oblivious_to_simple <- function(tree_json, scale = 1.0,
                                         float_features = NULL) {
  splits <- tree_json$splits
  if (is.null(splits)) splits <- list()
  leaf_values <- as.numeric(unlist(tree_json$leaf_values, use.names = FALSE)) * scale
  leaf_weights <- as.numeric(unlist(tree_json$leaf_weights, use.names = FALSE))

  if (!length(leaf_values) || length(leaf_values) != length(leaf_weights) ||
      anyNA(leaf_values) || any(!is.finite(leaf_values)) ||
      anyNA(leaf_weights) || any(!is.finite(leaf_weights)) ||
      any(leaf_weights < 0)) {
    stop("Invalid scalar CatBoost leaf values or weights.", call. = FALSE)
  }

  # CatBoost trees can have empty training leaves (weight = 0).  Replace only
  # the cover with a negligible positive floor for T2; the exported value is
  # retained so future rows routed into that leaf reproduce native predictions.
  empty_mask <- leaf_weights == 0
  if (any(empty_mask)) {
    leaf_weights[empty_mask] <- catboost_cover_floor(leaf_weights)
  }

  k <- length(splits)  # tree depth
  expected_leaves <- 2^k
  if (length(leaf_values) != expected_leaves) {
    stop(
      "CatBoost oblivious tree has ", length(leaf_values),
      " leaf values; expected ", expected_leaves,
      ". Multi-output CatBoost models are not currently supported.",
      call. = FALSE
    )
  }

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

  # Every node at one depth of a symmetric tree uses the same split. Parse
  # each level once, then expand the validated metadata to BFS layout. This
  # preserves the general-tree validation while avoiding 2^depth - 1 repeated
  # R calls for every CatBoost tree.
  level_info <- lapply(seq_len(k), function(level) {
    catboost_split_info(
      splits_topdown[[level]],
      float_features,
      context = paste0("oblivious-tree depth ", level - 1L),
      round_border = FALSE
    )
  })
  level_feature <- vapply(level_info, `[[`, integer(1), "feature")
  level_threshold <- catboost_float32(vapply(
    level_info, `[[`, numeric(1), "threshold"
  ))
  level_default_left <- vapply(
    level_info, `[[`, logical(1), "default_left"
  )

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
  default_left   <- logical(total_nodes)

  # Fill internal nodes in BFS order. A depth-d split occurs 2^d times.
  internal_pos <- seq_len(num_internal)
  internal_zero_based <- internal_pos - 1L
  level_counts <- 2^(seq_len(k) - 1L)
  children_left[internal_pos] <- 2L * internal_zero_based + 1L
  children_right[internal_pos] <- 2L * internal_zero_based + 2L
  feature[internal_pos] <- rep(level_feature, times = level_counts)
  threshold[internal_pos] <- rep(level_threshold, times = level_counts)
  default_left[internal_pos] <- rep(
    level_default_left, times = level_counts
  )

  # CatBoost leaf j maps directly to the BFS leaf block.
  leaf_pos <- (num_internal + 1L):total_nodes
  children_left[leaf_pos] <- -1L
  children_right[leaf_pos] <- -1L
  feature[leaf_pos] <- -1L
  value[leaf_pos] <- leaf_values
  n_node_samples[leaf_pos] <- leaf_weights

  # Compute covers and conditional values one complete level at a time.
  for (depth in rev(seq_len(k) - 1L)) {
    node_pos <- seq.int(2^depth, 2^(depth + 1L) - 1L)
    left_pos <- 2L * node_pos
    right_pos <- left_pos + 1L
    nl <- n_node_samples[left_pos]
    nr <- n_node_samples[right_pos]
    total <- nl + nr
    n_node_samples[node_pos] <- total
    value[node_pos] <- (
      nl * value[left_pos] + nr * value[right_pos]
    ) / total
  }

  simple_tree(
    children_left  = children_left,
    children_right = children_right,
    feature        = feature,
    threshold      = threshold,
    max_depth      = as.integer(k),
    n_node_samples = n_node_samples,
    value          = value,
    node_count     = as.integer(total_nodes),
    default_left   = default_left
  )
}


# Convert CatBoost's recursive `trees` JSON representation, used by Depthwise
# and Lossguide grow policies, to the package's arbitrary binary simple_tree.
# Nodes are numbered in deterministic pre-order and children remain 0-based.
#' @keywords internal
catboost_general_to_simple <- function(tree_json, scale = 1.0,
                                       float_features = NULL) {
  nodes <- list()
  node_counter <- 0L
  actual_max_depth <- 0L

  scalar_number <- function(value, field, context, nonnegative = FALSE) {
    value <- suppressWarnings(as.numeric(unlist(value, use.names = FALSE)))
    if (length(value) != 1L || is.na(value) || !is.finite(value) ||
        (nonnegative && value < 0)) {
      stop(
        "Invalid scalar CatBoost ", field, " at ", context,
        if (nonnegative) "; expected a non-negative number." else ".",
        call. = FALSE
      )
    }
    value
  }

  collect_leaf_weights <- function(node) {
    if (!is.list(node)) return(numeric())
    if (is.null(node[["split"]])) {
      return(suppressWarnings(as.numeric(unlist(
        node[["weight"]], use.names = FALSE
      ))))
    }
    c(
      collect_leaf_weights(node[["left"]]),
      collect_leaf_weights(node[["right"]])
    )
  }
  empty_leaf_floor <- catboost_cover_floor(collect_leaf_weights(tree_json))

  traverse <- function(node, depth = 0L, path = "root") {
    if (!is.list(node)) {
      stop("Invalid CatBoost node at ", path, ".", call. = FALSE)
    }

    current <- node_counter
    node_counter <<- node_counter + 1L
    actual_max_depth <<- max(actual_max_depth, depth)

    has_split <- !is.null(node[["split"]])
    has_left <- !is.null(node[["left"]])
    has_right <- !is.null(node[["right"]])

    if (!has_split) {
      if (has_left || has_right || is.null(node[["value"]]) || is.null(node[["weight"]])) {
        stop(
          "CatBoost node at ", path,
          " is neither a valid FloatFeature split nor a scalar leaf.",
          call. = FALSE
        )
      }
      leaf_value <- scalar_number(node[["value"]], "leaf value", path) * scale
      leaf_weight <- scalar_number(
        node[["weight"]], "leaf weight", path, nonnegative = TRUE
      )

      # Empty leaves occur in real Depthwise JSON.  qshap's T2 representation
      # uses inverse child cover, so retain the native leaf value for exact
      # predictions while replacing a zero cover with a negligible positive
      # floor derived from this tree's observed cover.
      effective_weight <- if (leaf_weight == 0) empty_leaf_floor else leaf_weight
      nodes[[current + 1L]] <<- list(
        left_child = -1L,
        right_child = -1L,
        feature = -1L,
        threshold = 0.0,
        value = leaf_value,
        n_samples = effective_weight,
        default_left = FALSE
      )
      return(current)
    }

    if (!has_left || !has_right) {
      stop("CatBoost split at ", path, " is missing a child.", call. = FALSE)
    }
    split_info <- catboost_split_info(
      node[["split"]],
      float_features,
      context = paste0("general-tree node ", path)
    )
    left <- traverse(node[["left"]], depth + 1L, paste0(path, ".left"))
    right <- traverse(node[["right"]], depth + 1L, paste0(path, ".right"))
    nl <- nodes[[left + 1L]]$n_samples
    nr <- nodes[[right + 1L]]$n_samples
    cover <- nl + nr
    if (!is.finite(cover) || cover <= 0) {
      stop("Invalid CatBoost child weights at ", path, ".", call. = FALSE)
    }

    nodes[[current + 1L]] <<- list(
      left_child = left,
      right_child = right,
      feature = split_info$feature,
      threshold = split_info$threshold,
      value = (nl * nodes[[left + 1L]]$value + nr * nodes[[right + 1L]]$value) / cover,
      n_samples = cover,
      default_left = split_info$default_left
    )
    current
  }

  root <- traverse(tree_json)
  if (root != 0L || length(nodes) != node_counter) {
    stop("Failed to reconstruct the recursive CatBoost tree.", call. = FALSE)
  }

  simple_tree(
    children_left = vapply(nodes, `[[`, integer(1), "left_child"),
    children_right = vapply(nodes, `[[`, integer(1), "right_child"),
    feature = vapply(nodes, `[[`, integer(1), "feature"),
    threshold = vapply(nodes, `[[`, numeric(1), "threshold"),
    max_depth = as.integer(actual_max_depth),
    n_node_samples = vapply(nodes, `[[`, numeric(1), "n_samples"),
    value = vapply(nodes, `[[`, numeric(1), "value"),
    node_count = as.integer(length(nodes)),
    default_left = vapply(nodes, `[[`, logical(1), "default_left")
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

  if (!is.null(model_json$model_info$class_params)) {
    stop(
      "CatBoost classification models are not currently supported; ",
      "gazer() expects a scalar regression model.",
      call. = FALSE
    )
  }

  # Extract scalar scale and bias.  The current decomposition is for a scalar
  # raw formula value; reject multi-output models instead of flattening them.
  scale <- 1.0
  bias <- 0.0
  if (!is.null(model_json$scale_and_bias)) {
    sb <- model_json$scale_and_bias
    if (is.list(sb) && length(sb) >= 2) {
      scale <- as.numeric(unlist(sb[[1]], use.names = FALSE))
      bias <- as.numeric(unlist(sb[[2]], use.names = FALSE))
    }
  }
  if (length(scale) != 1L || length(bias) != 1L ||
      is.na(scale) || !is.finite(scale) || is.na(bias) || !is.finite(bias)) {
    stop(
      "Only scalar-output CatBoost models are currently supported.",
      call. = FALSE
    )
  }

  float_features <- model_json$features_info$float_features
  if (is.null(float_features)) float_features <- list()

  # CatBoost writes SymmetricTree models as `oblivious_trees`, while Depthwise
  # and Lossguide use recursive `trees` nodes.
  trees_data <- model_json$oblivious_trees
  tree_type <- "oblivious"
  converter <- catboost_oblivious_to_simple
  if (is.null(trees_data)) {
    trees_data <- model_json$trees
    tree_type <- "general"
    converter <- catboost_general_to_simple
  }
  if (is.null(trees_data)) {
    stop(
      "Could not find oblivious_trees or trees in CatBoost JSON.",
      call. = FALSE
    )
  }

  num_trees <- length(trees_data)
  result <- vector("list", num_trees)
  actual_max_depth <- 0L

  for (i in seq_len(num_trees)) {
    result[[i]] <- converter(
      trees_data[[i]],
      scale = scale,
      float_features = float_features
    )
    actual_max_depth <- max(actual_max_depth, result[[i]]$max_depth)
  }

  # Return trees and metadata
  attr(result, "bias") <- bias
  attr(result, "scale") <- scale
  attr(result, "max_depth") <- max(1L, actual_max_depth)
  attr(result, "catboost_tree_type") <- tree_type

  split_nodes <- lapply(result, function(tree) {
    if (identical(tree_type, "oblivious")) {
      if (tree$max_depth <= 0L) return(integer())
      # First BFS node at depths 0, ..., max_depth - 1 (R positions).
      return(as.integer(2^(0:(tree$max_depth - 1L))))
    }
    which(tree$children_left >= 0L)
  })
  split_features <- unlist(Map(
    function(tree, nodes) tree$feature[nodes], result, split_nodes
  ), use.names = FALSE)
  split_defaults <- unlist(Map(
    function(tree, nodes) tree$default_left[nodes], result, split_nodes
  ), use.names = FALSE)
  if (length(split_features)) {
    feature_default_left <- rep.int(TRUE, max(split_features) + 1L)
    for (feature_id in unique(split_features)) {
      directions <- unique(split_defaults[split_features == feature_id])
      if (length(directions) != 1L) {
        stop(
          "Inconsistent CatBoost missing-value direction for feature ",
          feature_id, ".",
          call. = FALSE
        )
      }
      feature_default_left[feature_id + 1L] <- directions[[1L]]
    }
  } else {
    feature_default_left <- logical(0)
  }
  attr(result, "catboost_feature_default_left") <- feature_default_left

  result
}
