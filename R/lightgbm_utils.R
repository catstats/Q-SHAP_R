#' @import lightgbm
#' @importFrom stats predict
NULL

# Loss implementation for LightGBM model
#' @keywords internal
qshap_loss_lightgbm <- function(explainer, x, y, y_mean_ori = NULL) {
  # Check if lightgbm is available
  if (!requireNamespace("lightgbm", quietly = TRUE)) {
    stop("lightgbm package is required for LightGBM support. Please install it with: install.packages('lightgbm')")
  }
  
  model <- explainer$model
  store_v_invc <- explainer$store_v_invc
  store_z <- explainer$store_z
  lgb_trees <- explainer$trees
  tree_summaries <- explainer$tree_summaries
  if (is.null(tree_summaries)) {
    tree_summaries <- lapply(lgb_trees, summarize_tree)
  }
  
  num_tree <- length(lgb_trees)
  
  if (!is.matrix(x)) {
    x <- as.matrix(x)
  }
  n <- nrow(x)
  p <- ncol(x)
  loss <- matrix(0, nrow = n, ncol = p)
  cum_pred <- numeric(n)

    pb <- if (interactive()) progress::progress_bar$new(
      format = "Progress [:bar] :current/:total (:percent)",
      total = num_tree,
      clear = FALSE,
      width = 60
    ) else NULL

  for (i in seq_len(num_tree)) {

    if (!is.null(pb)) pb$tick()
    local_res <- y - cum_pred
    
    # Calculate real SHAP values using LightGBM's built-in SHAP functionality
    # This is equivalent to XGBoost's predcontrib=TRUE and Python's explainer.shap_values(x)
    tryCatch({
      if (i == 1) {
        # For the first tree, get SHAP values from just the first iteration
        shap_contrib_matrix <- stats::predict(model, x, type = "contrib", num_iteration = 1)
        tree_pred <- rowSums(shap_contrib_matrix)
        # Remove the bias column (last column) to get just feature contributions
        T0_x_tree <- shap_contrib_matrix[, -ncol(shap_contrib_matrix), drop = FALSE]
      } else {
        shap_i <- stats::predict(
            model, x,
            type = "contrib",
            start_iteration = i - 1,  # LightGBM is 0-based here
            num_iteration   = 1
          )

        tree_pred <- rowSums(shap_i)
        # remove bias column
        T0_x_tree <- shap_i[, -ncol(shap_i), drop = FALSE]
  
      }
    }, error = function(e) {
      stop(
        "Failed to compute exact per-tree LightGBM contributions for tree ",
        i,
        "; this model/version is not currently supported. Details: ",
        conditionMessage(e),
        call. = FALSE
      )
    })
    
    summary_tree <- tree_summaries[[i]]
    
    # Call C++ loss_treeshap with real SHAP values and correct residuals
    current_tree_loss <- loss_treeshap(x, local_res, summary_tree, store_v_invc, store_z, T0_x_tree, 1.0)
    
    if (i == 1) {
      loss <- current_tree_loss
    } else {
      loss <- loss + current_tree_loss
    }

    cum_pred <- cum_pred + tree_pred
  }
  
  loss
}


# Formats a LightGBM model into a list of simple_tree objects
#' @keywords internal
lgb_formatter <- function(lgb_model, max_depth) {
  # Check if lightgbm is available
  if (!requireNamespace("lightgbm", quietly = TRUE)) {
    stop("lightgbm package is required for LightGBM support. Please install it with: install.packages('lightgbm')")
  }
  
  # Get number of trees from model
  num_trees <- lgb_model$current_iter()
  if (is.null(num_trees) || num_trees <= 0) {
    stop("Could not determine number of trees in LightGBM model")
  }
  
  # Exact decomposition requires the fitted tree structures. Never substitute
  # synthetic trees when the model dump cannot be parsed.
  tryCatch({
    if (!requireNamespace("jsonlite", quietly = TRUE)) {
      stop("jsonlite package required for tree parsing")
    }
    dump_json <- lgb_model$dump_model()
    parsed_model <- jsonlite::fromJSON(dump_json)
    tree_info_df <- parsed_model$tree_info

    if (!is.data.frame(tree_info_df) || nrow(tree_info_df) != num_trees) {
      stop(
        "Expected ", num_trees, " trees in the model dump, found ",
        if (is.data.frame(tree_info_df)) nrow(tree_info_df) else 0L
      )
    }

    lapply(seq_len(num_trees), function(tree_idx) {
      tree_structure_df <- tree_info_df[tree_idx, ]$tree_structure
      if (!is.data.frame(tree_structure_df) || nrow(tree_structure_df) != 1L) {
        stop("Tree ", tree_idx, " does not contain exactly one root node")
      }

      lgb_tree_to_simple(tree_structure_df[1, ], max_depth)
    })
  }, error = function(e) {
    stop(
      "Failed to parse the fitted LightGBM trees exactly; ",
      "this model/version is not currently supported. Details: ",
      conditionMessage(e),
      call. = FALSE
    )
  })
}

# Convert LightGBM tree structure to simple_tree format
#' @keywords internal
lgb_tree_to_simple <- function(tree_structure, max_depth) {
  # tree_structure is a data frame row, need to access the actual tree
  root_node <- tree_structure

  required_numeric <- function(node_data, field, node_index) {
    if (!field %in% names(node_data)) {
      stop("Node ", node_index, " is missing required field '", field, "'")
    }

    value <- node_data[[field]]
    if (length(value) != 1L || !is.numeric(value) || is.na(value) || !is.finite(value)) {
      stop("Node ", node_index, " has an invalid '", field, "' value")
    }
    as.numeric(value)
  }

  required_child <- function(node_data, field, node_index) {
    if (!field %in% names(node_data)) {
      stop("Node ", node_index, " is missing required field '", field, "'")
    }

    child <- node_data[[field]]
    if (!is.data.frame(child) || nrow(child) != 1L) {
      stop("Node ", node_index, " has an invalid '", field, "' node")
    }
    child
  }
  
  # Initialize storage for tree nodes
  nodes <- list()
  node_counter <- 0
  
  # Recursive function to traverse LightGBM tree structure
  traverse_lgb_tree <- function(node_data) {
    node_counter <<- node_counter + 1
    current_idx <- node_counter
    
    # Initialize node information
    node_info <- list(
      index = current_idx - 1,  # 0-based indexing for C++
      left_child = -1,
      right_child = -1,
      feature = -1,
      threshold = 0.0,
      value = 0.0,
      n_samples = NA_real_
    )

    node_index <- current_idx - 1L
    
    # Check if this is a leaf node by looking for leaf_value.
    if ("leaf_value" %in% names(node_data) &&
        length(node_data$leaf_value) > 0 &&
        !is.na(node_data$leaf_value[1])) {
      # Leaf node
      node_info$value <- required_numeric(node_data, "leaf_value", node_index)
      node_info$n_samples <- required_numeric(node_data, "leaf_count", node_index)
    } else if ("split_feature" %in% names(node_data) &&
         length(node_data$split_feature) > 0 &&
         !is.na(node_data$split_feature[1])) {
      # This parser currently supports numeric threshold splits only.
      if (!"decision_type" %in% names(node_data) ||
          length(node_data$decision_type) != 1L ||
          !identical(node_data$decision_type, "<=")) {
        stop("Node ", node_index, " uses an unsupported LightGBM split type")
      }

      split_feature <- required_numeric(node_data, "split_feature", node_index)
      if (split_feature < 0 || split_feature != floor(split_feature)) {
        stop("Node ", node_index, " has an invalid 'split_feature' value")
      }
      node_info$feature <- as.integer(split_feature)
      node_info$threshold <- required_numeric(node_data, "threshold", node_index)
      node_info$value <- required_numeric(node_data, "internal_value", node_index)
      node_info$n_samples <- required_numeric(node_data, "internal_count", node_index)

      if (!"default_left" %in% names(node_data) ||
          length(node_data$default_left) != 1L ||
          !is.logical(node_data$default_left) ||
          is.na(node_data$default_left)) {
        stop("Node ", node_index, " has an invalid 'default_left' value")
      }
      node_info$default_left <- node_data$default_left

      left_child_df <- required_child(node_data, "left_child", node_index)
      right_child_df <- required_child(node_data, "right_child", node_index)
      node_info$left_child <- traverse_lgb_tree(left_child_df[1, ])
      node_info$right_child <- traverse_lgb_tree(right_child_df[1, ])
    } else {
      stop("Node ", node_index, " is neither a supported split node nor a leaf")
    }
    
    nodes[[current_idx]] <<- node_info
    return(current_idx - 1)  # Return 0-based index
  }
  
  # Start traversal from root
  traverse_lgb_tree(root_node)
  
  # Convert to arrays expected by simple_tree
  node_count <- length(nodes)
  children_left <- integer(node_count)
  children_right <- integer(node_count)
  feature <- integer(node_count)
  threshold <- numeric(node_count)
  n_node_samples <- numeric(node_count)
  value <- numeric(node_count)
  default_left <- logical(node_count)
  
  for (i in seq_len(node_count)) {
    idx <- i  # 1-based for R arrays
    node <- nodes[[i]]
    
    children_left[idx] <- node$left_child
    children_right[idx] <- node$right_child
    feature[idx] <- node$feature
    threshold[idx] <- node$threshold
    n_node_samples[idx] <- node$n_samples
    value[idx] <- node$value
    default_left[idx] <- isTRUE(node$default_left)
  }
  
  # Create simple_tree object
  simple_tree(
    children_left = children_left,
    children_right = children_right,
    feature = feature,
    threshold = threshold,
    max_depth = max_depth,
    n_node_samples = n_node_samples,
    value = value,
    node_count = node_count,
    default_left = default_left
  )
}
