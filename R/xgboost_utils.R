#' @importFrom xgboost xgb.model.dt.tree
NULL

# Loss implementation for xgboost model
#' @keywords internal
qshap_loss_xgboost <- function(explainer, x, y, y_mean_ori = NULL) {
  model <- explainer$model
  store_v_invc <- explainer$store_v_invc
  store_z <- explainer$store_z
  base_score <- explainer$base_score
  xgb_trees <- explainer$trees # This is a list of simple_tree objects
  
  num_tree <- length(xgb_trees)
  loss <- matrix(0, nrow = nrow(x), ncol = ncol(x))
  
  if (!is.matrix(x)) {
    x <- as.matrix(x)
  }

  # iterations are 1-based in xgboost predict
  for (i in seq_len(num_tree)) { # i is the 1-based index of the current tree (round i)
    
    local_res <- NULL 
    
    if (i == 1) { # For the first tree (round 1)
      local_res <- y - base_score
      
      # SHAP values for the first tree (round 1 only)
      # iterationrange = c(1, 2) means use tree from round 1 up to (but not including) round 2. So, only round 1.
      shap_round_1_cumulative <- predict(model, x, predcontrib = TRUE, iterationrange = c(1, 2))
      T0_x_tree <- shap_round_1_cumulative[, -ncol(shap_round_1_cumulative), drop = FALSE]
    } else { # For subsequent trees (round i, where i > 1)
      # Calculate residual: y - prediction_from_rounds_1_to_(i-1)
      # iterationrange = c(1, i) means use trees from round 1 up to (but not including) round i. So, rounds 1 to i-1.
      pred_partial <- predict(model, x, iterationrange = c(1, i))
      local_res <- y - pred_partial
      
      # SHAP values for current tree (round i)
      # SHAP from rounds 1 to i: iterationrange = c(1, i + 1)
      shap_total_up_to_round_i <- predict(model, x, predcontrib = TRUE, iterationrange = c(1, i + 1))
      # SHAP from rounds 1 to i-1: iterationrange = c(1, i)
      shap_total_up_to_round_i_minus_1 <- predict(model, x, predcontrib = TRUE, iterationrange = c(1, i))
      
      # Marginal SHAP contribution of tree i (remove bias columns)
      T0_x_tree <- shap_total_up_to_round_i[, -ncol(shap_total_up_to_round_i), drop = FALSE] - 
                   shap_total_up_to_round_i_minus_1[, -ncol(shap_total_up_to_round_i_minus_1), drop = FALSE]
    }
    
    # xgb_trees is a 1-indexed list in R. xgb_trees[[i]] is the tree for round i.
    summary_tree <- summarize_tree(xgb_trees[[i]])
    
    # Call C++ loss_treeshap with per-tree SHAP values (T0_x_tree) and correct residuals (local_res)
    # The learning rate for individual XGBoost trees in this SHAP context is effectively 1.0,
    # as tree outputs are already scaled.
    current_tree_loss <- loss_treeshap(x, local_res, summary_tree, store_v_invc, store_z, T0_x_tree, 1.0)
    
    if (i == 1) {
      loss <- current_tree_loss
    } else {
      loss <- loss + current_tree_loss
    }
  }
  
  loss
}


# Formats an xgboost model into a list of simple_tree objects
#' @keywords internal
xgb_formatter <- function(model_json, max_depth) {
  # model_json can be:
  #  (1) a parsed list from jsonlite::fromJSON(..., simplifyVector = FALSE), or
  #  (2) a filename to a JSON model file, or
  #  (3) an xgb.Booster (we'll save->read automatically)

  # case (3): booster
  if (inherits(model_json, "xgb.Booster")) {
    fn <- tempfile(fileext = ".json")
    xgboost::xgb.save(model_json, fn)
    model_json <- jsonlite::fromJSON(fn, simplifyVector = FALSE)
    unlink(fn)
  }

  # case (2): filename
  if (is.character(model_json) && length(model_json) == 1L && file.exists(model_json)) {
    model_json <- jsonlite::fromJSON(model_json, simplifyVector = FALSE)
  }

  # case (1): parsed json list
  if (!is.list(model_json)) stop("model_json must be xgb.Booster, a parsed JSON list, or a JSON filename.")

  # ---- robust tree path (xgboost JSON differs across builds) ----
  gb <- model_json$learner$gradient_booster
  if (is.null(gb)) stop("JSON missing learner$gradient_booster")

  # common locations:
  # A) gb$model$trees
  trees_data <- gb$model$trees

  # B) gb$gbtree_model$trees (seen in many versions)
  if (is.null(trees_data)) trees_data <- gb$gbtree_model$trees

  # C) gb$model$gbtree_model_param + gb$model$trees (rare combos)
  if (is.null(trees_data) && !is.null(gb$model) && !is.null(gb$model$trees)) trees_data <- gb$model$trees

  if (is.null(trees_data)) {
    stop("Could not find trees in JSON. Inspect with names(model_json$learner$gradient_booster).")
  }

  out <- vector("list", length(trees_data))

  for (i in seq_along(trees_data)) {
    tr <- trees_data[[i]]

    out[[i]] <- simple_tree(
      children_left  = as.integer(unlist(tr$left_children)),
      children_right = as.integer(unlist(tr$right_children)),
      feature        = as.integer(unlist(tr$split_indices)),
      threshold      = as.numeric(unlist(tr$split_conditions)),
      max_depth      = as.integer(max_depth),
      n_node_samples = as.numeric(unlist(tr$sum_hessian)),
      value          = as.numeric(unlist(tr$base_weights)),
      node_count     = as.integer(tr$tree_param$num_nodes)
    )
  }

  out
}