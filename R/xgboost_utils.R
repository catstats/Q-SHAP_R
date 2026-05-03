#' @importFrom xgboost xgb.model.dt.tree
#' @importFrom stats predict
NULL


# Loss implementation for xgboost model
#' @keywords internal
qshap_loss_xgboost <- function(explainer, x, y, y_mean_ori = NULL) {
  model <- explainer$model
  store_v_invc <- explainer$store_v_invc
  store_z <- explainer$store_z
  base_score <- explainer$base_score
  xgb_trees <- explainer$trees # This is a list of simple_tree objects
  tree_summaries <- explainer$tree_summaries
  if (is.null(tree_summaries)) {
    tree_summaries <- lapply(xgb_trees, summarize_tree)
  }
  
  num_tree <- length(xgb_trees)
  
  if (!is.matrix(x)) {
    x <- as.matrix(x)
  }
  n <- nrow(x)
  p <- ncol(x)
  loss <- matrix(0, nrow = n, ncol = p)
  pb <- if (interactive()) progress::progress_bar$new(
      format = "Progress [:bar] :current/:total (:percent)",
      total = num_tree,
      clear = FALSE,
      width = 60
    ) else NULL

  cum_pred <- rep(base_score, n)
  # XGBoost R uses 0-based indexing for iterationrange with exclusive end
  # c(0, 1) = tree 0 only, c(0, 2) = trees 0+1, etc.
  # Note: c(0, n) and c(1, n) are equivalent (both give trees 0 to n-1)
  # also enable slicing by tree index

  # XGBoost R uses 1-based indexing for model list !!!!
  # https://xgboost.readthedocs.io/en/stable/tutorials/slicing_model.html

  for (i in seq_len(num_tree)) { # i is the 1-based loop index; tree index is i-1 (0-based)
  
    if (!is.null(pb)) pb$tick()

    local_res <- y - cum_pred

    shap_tree_i <- stats::predict(model[i], x, predcontrib = TRUE)
    tree_pred <- rowSums(shap_tree_i) - base_score
    T0_x_tree <- shap_tree_i[, -ncol(shap_tree_i), drop = FALSE]
    summary_tree <- tree_summaries[[i]]
    
    # Call C++ loss_treeshap with per-tree SHAP values (T0_x_tree) and correct residuals (local_res)
    # The learning rate for individual XGBoost trees in this SHAP context is effectively 1.0,
    # as tree outputs are already scaled.
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


# Formats an xgboost model into a list of simple_tree objects
# Becareful that this part is different from the Python version, we scale leaf weights here 
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

    # base_weights in XGBoost JSON are already the final tree outputs (eta-scaled during training)
    # Do NOT scale again here - the values are ready to use as-is
    base_w <- as.numeric(unlist(tr$base_weights))

    out[[i]] <- simple_tree(
      children_left  = as.integer(unlist(tr$left_children)),
      children_right = as.integer(unlist(tr$right_children)),
      feature        = as.integer(unlist(tr$split_indices)),
      threshold      = as.numeric(unlist(tr$split_conditions)),
      max_depth      = as.integer(max_depth),
      n_node_samples = as.numeric(unlist(tr$sum_hessian)),
      value          = base_w,
      node_count     = as.integer(tr$tree_param$num_nodes)
    )
  }

  out
}
