library(qshap)

capture_error <- function(expr) {
  tryCatch(
    {
      force(expr)
      NULL
    },
    error = identity
  )
}

malformed_model <- list(
  current_iter = function() 1L,
  dump_model = function() "{not valid JSON"
)
parse_error <- capture_error(qshap:::lgb_formatter(malformed_model, max_depth = 1L))
stopifnot(inherits(parse_error, "error"))
stopifnot(grepl(
  "Failed to parse the fitted LightGBM trees exactly",
  conditionMessage(parse_error),
  fixed = TRUE
))

unsupported_node_model <- list(
  current_iter = function() 1L,
  dump_model = function() {
    jsonlite::toJSON(
      list(tree_info = list(list(
        tree_index = 0L,
        num_leaves = 1L,
        tree_structure = list(unknown_node = 1L)
      ))),
      auto_unbox = TRUE
    )
  }
)
unsupported_node_error <- capture_error(qshap:::lgb_formatter(
  unsupported_node_model,
  max_depth = 1L
))
stopifnot(inherits(unsupported_node_error, "error"))
stopifnot(grepl(
  "neither a supported split node nor a leaf",
  conditionMessage(unsupported_node_error),
  fixed = TRUE
))

tree <- qshap:::simple_tree(
  children_left = c(1L, -1L, -1L),
  children_right = c(2L, -1L, -1L),
  feature = c(0L, -1L, -1L),
  threshold = c(0, 0, 0),
  max_depth = 1L,
  n_node_samples = c(4, 2, 2),
  value = c(0, -1, 1),
  node_count = 3L
)
explainer <- list(
  model = structure(list(), class = "qshap_lightgbm_prediction_failure"),
  trees = list(tree),
  tree_summaries = list(qshap:::summarize_tree(tree)),
  store_v_invc = qshap:::store_complex_v_invc(2L),
  store_z = qshap:::store_complex_root(2L)
)

predict.qshap_lightgbm_prediction_failure <- function(object, ...) {
  stop("deliberate per-tree prediction failure")
}

contribution_error <- capture_error(qshap:::qshap_loss_lightgbm(
  explainer,
  x = matrix(c(-1, 1), ncol = 1L),
  y = c(-1, 1)
))
stopifnot(inherits(contribution_error, "error"))
stopifnot(grepl(
  "Failed to compute exact per-tree LightGBM contributions for tree 1",
  conditionMessage(contribution_error),
  fixed = TRUE
))
stopifnot(grepl(
  "deliberate per-tree prediction failure",
  conditionMessage(contribution_error),
  fixed = TRUE
))
