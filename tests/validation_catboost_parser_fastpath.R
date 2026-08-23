suppressPackageStartupMessages(library(qshap))

validate_catboost_parser_fastpath <- function(depth = 8L, p = 4L) {
  namespace <- asNamespace("qshap")
  original <- get("catboost_split_info", envir = namespace, inherits = FALSE)
  calls <- 0L
  counted <- function(...) {
    calls <<- calls + 1L
    original(...)
  }
  assignInNamespace("catboost_split_info", counted, ns = "qshap")
  on.exit(
    assignInNamespace("catboost_split_info", original, ns = "qshap"),
    add = TRUE
  )

  splits <- lapply(seq_len(depth), function(level) {
    list(
      split_type = "FloatFeature",
      float_feature_index = as.integer((level - 1L) %% p),
      border = level / 10
    )
  })
  float_features <- lapply(seq_len(p), function(feature) {
    list(
      flat_feature_index = as.integer(feature - 1L),
      nan_value_treatment = "AsFalse"
    )
  })
  num_leaves <- 2^depth
  tree_json <- list(
    splits = splits,
    leaf_values = seq_len(num_leaves) / num_leaves,
    leaf_weights = rep(1, num_leaves)
  )

  converter <- get("catboost_oblivious_to_simple", envir = namespace)
  tree <- converter(tree_json, float_features = float_features)

  stopifnot(calls == depth)
  stopifnot(tree$node_count == 2^(depth + 1L) - 1L)
  for (tree_depth in 0:(depth - 1L)) {
    positions <- seq.int(2^tree_depth, 2^(tree_depth + 1L) - 1L)
    stopifnot(length(unique(tree$feature[positions])) == 1L)
    stopifnot(length(unique(tree$threshold[positions])) == 1L)
  }
}

validate_catboost_parser_fastpath()
message("CatBoost SymmetricTree parser fast-path validation passed.")
