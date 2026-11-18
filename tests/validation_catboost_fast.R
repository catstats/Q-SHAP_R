if (!requireNamespace("catboost", quietly = TRUE)) {
  message("catboost is not installed; skipping CatBoost backend validation.")
  quit(save = "no", status = 0)
}

library(qshap)

parsed_predict <- function(explainer, x) {
  qshap:::catboost_predict_from_trees(
    x,
    explainer$trees,
    explainer$base_score
  )
}

fit_catboost <- function(pool, grow_policy, seed, extra = list()) {
  params <- utils::modifyList(
    list(
      loss_function = "RMSE",
      iterations = 4,
      depth = 3,
      learning_rate = 0.15,
      grow_policy = grow_policy,
      verbose = 0,
      allow_writing_files = FALSE,
      random_seed = seed,
      thread_count = 1
    ),
    extra
  )
  if (identical(grow_policy, "Lossguide") && is.null(params$max_leaves)) {
    params$max_leaves <- 6
  }
  catboost::catboost.train(pool, params = params)
}

set.seed(20260512)
n <- 90
p <- 6
X <- matrix(rnorm(n * p), nrow = n, ncol = p)
y <- 2 * X[, 1] - 1.3 * X[, 3] + 0.4 * X[, 4] * X[, 6] +
  rnorm(n, sd = 0.1)
pool <- catboost::catboost.load_pool(data = X, label = y)
sst <- sum((y - mean(y))^2)

for (policy in c("SymmetricTree", "Depthwise", "Lossguide")) {
  # CatBoost indexes ignored_features from zero.  Keeping two gaps exercises
  # float_feature_index -> flat_feature_index mapping in both JSON layouts.
  model <- fit_catboost(
    pool,
    policy,
    seed = 20260512,
    extra = list(ignored_features = c(1, 4))
  )
  explainer <- gazer(model)
  native_prediction <- catboost::catboost.predict(model, pool)
  reconstructed_prediction <- parsed_predict(explainer, X)

  expected_type <- if (identical(policy, "SymmetricTree")) "oblivious" else "general"
  stopifnot(identical(
    attr(explainer$trees, "catboost_tree_type", exact = TRUE),
    expected_type
  ))
  stopifnot(max(abs(native_prediction - reconstructed_prediction)) < 1e-10)

  # Models trained without missing values export AsIs.  CatBoost still routes
  # a later missing value to the false/left branch; parsed routing must agree.
  if (identical(policy, "Depthwise")) {
    X_as_is <- X[seq_len(8L), , drop = FALSE]
    root_feature <- explainer$trees[[1L]]$feature[1L] + 1L
    X_as_is[1L, root_feature] <- NA_real_
    as_is_pool <- catboost::catboost.load_pool(X_as_is)
    stopifnot(max(abs(
      catboost::catboost.predict(model, as_is_pool) -
        parsed_predict(explainer, X_as_is)
    )) < 1e-10)
  }

  split_features <- unlist(lapply(
    explainer$trees,
    function(tree) tree$feature[tree$children_left >= 0L]
  ))
  stopifnot(!any(split_features %in% c(1L, 4L)))
  stopifnot(all(split_features >= 0L & split_features < ncol(X)))

  global <- rsq(explainer, X, y, local = FALSE, sd_out = TRUE)
  local <- rsq(explainer, X, y, local = TRUE, sd_out = TRUE)
  stopifnot(max(abs(global$rsq - local$rsq)) < 1e-9)
  stopifnot(max(abs(global$sd_rsq - local$sd_rsq)) < 1e-9)
  stopifnot(max(abs(local$local_rsq + local$loss / sst)) < 1e-12)
  stopifnot(max(abs(colSums(local$local_rsq) - local$rsq)) < 1e-9)

  # The per-feature decomposition must add back to the fitted model R2 up to
  # the small numerical intercept remainder already present in CatBoost trees.
  fitted_rsq <- 1 - sum((y - native_prediction)^2) / sst
  stopifnot(abs(sum(global$rsq) - fitted_rsq) < 5e-4)

  if (identical(policy, "SymmetricTree")) {
    # Cross-check the optimized leaf-grouped implementation against the same
    # arbitrary-tree path used as the non-symmetric fallback.
    generic_loss <- qshap:::qshap_loss_catboost_general(explainer, X, y)
    stopifnot(max(abs(local$loss - generic_loss)) < 1e-9)
  } else if (identical(policy, "Lossguide")) {
    # Lossguide's leaf budget should produce at least one non-complete tree,
    # proving the fallback is not accidentally assuming a 2^depth layout.
    is_complete <- vapply(explainer$trees, function(tree) {
      tree$node_count == 2^(tree$max_depth + 1L) - 1L
    }, logical(1))
    stopifnot(any(!is_complete))
  }
}

# CatBoost applies a scalar scale to all exported tree values and adds bias to
# the ensemble.  Exercise non-default values in both JSON layouts by editing a
# saved native model, then loading it through CatBoost itself.
for (policy in c("SymmetricTree", "Depthwise", "Lossguide")) {
  model <- fit_catboost(pool, policy, seed = 20260515)
  scaled_file <- tempfile(fileext = ".json")
  catboost::catboost.save_model(model, scaled_file, file_format = "json")
  scaled_json <- jsonlite::fromJSON(scaled_file, simplifyVector = FALSE)
  scaled_json$scale_and_bias <- list(1.7, list(-0.3))
  jsonlite::write_json(
    scaled_json,
    scaled_file,
    auto_unbox = TRUE,
    digits = NA,
    null = "null"
  )
  scaled_model <- catboost::catboost.load_model(
    scaled_file,
    file_format = "json"
  )
  unlink(scaled_file)

  scaled_explainer <- gazer(scaled_model)
  scaled_native <- catboost::catboost.predict(scaled_model, pool)
  scaled_parsed <- parsed_predict(scaled_explainer, X)
  stopifnot(identical(attr(scaled_explainer$trees, "scale"), 1.7))
  stopifnot(identical(scaled_explainer$base_score, -0.3))
  stopifnot(max(abs(scaled_native - scaled_parsed)) < 1e-12)

  scaled_global <- rsq(
    scaled_explainer, X, y, local = FALSE, sd_out = TRUE
  )
  scaled_local <- rsq(
    scaled_explainer, X, y, local = TRUE, sd_out = TRUE
  )
  stopifnot(max(abs(scaled_global$rsq - scaled_local$rsq)) < 1e-9)
  stopifnot(max(abs(scaled_global$sd_rsq - scaled_local$sd_rsq)) < 1e-9)
  stopifnot(max(abs(colSums(scaled_local$local_rsq) - scaled_local$rsq)) < 1e-9)
}

# Missing-value directions are parsed for both JSON layouts and are verified
# against native CatBoost predictions and the Q-SHAP fallback.
set.seed(20260513)
X_missing <- matrix(rnorm(80 * 5), nrow = 80, ncol = 5)
X_missing[sample(length(X_missing), 28)] <- NA_real_
y_missing <- ifelse(is.na(X_missing[, 1]), 1.0, X_missing[, 1]) -
  0.7 * ifelse(is.na(X_missing[, 3]), 0.0, X_missing[, 3]) +
  rnorm(nrow(X_missing), sd = 0.1)
missing_pool <- catboost::catboost.load_pool(X_missing, label = y_missing)

missing_cases <- list(
  list(policy = "SymmetricTree", nan_mode = "Min", default_left = TRUE),
  list(policy = "Depthwise", nan_mode = "Max", default_left = FALSE)
)

for (case in missing_cases) {
  model <- fit_catboost(
    missing_pool,
    case$policy,
    seed = 20260513,
    extra = list(nan_mode = case$nan_mode)
  )
  explainer <- gazer(model)
  native_prediction <- catboost::catboost.predict(model, missing_pool)
  reconstructed_prediction <- parsed_predict(explainer, X_missing)
  stopifnot(max(abs(native_prediction - reconstructed_prediction)) < 1e-10)

  internal_defaults <- unlist(lapply(explainer$trees, function(tree) {
    tree$default_left[tree$children_left >= 0L]
  }))
  stopifnot(length(internal_defaults) > 0L)
  stopifnot(all(internal_defaults == case$default_left))

  global <- rsq(explainer, X_missing, y_missing, local = FALSE, sd_out = FALSE)
  local <- rsq(explainer, X_missing, y_missing, local = TRUE, sd_out = FALSE)
  stopifnot(max(abs(global$rsq - local$rsq)) < 1e-9)
  stopifnot(max(abs(colSums(local$local_rsq) - local$rsq)) < 1e-9)
}

# CatBoost compares float32-quantized values.  A double just above a JSON
# border can round back to the border and must therefore remain on the left.
set.seed(20260514)
X_boundary_train <- matrix(seq(-2, 2, length.out = 100), ncol = 1L)
y_boundary_train <- ifelse(X_boundary_train[, 1] > 0.25, 2.0, -1.0)
boundary_pool <- catboost::catboost.load_pool(
  X_boundary_train,
  label = y_boundary_train
)

for (policy in c("SymmetricTree", "Depthwise", "Lossguide")) {
  model <- fit_catboost(
    boundary_pool,
    policy,
    seed = 20260514,
    extra = list(iterations = 1, depth = 1, max_leaves = 2)
  )
  explainer <- gazer(model)
  border <- explainer$trees[[1L]]$threshold[1L]
  delta <- if (border == 0) 1e-46 else abs(border) * .Machine$double.eps
  just_above <- border + delta
  if (!(just_above > border)) {
    just_above <- border + max(abs(border), 1.0) * .Machine$double.eps
  }
  stopifnot(just_above > border)
  stopifnot(identical(
    qshap:::catboost_float32(just_above),
    qshap:::catboost_float32(border)
  ))

  X_boundary <- matrix(c(border, just_above, border + 1e-3), ncol = 1L)
  native <- catboost::catboost.predict(
    model,
    catboost::catboost.load_pool(X_boundary)
  )
  parsed <- parsed_predict(explainer, X_boundary)
  stopifnot(max(abs(native - parsed)) < 1e-10)
  stopifnot(identical(native[1L], native[2L]))
}

# Do not reinterpret categorical/CTR split identifiers as numeric columns.
unsupported_error <- tryCatch(
  {
    qshap:::catboost_split_info(
      list(split_type = "OnlineCtr", split_index = 0L),
      float_features = list(),
      context = "validation split"
    )
    ""
  },
  error = conditionMessage
)
stopifnot(grepl("only numeric FloatFeature splits", unsupported_error, fixed = TRUE))

missing_mapping_error <- tryCatch(
  {
    qshap:::catboost_split_info(
      list(
        split_type = "FloatFeature",
        float_feature_index = 0L,
        border = 0.0
      ),
      float_features = list(list(
        feature_index = 0L,
        nan_value_treatment = "AsFalse"
      )),
      context = "validation split"
    )
    ""
  },
  error = conditionMessage
)
stopifnot(grepl("refusing to guess an input column", missing_mapping_error,
                fixed = TRUE))

# Classification has vector/probability semantics that are not part of the
# scalar R2 decomposition; reject it before returning a misleading explainer.
classification_pool <- catboost::catboost.load_pool(
  X_boundary_train,
  label = as.integer(y_boundary_train > 0)
)
classification_model <- catboost::catboost.train(
  classification_pool,
  params = list(
    loss_function = "Logloss",
    iterations = 1,
    depth = 1,
    verbose = 0,
    allow_writing_files = FALSE,
    random_seed = 20260514,
    thread_count = 1
  )
)
classification_error <- tryCatch(
  {
    gazer(classification_model)
    ""
  },
  error = conditionMessage
)
stopifnot(grepl("classification models are not currently supported", classification_error,
                fixed = TRUE))

# Empty CatBoost leaves need positive cover for Q-SHAP path probabilities, but
# assigning them a whole synthetic sample measurably changes the result.  Both
# JSON layouts use a tiny cover relative to the tree total while preserving the
# exact exported leaf prediction.
synthetic_float_features <- list(list(
  flat_feature_index = 0L,
  nan_value_treatment = "AsFalse"
))
synthetic_split <- list(
  split_type = "FloatFeature",
  float_feature_index = 0L,
  border = 0.0
)
synthetic_oblivious <- qshap:::catboost_oblivious_to_simple(
  list(
    splits = list(synthetic_split),
    leaf_values = list(1.0, 9.0),
    leaf_weights = list(40.0, 0.0)
  ),
  float_features = synthetic_float_features
)
synthetic_general <- qshap:::catboost_general_to_simple(
  list(
    split = synthetic_split,
    left = list(value = 1.0, weight = 40.0),
    right = list(value = 9.0, weight = 0.0)
  ),
  float_features = synthetic_float_features
)
synthetic_x <- matrix(c(-1.0, 1.0), ncol = 1L)
for (synthetic_tree in list(synthetic_oblivious, synthetic_general)) {
  leaf_cover <- synthetic_tree$n_node_samples[
    synthetic_tree$children_left < 0L
  ]
  stopifnot(any(abs(leaf_cover - 40e-12) < 1e-20))
  stopifnot(max(leaf_cover[leaf_cover < 1.0]) < 1e-6)
  stopifnot(identical(
    qshap:::catboost_predict_simple_tree(synthetic_x, synthetic_tree),
    c(1.0, 9.0)
  ))
}

# Reject an underspecified input matrix before either native traversal can
# dereference a split feature outside x.
feature_bounds_error <- tryCatch(
  {
    qshap:::catboost_qshap_matrix(
      matrix(numeric(), nrow = 2L, ncol = 0L),
      list(synthetic_oblivious)
    )
    ""
  },
  error = conditionMessage
)
stopifnot(grepl("x has only 0 columns", feature_bounds_error, fixed = TRUE))

# The native symmetric kernel also validates structure defensively.  A complete
# general tree can have the same node count as an oblivious tree but different
# splits within one level; shape alone must never select the fast algorithm.
malformed_fast_tree <- synthetic_oblivious
malformed_fast_tree$max_depth <- 2L
malformed_fast_tree$node_count <- 7L
malformed_fast_tree$children_left <- c(1L, 3L, 5L, -1L, -1L, -1L, -1L)
malformed_fast_tree$children_right <- c(2L, 4L, 6L, -1L, -1L, -1L, -1L)
malformed_fast_tree$feature <- c(0L, 0L, 1L, -1L, -1L, -1L, -1L)
malformed_fast_tree$threshold <- c(0.0, -0.5, 0.5, 0.0, 0.0, 0.0, 0.0)
malformed_fast_tree$n_node_samples <- c(40, 20, 20, 10, 10, 10, 10)
malformed_fast_tree$value <- c(0, 0, 0, 1, 2, 3, 4)
malformed_fast_error <- tryCatch(
  {
    qshap:::catboost_qshap_r2_fast(
      matrix(c(-1, -1, 1, 1), nrow = 2L, byrow = TRUE),
      c(0, 1),
      list(malformed_fast_tree),
      0.0,
      FALSE
    )
    ""
  },
  error = conditionMessage
)
stopifnot(grepl("one shared split per level", malformed_fast_error,
                fixed = TRUE))

# General CatBoost trees can use more than 20 distinct split features. Verify
# that the polynomial TreeSHAP backend and the general Q-SHAP fallback remain
# finite and agree with native CatBoost outputs in this wider setting.
compute_polynomial_treeshap <- function(tree, x) {
  qshap:::compute_treeshap(
    x,
    tree$children_left,
    tree$children_right,
    tree$feature,
    tree$threshold,
    tree$value,
    tree$n_node_samples
  )
}

set.seed(2205)
n_wide <- 1200L
p_wide <- 100L
X_wide <- matrix(stats::rnorm(n_wide * p_wide), nrow = n_wide, ncol = p_wide)
y_wide <- rowSums(X_wide[, seq_len(60L), drop = FALSE]) +
  0.05 * stats::rnorm(n_wide)
wide_pool <- catboost::catboost.load_pool(X_wide, label = y_wide)

for (policy in c("Depthwise", "Lossguide")) {
  params <- list(
    loss_function = "RMSE",
    iterations = 1,
    depth = 6,
    learning_rate = 0.2,
    grow_policy = policy,
    random_strength = 0,
    verbose = 0,
    allow_writing_files = FALSE,
    random_seed = 2205,
    thread_count = 1
  )
  if (identical(policy, "Lossguide")) {
    params$max_leaves <- 64
  }

  model <- catboost::catboost.train(wide_pool, params = params)
  explainer <- gazer(model)
  tree <- explainer$trees[[1L]]
  split_features <- unique(tree$feature[tree$children_left >= 0L])
  stopifnot(length(split_features) > 20L)

  native_prediction <- catboost::catboost.predict(model, wide_pool)
  parsed_prediction <- qshap:::catboost_predict_from_trees(
    X_wide, explainer$trees, explainer$base_score
  )
  stopifnot(max(abs(native_prediction - parsed_prediction)) < 1e-12)

  X_float32 <- qshap:::catboost_qshap_matrix(X_wide, list(tree))
  polynomial_t0 <- compute_polynomial_treeshap(tree, X_float32)
  native_t0 <- catboost::catboost.get_feature_importance(
    model, wide_pool, type = "ShapValues"
  )[, seq_len(p_wide), drop = FALSE]
  stopifnot(max(abs(polynomial_t0 - native_t0)) < 1e-10)

  loss <- qshap:::qshap_loss_catboost_general(explainer, X_wide, y_wide)
  stopifnot(identical(dim(loss), c(n_wide, p_wide)))
  stopifnot(all(is.finite(loss)))
}

message("CatBoost SymmetricTree, Depthwise, and Lossguide validation passed.")
