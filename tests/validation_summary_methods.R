library(qshap)

expect_error <- function(expr, pattern) {
  error <- tryCatch({ force(expr); NULL }, error = identity)
  stopifnot(inherits(error, "error"), grepl(pattern, conditionMessage(error)))
}

# Summary construction is silent, visible, and separate from the input.
values <- c(0.123456789012345, -0.04, 0.3, 0.02, NA_real_)
result <- qshap_result(
  values, feature_names = c("first", "negative", "largest", "small", "missing"),
  n_samples = 2L, loss = matrix(0, 2L, 5L), local_rsq = matrix(0, 2L, 5L)
)
before <- serialize(result, NULL)
output <- capture.output(returned <- withVisible(summary(result)))
s <- returned$value
expected_distribution <- c(
  min = -0.04, q1 = unname(quantile(values, 0.25, na.rm = TRUE)),
  median = median(values, na.rm = TRUE), mean = mean(values, na.rm = TRUE),
  q3 = unname(quantile(values, 0.75, na.rm = TRUE)), max = 0.3
)
stopifnot(
  length(output) == 0L, returned$visible,
  identical(class(s), "summary.qshap_result"), !identical(s, result),
  identical(serialize(result, NULL), before),
  identical(s$features, as.data.frame(result)),
  identical(s$total_rsq, result$total_rsq),
  identical(s$rsq_distribution, expected_distribution),
  identical(s$n_missing, 1L), identical(s$n_above_threshold, 3L),
  identical(s$loss_dim, c(2L, 5L)), identical(s$local_rsq_dim, c(2L, 5L)),
  s$features$rsq[s$features$feature == "first"] == values[1L],
  !any(c("loss", "local_rsq") %in% names(s))
)

# Printing controls display only and returns the summary invisibly.
before_summary <- serialize(s, NULL)
output <- capture.output(printed <- withVisible(print(s, n = 1L, digits = 3L)))
stopifnot(
  any(grepl("Top 1 features", output, fixed = TRUE)),
  any(grepl("largest", output, fixed = TRUE)),
  !any(grepl("negative", output, fixed = TRUE)),
  identical(printed$value, s), !printed$visible,
  identical(serialize(s, NULL), before_summary), nrow(s$features) == 5L
)
expect_error(print(s, n = -1), "non-negative integer")
expect_error(print(s, n = c(1, 2)), "non-negative integer")
expect_error(print(s, n = "2"), "non-negative integer")
expect_error(print(s, digits = 0), "between 1 and 22")
expect_error(print(s, digits = NA_real_), "between 1 and 22")
invisible(capture.output(print(s, n = 0L)))

# Missing/empty contributions and absent local matrices remain usable.
all_missing <- summary(qshap_result(c(NA_real_, NA_real_)))
empty <- summary(qshap_result(numeric(), feature_names = character()))
stopifnot(
  all(is.na(all_missing$rsq_distribution)),
  all_missing$n_missing == 2L, all_missing$n_above_threshold == 0L,
  all(is.na(empty$rsq_distribution)), nrow(empty$features) == 0L,
  is.null(empty$loss_dim), is.null(empty$local_rsq_dim), is.null(empty$n_samples)
)
invisible(capture.output(print(empty)))

# Tree statistics cover every tree, including the sixth, and omit heavy caches.
counts <- c(3L, 7L, 7L, 3L, 7L, 31L)
explainer <- structure(list(
  model_type = "xgboost", max_depth = 4L, base_score = 0.123456789012345,
  trees = lapply(counts, function(n) list(node_count = n)),
  store_v_invc = matrix(0i, 9L, 5L), store_z = matrix(0i, 9L, 5L)
), class = c("qshap_tree_explainer", "xgboost_explainer"))
before <- serialize(explainer, NULL)
output <- capture.output(returned <- withVisible(summary(explainer)))
es <- returned$value
stopifnot(
  length(output) == 0L, returned$visible,
  identical(class(es), "summary.qshap_tree_explainer"),
  identical(serialize(explainer, NULL), before),
  identical(es$node_counts, as.numeric(counts)),
  isTRUE(all.equal(es$node_count_summary, c(min = 3, mean = mean(counts), max = 31))),
  es$n_trees == 6L, identical(es$base_score, explainer$base_score),
  es$precomputed_initialized,
  identical(es$precomputed_dimensions$v_invc, c(9L, 5L)),
  !any(c("model", "trees", "store_v_invc", "store_z") %in% names(es))
)
before_summary <- serialize(es, NULL)
output <- capture.output(printed <- withVisible(print(es, digits = 3L)))
stopifnot(
  identical(printed$value, es), !printed$visible,
  identical(serialize(es, NULL), before_summary),
  any(grepl("all trees", output, fixed = TRUE))
)
expect_error(print(es, digits = 1.5), "between 1 and 22")
explainer$trees <- list()
explainer$base_score <- explainer$store_v_invc <- explainer$store_z <- NULL
es_empty <- summary(explainer)
stopifnot(
  es_empty$n_trees == 0L, all(is.na(es_empty$node_count_summary)),
  !es_empty$precomputed_initialized, is.null(es_empty$base_score)
)
invisible(capture.output(print(es_empty)))

# Exercise actual gazer()/rsq() outputs and S3 dispatch for every available backend.
set.seed(20261113)
X <- matrix(rnorm(64L * 4L), ncol = 4L,
            dimnames = list(NULL, paste0("x", 1:4)))
y <- 1.5 * X[, 1L] - X[, 2L] + 0.2 * X[, 3L] * X[, 4L]
models <- list(
  xgboost = xgboost::xgboost(
    x = X, y = y, nrounds = 4L, max_depth = 2L, learning_rate = 0.2,
    objective = "reg:squarederror", nthread = 1L, verbosity = 0L
  ),
  lightgbm = lightgbm::lgb.train(
    list(objective = "regression", max_depth = 2L, num_leaves = 4L,
         min_data_in_leaf = 3L, num_threads = 1L, verbosity = -1L),
    lightgbm::lgb.Dataset(X, label = y), nrounds = 4L
  )
)
for (name in names(models)) {
  g <- gazer(models[[name]])
  result <- rsq(g, X, y, local = TRUE)
  baseline <- result$rsq
  gs <- summary(g)
  s <- summary(result)
  stopifnot(
    inherits(gs, "summary.qshap_tree_explainer"),
    inherits(s, "summary.qshap_result"), gs$model_type == name,
    gs$n_trees == length(g$trees), gs$precomputed_initialized,
    identical(s$features, as.data.frame(result)),
    identical(s$total_rsq, result$total_rsq),
    isTRUE(all.equal(sum(s$features$rsq), sum(baseline))),
    isTRUE(all.equal(sum(s$features$rsq), result$total_rsq)),
    identical(s$loss_dim, dim(X)), identical(s$local_rsq_dim, dim(X)),
    identical(result$rsq, baseline)
  )
  cat("Summary methods validated:", name, "\n")
}
