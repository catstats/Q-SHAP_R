library(qshap)

if (requireNamespace("lightgbm", quietly = TRUE)) {
  n <- 240L
  raw_x1 <- seq(-2, 2, length.out = n)
  missing_idx <- seq_len(60L)
  y <- ifelse(raw_x1 <= 0, -4, 4)

  X <- cbind(
    x1 = raw_x1,
    x2 = sin(seq_len(n))
  )
  X[missing_idx, 1L] <- NA_real_

  data <- lightgbm::lgb.Dataset(data = X, label = y)
  model <- lightgbm::lgb.train(
    params = list(
      objective = "regression",
      max_depth = 2L,
      num_leaves = 4L,
      learning_rate = 0.3,
      min_data_in_leaf = 1L,
      min_data_in_bin = 1L,
      seed = 42L,
      deterministic = TRUE,
      force_col_wise = TRUE,
      num_threads = 1L,
      verbosity = -1L
    ),
    data = data,
    nrounds = 5L
  )

  tree_table <- lightgbm::lgb.model.dt.tree(model)
  learned_default_left <- tree_table[["default_left"]] == "TRUE"
  stopifnot(any(learned_default_left, na.rm = TRUE))

  explainer <- gazer(model)
  stored_default_left <- unlist(
    lapply(explainer[["trees"]], function(tree) tree[["default_left"]]),
    use.names = FALSE
  )
  stopifnot(any(stored_default_left))

  result <- rsq(explainer, X, y, sd_out = FALSE)
  pred <- stats::predict(model, X)
  fitted_rsq <- 1 - sum((y - pred)^2) / sum((y - mean(y))^2)

  stopifnot(abs(sum(result[["rsq"]]) - fitted_rsq) < 1e-8)
}
