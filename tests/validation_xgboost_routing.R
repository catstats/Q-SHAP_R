library(qshap)

if (requireNamespace("xgboost", quietly = TRUE)) {
  X <- matrix(rep(c(0, 1, 2, NA_real_), 50L), ncol = 1L)
  y <- rep(c(0, 10, 20, -10), 50L)
  model <- xgboost::xgboost(
    x = X,
    y = y,
    nrounds = 5L,
    max_depth = 1L,
    learning_rate = 0.3,
    nthread = 1L,
    verbosity = 0L
  )

  explainer <- gazer(model)
  result <- rsq(explainer, X, y, local = TRUE, sd_out = FALSE)
  global_result <- rsq(explainer, X, y, local = FALSE, sd_out = FALSE)
  prediction <- stats::predict(model, X)
  q_emptyset <- sum((y - mean(y))^2)
  fitted_rsq <- 1 - sum((y - prediction)^2) / q_emptyset
  qshap_rsq <- sum(result[["rsq"]])
  has_default_left <- any(vapply(
    explainer[["trees"]],
    function(tree) any(tree[["default_left"]]),
    logical(1L)
  ))

  stopifnot(has_default_left)
  stopifnot(abs(qshap_rsq - fitted_rsq) < 1e-6)
  stopifnot(is.null(global_result[["loss"]]))
  stopifnot(is.null(global_result[["local_rsq"]]))
  stopifnot(isTRUE(all.equal(
    unname(global_result[["rsq"]]),
    unname(result[["rsq"]]),
    tolerance = 1e-12
  )))
  stopifnot(isTRUE(all.equal(
    result[["loss"]],
    loss(explainer, X, y),
    tolerance = 1e-12,
    check.attributes = FALSE
  )))
  stopifnot(isTRUE(all.equal(
    result[["local_rsq"]],
    -result[["loss"]] / q_emptyset,
    tolerance = 1e-12,
    check.attributes = FALSE
  )))
  stopifnot(isTRUE(all.equal(
    colSums(result[["local_rsq"]]),
    unname(result[["rsq"]]),
    tolerance = 1e-10,
    check.attributes = FALSE
  )))

  # Binary logistic trees store summed Hessians as fractional node covers.
  # These values must remain numeric when converted to simple_tree objects.
  X_binary <- matrix(seq_len(37L), ncol = 1L)
  y_binary <- factor(as.integer(X_binary[, 1L] > 18L))
  model_binary <- xgboost::xgboost(
    x = X_binary,
    y = y_binary,
    objective = "binary:logistic",
    nrounds = 3L,
    max_depth = 1L,
    learning_rate = 0.3,
    min_child_weight = 0,
    nthread = 1L,
    verbosity = 0L
  )

  binary_explainer <- gazer(model_binary)
  node_covers <- unlist(lapply(
    binary_explainer[["trees"]],
    function(tree) tree[["n_node_samples"]]
  ), use.names = FALSE)

  stopifnot(is.double(node_covers))
  stopifnot(any(abs(node_covers - round(node_covers)) > 1e-6))
}
