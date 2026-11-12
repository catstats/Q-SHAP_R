# Optional backend validation, run separately from CRAN package checks.
if (!requireNamespace("catboost", quietly = TRUE)) {
  stop("Install catboost before running this optional backend test.", call. = FALSE)
}
library(qshap)
set.seed(20261113)
X <- matrix(rnorm(64L * 4L), ncol = 4L,
            dimnames = list(NULL, paste0("x", 1:4)))
y <- 1.5 * X[, 1L] - X[, 2L] + 0.2 * X[, 3L] * X[, 4L]
model <- catboost::catboost.train(
  catboost::catboost.load_pool(X, label = y),
  params = list(iterations = 4L, depth = 2L, loss_function = "RMSE",
                thread_count = 1L, logging_level = "Silent", allow_writing_files = FALSE)
)
g <- gazer(model)
result <- rsq(g, X, y, local = TRUE)
baseline <- result$rsq
output <- capture.output({ gs <- summary(g); s <- summary(result) })
stopifnot(
  length(output) == 0L,
  inherits(gs, "summary.qshap_tree_explainer"),
  inherits(s, "summary.qshap_result"), gs$model_type == "catboost",
  gs$n_trees == length(g$trees), gs$precomputed_initialized,
  identical(s$features, as.data.frame(result)),
  identical(s$total_rsq, result$total_rsq),
  isTRUE(all.equal(sum(s$features$rsq), sum(baseline))),
  identical(s$loss_dim, dim(X)), identical(s$local_rsq_dim, dim(X)),
  identical(result$rsq, baseline)
)
invisible(capture.output(print(gs)))
invisible(capture.output(print(s, n = 2L)))
cat("Summary methods validated: catboost\n")
