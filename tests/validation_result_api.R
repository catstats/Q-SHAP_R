library(qshap)

values <- matrix(c(0.1, 0.2, -0.05, 0.15), nrow = 2L)
result <- qshap_result(colSums(values), feature_names = c("a", "b"),
                       n_samples = 2L, local_rsq = values)
before <- serialize(result, NULL)
stopifnot(
  identical(class(result), c("qshap_result", "qshap_rsq")),
  identical(local_rsq(result), `colnames<-`(values, c("a", "b"))),
  identical(serialize(result, NULL), before),
  isTRUE(all.equal(unname(colSums(local_rsq(result))), unname(result$rsq))),
  inherits(summary(result), "summary.qshap_result")
)
device <- tempfile(fileext = ".pdf")
grDevices::pdf(device)
stopifnot(inherits(plot(result), "ggplot"))
grDevices::dev.off()
unlink(device)

expect_error <- function(expr, pattern) {
  error <- tryCatch({ force(expr); NULL }, error = identity)
  stopifnot(inherits(error, "error"), grepl(pattern, conditionMessage(error)))
}
expect_error(local_rsq(list()), "Q-SHAP result")
expect_error(local_rsq(qshap_result(c(0.1, 0.2))), "local = TRUE")
broken <- result
broken$local_rsq <- matrix(0, 2, 3)
expect_error(local_rsq(broken), "one column per feature")
broken <- result
broken$feature_names <- "a"
expect_error(local_rsq(broken), "feature labels")

set.seed(20261113)
X <- matrix(rnorm(48L * 3L), ncol = 3L,
            dimnames = list(NULL, c("a", "b", "c")))
y <- X[, 1] - X[, 2]
model <- xgboost::xgboost(x = X, y = y, nrounds = 3L, max_depth = 2L,
                          nthread = 1L, verbosity = 0L)
g <- gazer(model)
stopifnot(identical(class(g), "qshap_tree_explainer"))
fitted <- rsq(g, X, y, local = TRUE)
stopifnot(
  identical(class(fitted), class(result)),
  identical(colnames(local_rsq(fitted)), colnames(X)),
  isTRUE(all.equal(unname(colSums(local_rsq(fitted))), unname(fitted$rsq))),
  inherits(get_tree(g), "simple_tree")
)
