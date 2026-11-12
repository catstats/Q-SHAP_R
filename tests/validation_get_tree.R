library(qshap)

tree <- qshap:::simple_tree(
  c(1L, -1L, -1L), c(2L, -1L, -1L), c(0L, -1L, -1L),
  c(0.5, 0, 0), 1L, c(10L, 6L, 4L), c(0.1, -0.2, 0.55), 3L
)
explainer <- structure(
  list(trees = list(tree)),
  class = "qshap_tree_explainer"
)

stopifnot(identical(get_tree(explainer), tree), inherits(get_tree(explainer), "simple_tree"))

expect_error <- function(expr, pattern) {
  error <- tryCatch({ force(expr); NULL }, error = identity)
  stopifnot(inherits(error, "error"), grepl(pattern, conditionMessage(error)))
}
expect_error(get_tree(list(trees = list(tree))), "qshap_tree_explainer")
expect_error(get_tree(NULL), "qshap_tree_explainer")
for (index in list(0, -1, 2, Inf, NaN, NA_real_, 1.5, "1", c(1, 1), numeric())) {
  expect_error(get_tree(explainer, index), "one-based tree index")
}
expect_error(get_tree(structure(list(), class = "qshap_tree_explainer")), "list of stored trees")
