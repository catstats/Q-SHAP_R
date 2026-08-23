library(qshap)

tree <- qshap:::simple_tree(
  c(1L, -1L, -1L), c(2L, -1L, -1L), c(0L, -1L, -1L),
  c(0.5, 0, 0), 1L, c(10L, 6L, 4L), c(0.1, -0.2, 0.55), 3L
)
explainer <- structure(
  list(trees = list(tree)),
  class = "qshap_tree_explainer"
)

stopifnot(identical(get_tree(explainer), unclass(tree)))
