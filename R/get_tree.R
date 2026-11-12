#' Return a Tree Stored by Gazer
#'
#' @param explainer A Q-SHAP explainer created by \code{gazer()}.
#' @param tree One-based index of the tree to return.
#'
#' @return The stored \code{simple_tree} object, containing the low-level tree
#'   fields and retaining its tree class and print method.
#' @details The generic rejects unsupported explainer classes. The method for
#'   \code{qshap_tree_explainer} validates the stored tree list and requires a
#'   finite, scalar, one-based integer index within that list.
#' @export
get_tree <- function(explainer, tree = 1L) {
  UseMethod("get_tree")
}

#' @rdname get_tree
#' @export
get_tree.default <- function(explainer, tree = 1L) {
  stop("explainer must be a qshap_tree_explainer created by gazer()", call. = FALSE)
}

#' @rdname get_tree
#' @export
get_tree.qshap_tree_explainer <- function(explainer, tree = 1L) {
  trees <- explainer[["trees"]]
  if (!is.list(trees)) {
    stop("explainer must contain a list of stored trees", call. = FALSE)
  }
  if (!is.numeric(tree) || length(tree) != 1L || is.na(tree) ||
      !is.finite(tree) || tree != floor(tree) || tree < 1L || tree > length(trees)) {
    stop("tree must be a valid one-based tree index", call. = FALSE)
  }
  stored <- trees[[as.integer(tree)]]
  validate_simple_tree(stored)
  stored
}
