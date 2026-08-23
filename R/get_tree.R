#' Return a Tree Stored by Gazer
#'
#' @param explainer A Q-SHAP explainer created by \code{gazer()}.
#' @param tree One-based index of the tree to return.
#'
#' @return An unclassed named list containing the stored low-level tree fields.
#' @export
get_tree <- function(explainer, tree = 1L) {
  trees <- explainer[["trees"]]
  if (!is.numeric(tree) || length(tree) != 1L || is.na(tree) ||
      tree != as.integer(tree) || tree < 1L || tree > length(trees)) {
    stop("tree must be a valid one-based tree index", call. = FALSE)
  }
  unclass(trees[[as.integer(tree)]])
}
