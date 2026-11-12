#' Extract Local Contributions to the Global R-squared Decomposition
#'
#' @param x A result returned by \code{rsq(..., local = TRUE)}.
#' @param ... Additional arguments (currently unused).
#' @return A numeric matrix with observations in rows and features in columns.
#'   Feature labels are taken from the result metadata when available. Column
#'   sums reproduce its global feature contributions. These entries are
#'   contributions to the global decomposition, not individual R-squared values.
#' @details The accessor reports an error when local contributions were not
#'   requested, or when the stored matrix or feature labels are inconsistent.
#'   The input result is not modified.
#' @examples
#' contributions <- matrix(c(0.1, 0.2, -0.05, 0.15), nrow = 2)
#' result <- qshap_result(colSums(contributions),
#'                        feature_names = c("a", "b"), n_samples = 2,
#'                        local_rsq = contributions)
#' local_rsq(result)
#' @export
local_rsq <- function(x, ...) {
  UseMethod("local_rsq")
}

#' @rdname local_rsq
#' @export
local_rsq.default <- function(x, ...) {
  stop("x must be a Q-SHAP result returned by rsq()", call. = FALSE)
}

#' @rdname local_rsq
#' @export
local_rsq.qshap_result <- function(x, ...) {
  local_rsq.qshap_rsq(x, ...)
}

#' @rdname local_rsq
#' @export
local_rsq.qshap_rsq <- function(x, ...) {
  value <- x[["local_rsq"]]
  if (is.null(value)) {
    stop("Local contributions are unavailable; call rsq(..., local = TRUE)", call. = FALSE)
  }
  if (!is.matrix(value) || !is.numeric(value) ||
      ncol(value) != length(x[["rsq"]])) {
    stop("local_rsq must be a numeric matrix with one column per feature", call. = FALSE)
  }
  labels <- x[["feature_names"]]
  if (is.null(labels)) labels <- names(x[["rsq"]])
  if (!is.null(labels)) {
    if (length(labels) != ncol(value)) {
      stop("feature labels must match the local contribution columns", call. = FALSE)
    }
    colnames(value) <- labels
  }
  value
}
