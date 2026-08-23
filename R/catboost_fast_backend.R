# Fast global Q-SHAP R_j^2 backend for CatBoost symmetric trees.
# local=TRUE intentionally remains on the matrix-valued loss backend.

#' @keywords internal
qshap_catboost_fast_global_stats <- function(explainer, x, y, compute_sd = TRUE) {
  x <- catboost_qshap_matrix(x, explainer$trees, float32 = FALSE)
  y <- as.numeric(y)
  if (length(y) != nrow(x)) {
    stop("y must have one value per row of x.", call. = FALSE)
  }

  if (catboost_uses_oblivious_trees(explainer)) {
    return(catboost_qshap_r2_fast(
      x, y, explainer$trees, explainer$base_score, compute_sd
    ))
  }

  # Non-symmetric CatBoost trees use the shared arbitrary-tree Q-SHAP path.
  # Return the same sufficient-statistics contract as the optimized C++ path
  # so qshap_rsq() does not need backend-specific branching.
  loss <- qshap_loss_catboost_general(explainer, x, y)
  loss_sum <- colSums(loss)
  sst <- sum((y - mean(y))^2)
  if (!is.finite(sst) || sst <= 0) {
    stop("Cannot compute R2 decomposition when y has zero variance.", call. = FALSE)
  }

  loss_sumsq <- if (isTRUE(compute_sd)) colSums(loss * loss) else NULL
  sd_rsq <- NULL
  if (isTRUE(compute_sd)) {
    n <- nrow(loss)
    if (n > 1L) {
      loss_var <- pmax(
        (loss_sumsq - (loss_sum * loss_sum) / n) / (n - 1L),
        0.0
      )
      sd_rsq <- sqrt(n * loss_var) / sst
    } else {
      sd_rsq <- rep(NA_real_, ncol(loss))
    }
  }

  list(
    rsq = -loss_sum / sst,
    loss_sum = loss_sum,
    n = nrow(loss),
    SST = sst,
    loss_sumsq = loss_sumsq,
    sd_rsq = sd_rsq
  )
}
