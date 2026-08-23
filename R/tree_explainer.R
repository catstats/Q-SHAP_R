#' @import Rcpp
#' @importFrom methods new
#' @importFrom parallel makeCluster stopCluster detectCores
#' @importFrom parallel clusterEvalQ clusterExport parLapply
#' @useDynLib qshap, .registration = TRUE
NULL

#' Create a QSHAP Tree Explainer
#' 
#' Creates an explainer object for computing feature-specific Shapley values
#' from a trained tree ensemble model. Supports XGBoost, LightGBM, and
#' CatBoost models.
#' 
#' @param model A model object of class \code{xgboost} or \code{xgb.Booster} from \pkg{xgboost}, class \code{lgb.Booster} from \pkg{lightgbm}, or class \code{catboost.Model} from \pkg{catboost}
#' @param max_depth Maximum depth of trees, extracted from \code{model} by default.
#' @param base_score Base score for predictions, extracted from \code{model} by default.
#' @param ... Additional arguments, for future use
#' 
#' @return A class of \code{qshap_tree_explainer} object containing the model information and
#'   preprocessed tree structures for fast Shapley value computation
#'   
#' @examples
#' library(xgboost)
#' set.seed(42)
#' n <- 100
#' p <- 100
#' X <- matrix(rnorm(n * p), nrow = n, ncol = p)
#' y <- X[, 1] - X[, 2] + rnorm(n, sd = 0.2)
#' model <- xgboost(X, y, nrounds = 15, max_depth = 2, verbosity = 0, nthread = 1)
#' explainer <- gazer(model)
#'
#' @export
gazer <- function(model, max_depth = NULL, base_score = NULL, ...) {
  UseMethod("gazer")
}

cache_tree_summaries <- function(explainer) {
  explainer$tree_summaries <- lapply(explainer$trees, summarize_tree)
  explainer
}

#' @export
gazer.xgb.Booster <- function(model, ...) {


  tmp <- tempfile(fileext = ".json")
  xgboost::xgb.save(model, tmp)
  model_json <- jsonlite::fromJSON(tmp, simplifyVector = FALSE)
  unlink(tmp)

  # For version 3.1.3.1 and later, we can get max_depth from attributes
  max_depth <- if (!is.null(attributes(model)$params$max_depth)) attributes(model)$params$max_depth else 6

  # Extract base_score - handle various JSON formats
  base_score_raw <- model_json$learner$learner_model_param$base_score
  if (is.character(base_score_raw)) {
    # Handle string format like "[-7.70459E-2]" or just a number string
    base_score_raw <- gsub("\\[|\\]", "", base_score_raw)  # Remove brackets
    base_score <- as.numeric(base_score_raw)
  } else if (is.list(base_score_raw)) {
    base_score <- as.numeric(base_score_raw[[1]])
  } else {
    base_score <- as.numeric(base_score_raw)
  }
  # Fallback to mean of predictions if still NA
  if (is.na(base_score)) {
    warning("Could not extract base_score from model, using 0.5 as default")
    base_score <- 0.5
  }

  # eta: try params first, else JSON
  # eta <- model$params$eta
  # if (is.null(eta)) {
  #   # common JSON locations depending on xgboost build
  #   eta <- model_json$learner$gradient_booster$gbtree_train_param$learning_rate
  #   if (is.null(eta)) eta <- model_json$learner$gradient_booster$gbtree_train_param$eta
  # }
  # if (is.null(eta)) eta <- 0.3
  # eta <- as.numeric(eta)

  xgb_trees <- xgb_formatter(model_json, max_depth)

  explainer <- new_qshap_tree_explainer(
    model = model,
    model_type = "xgboost",
    max_depth = max_depth,
    base_score = base_score,
    trees = xgb_trees,
    store_v_invc = store_complex_v_invc(max_depth * 2),
    store_z = store_complex_root(max_depth * 2)
  )

  explainer <- cache_tree_summaries(explainer)
  
  validate_qshap_tree_explainer(explainer)
  explainer
}

#' @export
gazer.lgb.Booster <- function(model, max_depth = NULL, ...) {
  # Get max_depth from model parameters or use default
  max_depth <- if (!is.null(model$params$max_depth)) model$params$max_depth else 31
  
  # Format LightGBM trees
  lgb_trees <- lgb_formatter(model, max_depth)

  explainer <- new_qshap_tree_explainer(
    model = model,
    model_type = "lightgbm",
    max_depth = max_depth,
    base_score = NULL,
    trees = lgb_trees,
    store_v_invc = store_complex_v_invc(max_depth * 2),
    store_z = store_complex_root(max_depth * 2)
  )

  explainer <- cache_tree_summaries(explainer)
  
  validate_qshap_tree_explainer(explainer)
  explainer
}

#' @export
gazer.catboost.Model <- function(model, max_depth = NULL, ...) {
  # Format CatBoost trees (returns list with metadata attributes)
  cb_trees <- catboost_formatter(model, max_depth)

  # Extract metadata set by formatter
  bias <- attr(cb_trees, "bias")
  if (is.null(bias)) bias <- 0.0
  actual_max_depth <- attr(cb_trees, "max_depth")
  if (is.null(actual_max_depth)) actual_max_depth <- 6L

  if (is.null(max_depth)) {
    max_depth <- actual_max_depth
  } else {
    if (length(max_depth) != 1L || is.na(max_depth) || !is.finite(max_depth) ||
        max_depth < 1 || max_depth != floor(max_depth)) {
      stop("max_depth must be a positive integer.", call. = FALSE)
    }
    # The parsed model gives the exact required depth.  Smaller hints are not
    # safe, while larger hints only inflate the O(depth^2) root caches and can
    # overflow integer dimensions without changing the result.
    max_depth <- actual_max_depth
  }

  explainer <- new_qshap_tree_explainer(
    model = model,
    model_type = "catboost",
    max_depth = max_depth,
    base_score = bias,
    trees = cb_trees,
    store_v_invc = store_complex_v_invc(max_depth * 2),
    store_z = store_complex_root(max_depth * 2)
  )

  explainer <- cache_tree_summaries(explainer)

  validate_qshap_tree_explainer(explainer)
  explainer
}

# #' @export
# gazer.gbm <- function(model, max_depth = NULL, ...) {
# }

#' @export
gazer.default <- function(model, ...) {
  stop(sprintf("gazer not implemented for class %s", class(model)[1]))
}

#' Calculate Q-SHAP Loss Contributions
#' 
#' Computes the feature-specific loss contributions using Q-SHAP decomposition.
#' This is an internal function typically called by \code{rsq()}.
#' 
#' @param explainer A qshap_tree_explainer object created by \code{gazer()}
#' @param x Feature matrix or data frame
#' @param y Response vector
#' @param y_mean_ori Optional pre-computed mean of y (for efficiency)
#' 
#' @return A matrix of loss contributions with dimensions (n_samples, n_features)
#' 
#' @keywords internal
qshap_loss <- function(explainer, x, y, y_mean_ori = NULL) {
  UseMethod("qshap_loss")
}

#' @export
qshap_loss.qshap_tree_explainer <- function(explainer, x, y, y_mean_ori = NULL) {
  switch(explainer$model_type,
    "xgboost" = qshap_loss_xgboost(explainer, x, y, y_mean_ori),
    "lightgbm" = qshap_loss_lightgbm(explainer, x, y, y_mean_ori),
    "catboost" = qshap_loss_catboost(explainer, x, y, y_mean_ori),
    # "gbm" = qshap_loss_gbm(explainer, x, y, y_mean_ori, progress_bar),
    stop("Unknown model type: ", explainer$model_type)
  )
}

 #' Calculate Feature-Specific R-Squared Values
 #' 
 #' Computes feature-specific R-squared values using Q-SHAP decomposition.
 #' Supports parallel processing and sampling for large datasets.
 #' 
 #' @param explainer A qshap_tree_explainer object created by \code{gazer()}
 #' @param x Feature matrix or data frame with n samples and p features
 #' @param y Response vector of length n
 #' @param local Logical; if TRUE, also returns the raw observation-level
 #'   squared-loss contributions in \code{loss} and their normalized
 #'   contributions to the global R-squared decomposition in
 #'   \code{local_rsq}.
 #' @param sd_out Logical; if TRUE, returns standard deviations of R-squared estimates
 #' @param nsample Optional integer; number of samples to use (random subsample if less than nrow(x))
 #' @param nfrac Optional numeric in (0,1); fraction of samples to use (alternative to nsample)
 #' @param random_state Integer seed for reproducible sampling
 #' @param ncore Number of cores for parallel processing. Use -1 for all available cores, 
 #'   or a positive integer. Default is 1 (no parallelization)
 #' 
 #' @return A \code{qshap_rsq} object containing the feature-specific
 #'   R-squared vector in \code{rsq} and, when requested, \code{sd_rsq}. If
 #'   \code{local=TRUE}, the object also contains \code{loss} and
 #'   \code{local_rsq}.
 #'   The \code{loss} matrix is the unchanged raw observation-level Shapley
 #'   decomposition of the change in squared loss. If
 #'   \eqn{Q_\emptyset = \sum_i (y_i - \bar y)^2}, then
 #'   \code{local_rsq = -loss / Q_emptyset}. Thus, \code{local_rsq} contains
 #'   observation-level contributions to the global R-squared decomposition;
 #'   it is not an observation-specific coefficient of determination.
 #'   
 #' @examples
 #' library(xgboost)
 #' set.seed(42)
 #' n <- 100
 #' p <- 100
 #' X <- matrix(rnorm(n * p), nrow = n, ncol = p)
 #' y <- X[, 1] - X[, 2] + rnorm(n, sd = 0.2)
 #' model <- xgboost(X, y, nrounds = 15, max_depth = 2, verbosity = 0, nthread = 1)
 #' explainer <- gazer(model)
 #' phi_rsq <- qshap(explainer, X, y)
 #' print(phi_rsq)
 #'
 #' @keywords internal
qshap_rsq <- function(explainer, x, y, local = FALSE, nsample = NULL, sd_out = TRUE,
                      nfrac = NULL, random_state = 42,
                      ncore = 1L) {
  # Sampling logic
  if (!is.null(nsample)) {
    if (nsample <= 0 || nsample >= nrow(x)) {
      stop("nsample must be > 0 and < total number of samples")
    }
    set.seed(random_state)
    sample_idx <- sample(nrow(x), nsample, replace = FALSE)
    x <- x[sample_idx, , drop = FALSE]
    y <- y[sample_idx]
  } else if (!is.null(nfrac)) {
    if (nfrac <= 0 || nfrac >= 1) {
      stop("nfrac must be between 0 and 1")
    }
    set.seed(random_state)
    nsample <- floor(nrow(x) * nfrac)
    sample_idx <- sample(nrow(x), nsample, replace = FALSE)
    x <- x[sample_idx, , drop = FALSE]
    y <- y[sample_idx]
  }

  y_mean_ori <- mean(y)
  sst <- sum((y - y_mean_ori)^2)

  # Parallel if number of samples is large enough
  # Normalize ncore
  if (is.null(ncore) || length(ncore) != 1 || is.na(ncore)) ncore <- 1L
  ncore <- as.integer(ncore)
  max_core <- parallel::detectCores(logical = TRUE)
  if (is.na(max_core) || max_core < 1) max_core <- 1L
  if (ncore == -1L) ncore <- max_core
  ncore <- max(1L, min(max_core, ncore))

  n <- nrow(x)

  if (identical(explainer$model_type, "catboost") && !isTRUE(local)) {
    # Fast CatBoost global backend uses the Q-SHAP paper decomposition:
    # loss[i, j] = T2[i, j] - 2 * r_i^(k-1) * T0[i, j],
    # R_j^2 = -sum_i,k loss[i, j] / SST.
    fast_stats <- qshap_catboost_fast_global_stats(explainer, x, y, compute_sd = sd_out)
    rsq <- fast_stats$rsq
    sd_rsq <- if (sd_out) fast_stats$sd_rsq else NULL

    out <- list(rsq = rsq, sd_rsq = sd_rsq)
    class(out) <- c("qshap_rsq", "list")
    return(out)
  }

  # Calculate loss (serial)
  if (ncore == 1L || n <= 1L) {
    loss <- qshap_loss(explainer, x, y, y_mean_ori)
    rsq <- -colSums(loss) / sst

     if (sd_out) {
      loss_mean <- colMeans(loss)
      if (n > 1L) {
        loss_sumsq <- colSums(loss * loss)
        loss_var <- pmax((loss_sumsq - n * loss_mean^2) / (n - 1), 0)
        sd_rsq <- sqrt(n * loss_var) / sst
      } else {
        sd_rsq <- rep(NA_real_, ncol(loss))
      }
    } else {
      sd_rsq <- NULL
    }
    if (local) {
      local_rsq <- -loss / sst
      out <- list(
        rsq = rsq,
        loss = loss,
        local_rsq = local_rsq,
        sd_rsq = sd_rsq
      )
    } else {
      out <- list(rsq = rsq, sd_rsq = sd_rsq)
    }
    class(out) <- c("qshap_rsq", "list")
    return(out)
  }

  # Divide indices into ncore chunks (preserve order)
  idx_chunks <- split(seq_len(n), cut(seq_len(n), breaks = ncore, labels = FALSE))

  # Parallel compute using PSOCK cluster (CRAN/Windows friendly)
  cl <- parallel::makeCluster(ncore)
  on.exit(parallel::stopCluster(cl), add = TRUE)

  # Ensure workers load qshap from the same library paths as the main session.
  # This matters after local/GitHub installs: otherwise PSOCK workers can pick
  # up an older qshap without newer backends such as CatBoost.
  qshap_lib_paths <- .libPaths()
  parallel::clusterExport(cl, varlist = "qshap_lib_paths", envir = environment())
  parallel::clusterEvalQ(cl, {
    .libPaths(qshap_lib_paths)
    suppressPackageStartupMessages(library(qshap))
    NULL
  })

  if (identical(explainer$model_type, "catboost")) {
    worker_has_catboost <- parallel::clusterEvalQ(
      cl,
      exists("qshap_loss_catboost", envir = asNamespace("qshap"), inherits = FALSE)
    )
    if (!all(unlist(worker_has_catboost))) {
      worker_versions <- parallel::clusterEvalQ(cl, as.character(utils::packageVersion("qshap")))
      stop(
        "Parallel CatBoost Q-SHAP requires workers to load a CatBoost-enabled qshap. ",
        "The workers loaded qshap version(s): ",
        paste(unique(unlist(worker_versions)), collapse = ", "),
        ". Reinstall qshap, restart R, and retry. For development sessions, use ",
        "devtools::install() rather than only source()/load_all() before ncore > 1.",
        call. = FALSE
      )
    }
  }

  # Export needed data once (avoid resending for every task)
  parallel::clusterExport(
    cl,
    varlist = c("x", "y", "explainer", "y_mean_ori", "idx_chunks", "local"),
    envir = environment()
  )

  worker <- function(idx) {
  lc <- qshap::loss(explainer, x[idx, , drop = FALSE], y[idx], y_mean_ori)

  if (local) {
    # keep full chunk loss matrix
    return(lc)
  } else {
    # return sufficient statistics only
    return(list(
      sum   = colSums(lc),
      sumsq = if (sd_out) colSums(lc^2) else NULL,
      n     = length(idx)
    ))
  }
  }

  results <- parallel::parLapply(cl, idx_chunks, worker)

if (local) {
  # Combine full loss matrix
  loss <- do.call(rbind, results)
  n_all <- nrow(loss)

  # point estimate
  rsq <- -colSums(loss) / sst

  if (sd_out) {
      loss_mean <- colMeans(loss)
      if (n_all > 1L) {
        loss_sumsq <- colSums(loss * loss)
        loss_var <- pmax((loss_sumsq - n_all * loss_mean^2) / (n_all - 1), 0)
        sd_rsq <- sqrt(n_all * loss_var) / sst
      } else {
        sd_rsq <- rep(NA_real_, ncol(loss))
      }
    } else {
      sd_rsq <- NULL
    }
  local_rsq <- -loss / sst
  out <- list(
    rsq = rsq,
    loss = loss,
    local_rsq = local_rsq,
    sd_rsq = sd_rsq
  )
  class(out) <- c("qshap_rsq", "list")
  return(out)

} else {
  # Combine sufficient statistics
  sum_all <- Reduce(`+`, lapply(results, `[[`, "sum"))
  n_all <- sum(vapply(results, `[[`, numeric(1), "n"))

  # point estimate
  rsq <- -sum_all / sst

  if (sd_out) {
      if (n_all > 1L) {
        sumsq_all <- Reduce(`+`, lapply(results, `[[`, "sumsq"))
        loss_var <- pmax((sumsq_all - (sum_all^2) / n_all) / (n_all - 1), 0)
        sd_rsq <- sqrt(n_all * loss_var) / sst
      } else {
        sd_rsq <- rep(NA_real_, length(sum_all))
      }
    } else {
      sd_rsq <- NULL
    }
  out <- list(rsq = rsq, sd_rsq = sd_rsq)
  class(out) <- c("qshap_rsq", "list")
  return(out)
}

}

#' Calculate Feature-Specific R-Squared Values
#'
#' Computes feature-specific R-squared values using Q-SHAP decomposition,
#' returning a \code{qshap_result} object with better formatting and additional metadata.
#' The \code{qshap_result} object includes feature names, total R², sample counts,
#' and provides enhanced \code{print()}, \code{summary()}, and \code{as.data.frame()}
#' methods for easier analysis.
#'
#' @inheritParams qshap_rsq
#' @param feature_names Character vector of feature names. If NULL, uses column names from x.
#' @return A \code{qshap_result} object containing:
#'   \itemize{
#'     \item \code{rsq}: Numeric vector of feature-specific R² values
#'     \item \code{feature_names}: Character vector of feature names
#'     \item \code{total_rsq}: Total R² (sum of feature-specific values)
#'     \item \code{n_samples}: Number of samples
#'     \item \code{n_features}: Number of features
#'     \item \code{loss}: Unchanged raw observation-level contributions to the
#'       change in squared loss (if \code{local=TRUE})
#'     \item \code{local_rsq}: Observation-level contributions to the global
#'       R-squared decomposition, equal to \code{-loss / Q_emptyset}, where
#'       \eqn{Q_\emptyset = \sum_i (y_i - \bar y)^2}. Its column sums equal
#'       \code{rsq} up to numerical tolerance (if \code{local=TRUE})
#'   }
#'
#' @details
#' The \code{local_rsq} matrix contains local contributions on the R-squared
#' scale. It decomposes each global feature-specific \code{rsq} value across
#' observations and must not be interpreted as an observation-specific
#' coefficient of determination.
#'
#' This function provides a user-friendly interface for Q-SHAP R² computation:
#' \itemize{
#'   \item Automatically extracts feature names from the input data
#'   \item Returns a structured object with metadata
#'   \item Provides enhanced printing with top features displayed by default
#'   \item Includes a comprehensive \code{summary()} method
#'   \item Can be easily converted to a data frame with \code{as.data.frame()}
#' }
#'
#' @examples
#' library(xgboost)
#' set.seed(42)
#' n <- 100
#' p <- 100
#' X <- matrix(rnorm(n * p), nrow = n, ncol = p)
#' y <- X[, 1] - X[, 2] + rnorm(n, sd = 0.2)
#' model <- xgboost(X, y, nrounds = 15, max_depth = 2, verbosity = 0, nthread = 1)
#' explainer <- gazer(model)
#' result <- rsq(explainer, X, y)
#' print(result)
#'
#' @seealso \code{\link{qshap_result}}
#' @export
rsq <- function(explainer, x, y, feature_names = NULL, local = FALSE, nsample = NULL,
                sd_out = TRUE, nfrac = NULL,
                random_state = 42, ncore = 1L) {
  
  # Call qshap_rsq
  result <- qshap_rsq(
    explainer = explainer,
    x = x,
    y = y,
    local = local,
    nsample = nsample,
    sd_out = sd_out,
    nfrac = nfrac,
    random_state = random_state,
    ncore = ncore
  )

  # Extract feature names
  if (is.null(feature_names)) {
    feature_names <- colnames(x)
    if (is.null(feature_names)) {
      feature_names <- paste0("Feature_", seq_len(ncol(x)))
    }
  }

  # Add names to rsq vector
  names(result$rsq) <- feature_names

  # Make rsq() return the SAME core object as qshap_rsq(),
  # just with extra metadata and a more general class.
  result$feature_names <- feature_names
  result$total_rsq <- sum(result$rsq, na.rm = TRUE)
  result$n_samples <- nrow(x)
  result$n_features <- length(result$rsq)

  # Ensure local matrices are present only when local=TRUE.
  if (!isTRUE(local) && !is.null(result$loss)) {
    result$loss <- NULL
  }
  if (!isTRUE(local) && !is.null(result$local_rsq)) {
    result$local_rsq <- NULL
  }

  class(result) <- c("qshap_result", "qshap_rsq", "list")
  return(result)
}

#' Alias for rsq
#'
#' This is a convenience alias for \code{rsq()} that provides a shorter
#' function name for calculating feature-specific R-squared values.
#'
#' @inheritParams rsq
#' @return A \code{qshap_result} object; see \code{\link{rsq}} for details.
#'
#' @examples
#' library(xgboost)
#' set.seed(42)
#' n <- 100
#' p <- 100
#' X <- matrix(rnorm(n * p), nrow = n, ncol = p)
#' y <- X[, 1] - X[, 2] + rnorm(n, sd = 0.2)
#' model <- xgboost(X, y, nrounds = 15, max_depth = 2, verbosity = 0, nthread = 1)
#' explainer <- gazer(model)
#' phi_rsq <- qshap(explainer, X, y)
#' print(phi_rsq)
#'
#' @seealso \code{\link{rsq}}
#' @export
qshap <- function(explainer, x, y, feature_names = NULL, local = FALSE,
                  nsample = NULL, sd_out = TRUE,
                  nfrac = NULL, random_state = 42, ncore = 1L) {
  rsq(explainer, x, y, feature_names = feature_names, local = local,
      nsample = nsample, sd_out = sd_out,
      nfrac = nfrac, random_state = random_state, ncore = ncore)
}

#' Alias for qshap_loss
#'
#' This is a convenience alias for \code{qshap_loss()} that provides a shorter
#' function name for calculating feature-specific loss contributions.
#'
#' @inheritParams qshap_loss
#' @return A matrix of loss contributions with dimensions (n_samples, n_features)
#'
#' @examples
#' library(xgboost)
#' set.seed(42)
#' n <- 100
#' p <- 100
#' X <- matrix(rnorm(n * p), nrow = n, ncol = p)
#' y <- X[, 1] - X[, 2] + rnorm(n, sd = 0.2)
#' model <- xgboost(X, y, nrounds = 15, max_depth = 2, verbosity = 0, nthread = 1)
#' explainer <- gazer(model)
#' loss_matrix <- loss(explainer, X, y)
#' dim(loss_matrix)
#'
#' @seealso \code{\link{qshap_loss}}
#' @export
loss <- function(explainer, x, y, y_mean_ori = NULL) {
  qshap_loss(explainer, x, y, y_mean_ori)
}
