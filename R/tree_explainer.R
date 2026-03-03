#' @import Rcpp
#' @importFrom methods new
#' @importFrom parallel makeCluster stopCluster detectCores
#' @importFrom parallel clusterEvalQ clusterExport parLapply
#' @useDynLib qshap, .registration = TRUE
NULL

#' Create a QSHAP Tree Explainer
#' 
#' Creates an explainer object for computing feature-specific Shapley values
#' from a trained tree ensemble model. Supports XGBoost and LightGBM models.
#' 
#' @param model A model object of class \code{xgboost} or \code{xgb.Booster} from \pkg{xgboost}, or class \code{lgb.Booster} from \pkg{lightgbm}
#' @param max_depth Maximum depth of trees, extracted from \code{model} by default.
#' @param base_score Base score for predictions, extracted from \code{model} by default.
#' @param ... Additional arguments, for future use
#' 
#' @return A class of \code{qshap_tree_explainer} object containing the model information and
#'   preprocessed tree structures for fast Shapley value computation
#'   
#' @examples
#' \dontrun{
#' library(xgboost)
#' 
#' # Train a simple XGBoost model
#' set.seed(42)
#' X <- matrix(rnorm(100 * 5), nrow = 100, ncol = 5)
#' y <- rowSums(X[, 1:3]) + rnorm(100, 0, 0.1)
#' model <- xgboost(data = X, label = y, nrounds = 10, verbose = 0)
#' 
#' # Create explainer
#' explainer <- gazer(model)
#' }
#' 
#' @export
gazer <- function(model, max_depth = NULL, base_score = NULL, ...) {
  UseMethod("gazer")
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
 #' @param local Logical; if TRUE, returns both R-squared values and loss matrix
 #' @param sd_out Logical; if TRUE, returns standard deviations of R-squared estimates
 #' @param ci_out Logical; if TRUE, returns Wald-style confidence intervals for each feature's R-squared (normal approximation using sd_rsq)
 #' @param level Confidence level for the intervals (default 0.95)
 #' @param nsample Optional integer; number of samples to use (random subsample if less than nrow(x))
 #' @param nfrac Optional numeric in (0,1); fraction of samples to use (alternative to nsample)
 #' @param random_state Integer seed for reproducible sampling
 #' @param ncore Number of cores for parallel processing. Use -1 for all available cores, 
 #'   or a positive integer. Default is 1 (no parallelization)
 #' 
 #' @return If \code{local=FALSE} (default), returns a numeric vector of length p 
 #'   containing feature-specific R-squared values. If \code{local=TRUE}, returns 
 #'   a list with components \code{rsq} (the R-squared vector) and \code{loss} 
 #'   (an n x p matrix of loss contributions). When \code{ci_out=TRUE}, the returned list
 #'   also contains \code{ci_lower} and \code{ci_upper} vectors representing Wald-style confidence intervals.
 #'   
 #' @examples
 #' \dontrun{
 #' library(xgboost)
 #' 
 #' # Generate sample data
 #' set.seed(42)
 #' n <- 500
 #' X <- matrix(rnorm(n * 5), nrow = n, ncol = 5)
 #' colnames(X) <- paste0("Feature_", 1:5)
 #' y <- 2 * X[,1] + 1.5 * X[,2] - 0.8 * X[,3] + rnorm(n, 0, 0.5)
 #' 
 #' # Train model
 #' model <- xgboost(data = X, label = y, nrounds = 50, max_depth = 3, verbose = 0)
 #' 
 #' # Create explainer
 #' explainer <- gazer(model)
 #' 
 #' # Calculate R-squared values
 #' phi_rsq <- qshap_rsq(explainer, X, y)
 #' print(phi_rsq)
 #' 
 #' # With parallel processing
 #' phi_rsq_parallel <- qshap_rsq(explainer, X, y, ncore = 4)
 #' 
 #' # With sampling
 #' phi_rsq_sampled <- qshap_rsq(explainer, X, y, nsample = 100, random_state = 42)
 #' }
 #' 
 #' @keywords internal
qshap_rsq <- function(explainer, x, y, local = FALSE, nsample = NULL, sd_out = TRUE,
                      ci_out = TRUE, level = 0.95,
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

  # CI implies we need sd estimates
  if (isTRUE(ci_out)) sd_out <- TRUE
  if (is.null(level) || length(level) != 1 || is.na(level) || level <= 0 || level >= 1) {
    stop("level must be a single number in (0, 1)")
  }

  ci_from_sd <- function(rsq, sd_rsq, level) {
    z <- stats::qnorm(1 - (1 - level) / 2)
    list(
      ci_lower = rsq - z * sd_rsq,
      ci_upper = rsq + z * sd_rsq
    )
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

  # Calculate loss (serial)
  if (ncore == 1L || n <= 1L) {
    loss <- qshap_loss(explainer, x, y, y_mean_ori)
    rsq <- -colSums(loss) / sst

     if (sd_out) {
      # sample variance across i for each feature j
      loss_mean <- colMeans(loss)
      loss_var  <- colSums((loss - matrix(loss_mean, n, ncol(loss), byrow = TRUE))^2) / (n - 1)
      sd_rsq    <- sqrt(n * loss_var) / sst
    } else {
      sd_rsq <- NULL
    }
    if (ci_out && !is.null(sd_rsq)) {
      ci <- ci_from_sd(rsq, sd_rsq, level)
    } else {
      ci <- NULL
    }

    if (local) {
      out <- list(rsq = rsq, loss = loss, sd_rsq = sd_rsq)
    } else {
      out <- list(rsq = rsq, sd_rsq = sd_rsq)
    }
    if (!is.null(ci)) {
      out$ci_lower <- ci$ci_lower
      out$ci_upper <- ci$ci_upper
      out$level <- level
    }
    class(out) <- c("qshap_rsq", "list")
    return(out)
  }

  # Divide indices into ncore chunks (preserve order)
  idx_chunks <- split(seq_len(n), cut(seq_len(n), breaks = ncore, labels = FALSE))

  # Parallel compute using PSOCK cluster (CRAN/Windows friendly)
  cl <- parallel::makeCluster(ncore)
  on.exit(parallel::stopCluster(cl), add = TRUE)

  # Ensure package namespace is available on workers
  parallel::clusterEvalQ(cl, suppressPackageStartupMessages(library(qshap)))

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
      loss_var  <- colSums((loss - matrix(loss_mean, n_all, ncol(loss), byrow = TRUE))^2) / (n_all - 1)
      sd_rsq    <- sqrt(n_all * loss_var) / sst
    } else {
      sd_rsq <- NULL
    }
  if (ci_out && !is.null(sd_rsq)) {
    ci <- ci_from_sd(rsq, sd_rsq, level)
  } else {
    ci <- NULL
  }
  out <- list(rsq = rsq, loss = loss, sd_rsq = sd_rsq)
  if (!is.null(ci)) {
    out$ci_lower <- ci$ci_lower
    out$ci_upper <- ci$ci_upper
    out$level <- level
  }
  class(out) <- c("qshap_rsq", "list")
  return(out)

} else {
  # Combine sufficient statistics
  sum_all <- Reduce(`+`, lapply(results, `[[`, "sum"))
  sumsq_all <- Reduce(`+`, lapply(results, `[[`, "sumsq"))
  n_all <- sum(vapply(results, `[[`, numeric(1), "n"))

  # point estimate
  rsq <- -sum_all / sst

  if (sd_out) {
      sumsq_all <- Reduce(`+`, lapply(results, `[[`, "sumsq"))
      # unbiased sample variance across i
      loss_var <- (sumsq_all - (sum_all^2) / n_all) / (n_all - 1)
      sd_rsq   <- sqrt(n_all * loss_var) / sst
    } else {
      sd_rsq <- NULL
    }
  if (ci_out && !is.null(sd_rsq)) {
    ci <- ci_from_sd(rsq, sd_rsq, level)
  } else {
    ci <- NULL
  }
  out <- list(rsq = rsq, sd_rsq = sd_rsq)
  if (!is.null(ci)) {
    out$ci_lower <- ci$ci_lower
    out$ci_upper <- ci$ci_upper
    out$level <- level
  }
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
#'     \item \code{loss}: Loss matrix (if local=TRUE)
#'   }
#'
#' @details
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
#' \dontrun{
#' # Train a model
#' model <- xgboost(X, y, nrounds = 100)
#' explainer <- gazer(model)
#'
#' # Calculate R² contributions
#' result <- rsq(explainer, X, y)
#' print(result)  # Shows top 10 features by default
#' summary(result)  # Detailed summary
#' df <- as.data.frame(result)  # Convert to data frame
#'
#' # With custom feature names
#' result <- rsq(explainer, X, y, feature_names = c("Age", "Income", "Score"))
#'
#' # With parallel processing
#' result <- rsq(explainer, X, y, ncore = 4)
#' }
#'
#' @seealso \code{\link{qshap_result}}
#' @export
rsq <- function(explainer, x, y, feature_names = NULL, local = FALSE, nsample = NULL, 
                sd_out = TRUE, ci_out = TRUE, level = 0.95, nfrac = NULL, 
                random_state = 42, ncore = 1L) {
  
  # Call qshap_rsq
  result <- qshap_rsq(
    explainer = explainer,
    x = x,
    y = y,
    local = local,
    nsample = nsample,
    sd_out = sd_out,
    ci_out = ci_out,
    level = level,
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

  # Ensure loss is present only when local=TRUE
  if (!isTRUE(local) && !is.null(result$loss)) {
    result$loss <- NULL
  }

  class(result) <- c("qshap_result", "qshap_rsq", "list")
  return(result)
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
#' \dontrun{
#' # Train a model
#' model <- xgboost(X, y, nrounds = 100)
#' explainer <- gazer(model)
#'
#' # Calculate loss contributions using the shorter alias
#' loss_matrix <- loss(explainer, X, y)
#' }
#'
#' @seealso \code{\link{qshap_loss}}
#' @export
loss <- function(explainer, x, y, y_mean_ori = NULL) {
  qshap_loss(explainer, x, y, y_mean_ori)
}