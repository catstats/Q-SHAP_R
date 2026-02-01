#' @import Rcpp
#' @importFrom methods new
#' @importFrom parallel makeCluster stopCluster detectCores
#' @importFrom parallel clusterEvalQ clusterExport parLapply
#' @useDynLib qshapr, .registration = TRUE
NULL

#' Create a QSHAPR Tree Explainer
#' 
#' Creates an explainer object for computing feature-specific Shapley values
#' from a trained tree ensemble model. Supports XGBoost and LightGBM models.
#' 
#' @param tree_model A trained tree model object (xgb.Booster or lgb.Booster)
#' @param max_depth Maximum depth of trees (optional, extracted from model if not provided)
#' @param base_score Base score for predictions (optional, extracted from model if not provided)
#' @param ... Additional arguments (currently unused)
#' 
#' @return A qshapr_tree_explainer object containing the model information and
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
gazer <- function(tree_model, max_depth = NULL, base_score = NULL, ...) {
  UseMethod("gazer")
}

#' @export
gazer.xgb.Booster <- function(tree_model, ...) {


  tmp <- tempfile(fileext = ".json")
  xgboost::xgb.save(tree_model, tmp)
  model_json <- jsonlite::fromJSON(tmp, simplifyVector = FALSE)
  unlink(tmp)

  # For version 3.1.3.1 and later, we can get max_depth from attributes
  max_depth <- if (!is.null(attributes(tree_model)$params$max_depth)) attributes(tree_model)$params$max_depth else 6

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
  # eta <- tree_model$params$eta
  # if (is.null(eta)) {
  #   # common JSON locations depending on xgboost build
  #   eta <- model_json$learner$gradient_booster$gbtree_train_param$learning_rate
  #   if (is.null(eta)) eta <- model_json$learner$gradient_booster$gbtree_train_param$eta
  # }
  # if (is.null(eta)) eta <- 0.3
  # eta <- as.numeric(eta)

  xgb_trees <- xgb_formatter(model_json, max_depth)

  explainer <- new_qshapr_tree_explainer(
    model = tree_model,
    model_type = "xgboost",
    max_depth = max_depth,
    base_score = base_score,
    trees = xgb_trees,
    store_v_invc = store_complex_v_invc(max_depth * 2),
    store_z = store_complex_root(max_depth * 2)
  )
  
  validate_qshapr_tree_explainer(explainer)
  explainer
}

#' @export
gazer.lgb.Booster <- function(tree_model, max_depth = NULL, ...) {
  # Get max_depth from model parameters or use default
  max_depth <- if (!is.null(tree_model$params$max_depth)) tree_model$params$max_depth else 31
  
  # Format LightGBM trees
  lgb_trees <- lgb_formatter(tree_model, max_depth)

  explainer <- new_qshapr_tree_explainer(
    model = tree_model,
    model_type = "lightgbm",
    max_depth = max_depth,
    base_score = NULL,
    trees = lgb_trees,
    store_v_invc = store_complex_v_invc(max_depth * 2),
    store_z = store_complex_root(max_depth * 2)
  )
  
  validate_qshapr_tree_explainer(explainer)
  explainer
}

# #' @export
# gazer.gbm <- function(tree_model, max_depth = NULL, ...) {
# }

#' @export
gazer.default <- function(tree_model, ...) {
  stop(sprintf("gazer not implemented for class %s", class(tree_model)[1]))
}

#' Calculate Q-SHAP Loss Contributions
#' 
#' Computes the feature-specific loss contributions using Q-SHAP decomposition.
#' This is an internal function typically called by \code{qshap_rsq()}.
#' 
#' @param explainer A qshapr_tree_explainer object created by \code{gazer()}
#' @param x Feature matrix or data frame
#' @param y Response vector
#' @param y_mean_ori Optional pre-computed mean of y (for efficiency)
#' 
#' @return A matrix of loss contributions with dimensions (n_samples, n_features)
#' 
#' @export
qshap_loss <- function(explainer, x, y, y_mean_ori = NULL) {
  UseMethod("qshap_loss")
}

#' @export
qshap_loss.qshapr_tree_explainer <- function(explainer, x, y, y_mean_ori = NULL) {
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
 #' @param explainer A qshapr_tree_explainer object created by \code{gazer()}
 #' @param x Feature matrix or data frame with n samples and p features
 #' @param y Response vector of length n
 #' @param loss_out Logical; if TRUE, returns both R-squared values and loss matrix
 #' @param nsample Optional integer; number of samples to use (random subsample if less than nrow(x))
 #' @param nfrac Optional numeric in (0,1); fraction of samples to use (alternative to nsample)
 #' @param random_state Integer seed for reproducible sampling
 #' @param ncore Number of cores for parallel processing. Use -1 for all available cores, 
 #'   or a positive integer. Default is 1 (no parallelization)
 #' 
 #' @return If \code{loss_out=FALSE} (default), returns a numeric vector of length p 
 #'   containing feature-specific R-squared values. If \code{loss_out=TRUE}, returns 
 #'   a list with components \code{rsq} (the R-squared vector) and \code{loss} 
 #'   (an n x p matrix of loss contributions).
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
 #' @export 
qshap_rsq <- function(explainer, x, y, loss_out = FALSE, nsample = NULL,
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

  # Calculate loss (serial)
  if (ncore == 1L || n <= 1L) {
    loss <- qshap_loss(explainer, x, y, y_mean_ori)
    rsq <- -colSums(loss) / sst
    if (loss_out) {
      return(list(rsq = rsq, loss = loss))
    } else {
      return(rsq)
    }
  }

  # For better progress tracking with parallel processing, create more chunks
  # than cores. This allows load-balanced processing with progress updates.
  # Minimum of 10 chunks ensures reasonable progress granularity even with few cores,
  # while 2-4 chunks per core provides good load balancing for many-core systems.
  # For many cores, we create even more chunks to ensure frequent progress updates.
  if (ncore <= 4) {
    min_chunks <- max(ncore * 2, 10)  # 2 chunks per core, min 10
  } else {
    min_chunks <- max(ncore * 4, 20)  # 4 chunks per core for better progress with many cores
  }
  n_chunks <- min(min_chunks, n)  # But not more chunks than samples
  idx_chunks <- split(seq_len(n), cut(seq_len(n), breaks = n_chunks, labels = FALSE))

  message(sprintf("Processing %d samples using %d cores in %d chunks...", 
                  n, ncore, length(idx_chunks)))

  # Parallel compute using PSOCK cluster (CRAN/Windows friendly)
  cl <- parallel::makeCluster(ncore)
  on.exit(parallel::stopCluster(cl), add = TRUE)

  # Ensure package namespace is available on workers
  parallel::clusterEvalQ(cl, suppressPackageStartupMessages(library(qshapr)))

  # Export needed data once (avoid resending for every task)
  parallel::clusterExport(
    cl,
    varlist = c("x", "y", "explainer", "y_mean_ori", "loss_out"),
    envir = environment()
  )

  worker <- function(idx) {
    # Compute loss for this chunk
    lc <- qshapr::qshap_loss(explainer, x[idx, , drop = FALSE], y[idx], y_mean_ori)
    if (loss_out) {
      lc
    } else {
      colSums(lc)
    }
  }

  # Initialize progress bar
  pb <- progress::progress_bar$new(
    format = "  [:bar] :current/:total chunks (:percent) ETA: :eta",
    total = length(idx_chunks),
    clear = FALSE,
    width = 70
  )

  # Use a hybrid approach: process chunks in small batches with progress updates
  # This provides a good balance between parallelism and progress visibility
  # Each batch uses all available cores, processing 'ncore' chunks simultaneously
  batch_size <- ncore  # Process ncore chunks at a time (one per core)
  n_batches <- ceiling(length(idx_chunks) / batch_size)
  
  results <- vector("list", length(idx_chunks))
  
  for (batch_idx in seq_len(n_batches)) {
    # Determine which chunks to process in this batch
    start_chunk <- (batch_idx - 1) * batch_size + 1
    end_chunk <- min(batch_idx * batch_size, length(idx_chunks))
    batch_chunks <- idx_chunks[start_chunk:end_chunk]
    
    # Process this batch in parallel
    batch_results <- parallel::parLapply(cl, batch_chunks, worker)
    
    # Store results
    for (i in seq_along(batch_results)) {
      results[[start_chunk + i - 1]] <- batch_results[[i]]
      pb$tick()
    }
  }

  if (loss_out) {
    # Combine full loss matrix (row-bind like np.concatenate(axis=0))
    loss <- do.call(rbind, results)
    rsq <- -colSums(loss) / sst
    return(list(rsq = rsq, loss = loss))
  } else {
    # Combine by summing column-sums (memory efficient)
    loss_sum <- Reduce(`+`, results)
    rsq <- -loss_sum / sst
    return(rsq)
  }
}