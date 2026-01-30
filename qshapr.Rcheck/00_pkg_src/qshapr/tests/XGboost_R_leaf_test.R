library(xgboost)
library(jsonlite)

mean_abs_leaf_from_json <- function(model, tree_idx = 2L) {
  # tree_idx is 1-based in R
  fn <- tempfile(fileext = ".json")
  xgb.save(model, fn)
  mj <- jsonlite::fromJSON(fn, simplifyVector = FALSE)
  unlink(fn)

  gb <- mj$learner$gradient_booster
  trees <- if (!is.null(gb$model$trees)) gb$model$trees else gb$gbtree_model$trees
  tr <- trees[[tree_idx]]

  base_w <- as.numeric(unlist(tr$base_weights))
  left   <- as.integer(unlist(tr$left_children))
  is_leaf <- (left == -1L)

  mean(abs(base_w[is_leaf]))
}

run <- function(eta) {
  set.seed(0)
  n <- 200; p <- 10
  X <- matrix(runif(n*p), n, p)
  y <- X[,1] + 2*X[,2] + 0.5*X[,3] + rnorm(n, sd=0.1)

  dtrain <- xgb.DMatrix(X, label=y)
  model <- xgb.train(
    data = dtrain,
    nrounds = 20,
    params = list(objective="reg:squarederror", max_depth=3, eta=eta),
    verbose = 0
  )

  m <- mean_abs_leaf_from_json(model, tree_idx = 2L)
  cat("eta=", eta, " mean(|json leaf_value|)=", m, "\n")
}

run(1.0)
run(0.3)
run(0.1)