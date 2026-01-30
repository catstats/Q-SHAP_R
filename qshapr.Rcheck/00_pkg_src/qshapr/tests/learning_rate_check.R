library(xgboost)

set.seed(0)
n <- 200; p <- 5
X <- matrix(rnorm(n*p), n, p)
y <- 3*X[,1] - 2*X[,2] + 0.5*X[,3] + rnorm(n, sd=0.1)

dtrain <- xgb.DMatrix(X, label = y)

check_eta <- function(eta, nrounds = 20, max_depth = 3) {
  params <- list(
    objective = "reg:squarederror",
    max_depth = max_depth,
    eta = eta,
    seed = 0
  )

  model <- xgb.train(params = params, data = dtrain, nrounds = nrounds, verbose = 0)

  # 1) additivity check: contribs (incl. BIAS) sums to prediction
  contrib_all <- predict(model, X, predcontrib = TRUE)  # n x (p+1), last col is BIAS
  pred <- predict(model, X)
  add_err <- max(abs(rowSums(contrib_all) - pred))

  # 2) marginal SHAP for tree 2 (round 2): contrib(1..2) - contrib(1..1)
  c1 <- predict(model, X, predcontrib = TRUE, iterationrange = c(1, 2))  # round 1 only
  c2 <- predict(model, X, predcontrib = TRUE, iterationrange = c(1, 3))  # rounds 1..2
  marg2 <- c2 - c1

  # mean abs marginal over features only (exclude BIAS)
  marg_mean <- mean(abs(marg2[, 1:p, drop = FALSE]))

  list(model = model, additivity_max_err = add_err, mean_abs_marg_tree2 = marg_mean)
}

for (eta in c(1.0, 0.3, 0.1)) {
  out <- check_eta(eta)
  cat(sprintf("eta=%.1f  additivity_max_err=%.2e  mean(|marg_tree2_shap|)=%.4f\n",
              eta, out$additivity_max_err, out$mean_abs_marg_tree2))
}
