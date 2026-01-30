library(xgboost)
library(jsonlite)
library(qshapr)

# --- helper: eta extraction ---
xgb_get_eta <- function(tree_model, model_json, default = 0.3) {
  eta <- tree_model$params$eta
  if (is.null(eta)) {
    gb <- model_json$learner$gradient_booster
    eta <- gb$gbtree_train_param$learning_rate
    if (is.null(eta)) eta <- gb$gbtree_train_param$eta
  }
  if (is.null(eta)) eta <- default
  as.numeric(eta)
}

# --- sanity check: per-tree increment vs per-tree SHAP additivity ---
check_tree_increment <- function(model, model_json, X, tree_idx = 2L) {
  eta <- xgb_get_eta(model, model_json)

  # prediction increment from xgboost (tree_idx-th tree)
  p1 <- predict(model, X, iterationrange = c(1, tree_idx))
  p2 <- predict(model, X, iterationrange = c(1, tree_idx + 1L))
  inc_pred <- p2 - p1

  # marginal SHAP (tree_idx-th tree), summed over features (+bias change included)
  c1 <- predict(model, X, predcontrib = TRUE, iterationrange = c(1, tree_idx))
  c2 <- predict(model, X, predcontrib = TRUE, iterationrange = c(1, tree_idx + 1L))
  inc_shap <- rowSums(c2 - c1)

  cat("eta =", eta, "\n")
  cat("max(|inc_pred - inc_shap|) =", max(abs(inc_pred - inc_shap)), "\n")
}

# --- train ---
set.seed(0)
n <- 100; p <- 10
X <- matrix(runif(n * p), n, p)
y <- X[,1] + 2 * X[,2] + 0.5 * X[,3] + rnorm(n, sd=0.1)

dtrain <- xgb.DMatrix(X, label = y)
model <- xgb.train(
  data = dtrain,
  nrounds = 20,
  params = list(objective="reg:squarederror", eta=0.5, max_depth=3),
  verbose = 0
)

# --- get JSON (same as your explainer does) ---
tmp <- tempfile(fileext = ".json")
xgb.save(model, tmp)
model_json <- jsonlite::fromJSON(tmp, simplifyVector = FALSE)
unlink(tmp)

# --- run check ---
check_tree_increment(model, model_json, X, tree_idx = 2L)
check_tree_increment(model, model_json, X, tree_idx = 10L)  # optional











