# devtools::load_all()
suppressPackageStartupMessages({
  library(data.table)
  library(lightgbm)
  library(xgboost)
  library(qshapr)
})

# X <- as.matrix(fread("tests/X_data.csv", header = TRUE))
# y <- as.numeric(fread("tests/y_data.csv", header = TRUE)[[1]])
set.seed(0)

n_samples <- 1000
n_features <- 10
n_informative <- 5

# Design matrix
X <- matrix(rnorm(n_samples * n_features),
            nrow = n_samples,
            ncol = n_features)

# True coefficients (only first n_informative are non-zero)
coefficients <- numeric(n_features)
coefficients[1:n_informative] <- rnorm(n_informative)

# Response (no noise, sklearn default noise=0)
y <- as.numeric(X %*% coefficients)


storage.mode(X) <- "double"

cat("X dim:", paste(dim(X), collapse=" x "), "\n")
cat("y len:", length(y), "\n")

max_depth    <- 5L
n_estimators <- 5L

dtrain <- lgb.Dataset(data = X, label = y)

params <- list(
  objective = "regression",
  max_depth = max_depth,
  verbose   = -1
)


model <- lgb.train(
  params = params,
  data   = dtrain,
  nrounds = n_estimators,
  verbose = -1
)
ypred <- predict(model, X)
sst <- sum((y - mean(y))^2)
sse <- sum((y - ypred)^2)
model_rsq <- 1 - sse / sst


t0 <- proc.time()
explainer <- qshapr::create_tree_explainer(model)
rsq_contributions <- qshapr::qshap_rsq(explainer, X, y)
t1 <- proc.time()
cat("time:", t1 - t0, "\n")
cat("Q-SHAP R^2 sum:", sum(rsq_contributions), "\n")
cat("Model R^2 is:", model_rsq, "\n\n")



## xgboost test
max_depth = 5
nrounds = 5

dtrain <- xgb.DMatrix(data = X, label = y)

model <- xgb.train(
  data = dtrain,
  nrounds = nrounds,
  params = list(
    objective = "reg:squarederror",
    eta = 0.01,
    base_score = mean(y),
    max_depth = max_depth
  ),
  verbose = 0
)

ypred <- predict(model, X)
sst <- sum((y - mean(y))^2)
sse <- sum((y - ypred)^2)
model_rsq <- 1 - sse / sst


t0 <- proc.time()
explainer <- qshapr::create_tree_explainer(model)
rsq_contributions <- qshapr::qshap_rsq(explainer, X, y)
t1 <- proc.time()
cat("time:", t1 - t0, "\n")
cat("Q-SHAP R^2 sum:", sum(rsq_contributions), "\n")
cat("Model R^2 is:", model_rsq, "\n\n")


