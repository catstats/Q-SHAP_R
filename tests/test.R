# devtools::load_all()
suppressPackageStartupMessages({
  library(data.table)
  library(lightgbm)
  library(xgboost)
  library(qshapr)
  library(ggplot2)
})

X <- as.matrix(fread("tests/X_data.csv", header = TRUE))
y <- as.numeric(fread("tests/y_data.csv", header = TRUE)[[1]])
set.seed(0)

n_samples <- 1000
n_features <- 1000
n_informative <- 5

storage.mode(X) <- "double"

cat("X dim:", paste(dim(X), collapse=" x "), "\n")
cat("y len:", length(y), "\n")

max_depth    <- 7L
n_estimators <- 50L

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
max_depth = 2L
nrounds = 50L

dtrain <- xgb.DMatrix(data = X, label = y)

model <- xgb.train(
  data = dtrain,
  nrounds = nrounds,
  params = list(
    objective = "reg:squarederror",
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
rsq_cons <- qshapr::qshap_rsq(explainer, X, y, loss=TRUE)
t1 <- proc.time()
cat("time:", t1 - t0, "\n")
rsq_contributions <- rsq_cons[[1]]
cat("Q-SHAP R^2 sum:", sum(rsq_contributions), "\n")
cat("Model R^2 is:", model_rsq, "\n\n")


# if you would like to use sampling

t0 <- proc.time()
rsq_contributions_sample <- qshapr::qshap_rsq(explainer, X, y, nsample=128)
t1 <- proc.time()
cat("time:", t1 - t0, "\n")

# Let's check the real R^2
ypred <- predict(model, X)
sst <- sum((y - mean(y))^2)
sse <- sum((y - ypred)^2)
model_rsq <- 1 - sse / sst

cat("Q-SHAP R^2 sum is:", sum(rsq_contributions_sample), "\n")
cat("Model R^2 is:", model_rsq, "\n")


# Visualization

# rsq bar plot
qshapr::vis$rsq(rsq_contributions)

# change palette
qshapr::vis$rsq(rsq_contributions, color_map_name = "viridis")
qshapr::vis$rsq(rsq_contributions, color_map_name = "inferno")

# custom labels
feature_names <- paste0("feature", seq_along(rsq_contributions))
qshapr::vis$rsq(rsq_contributions, label = feature_names, rotation = 45)

# horizontal plot and save
qshapr::vis$rsq(rsq_contributions, horizontal = TRUE, model_rsq = FALSE, max_feature = 15, save_name = "rsq_eg")

# elbow plot
top_idx <- vis$elbow(rsq_contributions, max_comp = 15)

# cumulative explained
qshapr::vis$cumu(rsq_contributions, max_comp = 15, save_name = "cumu_eg")

# interactive loss explorer
# (this launches a small shiny app)
vis$loss(rsq_cons[[2]])