# devtools::load_all()
suppressPackageStartupMessages({
  library(data.table)
  library(lightgbm)
  library(xgboost)
  library(qshapr)
  library(ggplot2)
  library(ggrepel)
})

X <- as.matrix(fread("tests/X_data.csv", header = TRUE))
y <- as.numeric(fread("tests/y_data.csv", header = TRUE)[[1]])
set.seed(0)

n_samples <- 1000
n_features <- 1000
n_informative <- 5



# Simulate data: only first n_informative features affect y
X <- matrix(rnorm(n_samples * n_features), nrow = n_samples, ncol = n_features)

beta <- numeric(n_features)
beta[1:n_informative] <- runif(n_informative, min = 1.0, max = 2.0)

signal <- X %*% beta 
y <- as.numeric(signal + rnorm(n_samples, sd = 0.5))

cat("Simulated data with", n_informative, "informative features.\n")


storage.mode(X) <- "double"

cat("X dim:", paste(dim(X), collapse=" x "), "\n")
cat("y len:", length(y), "\n")

max_depth    <- 2L
n_estimators <- 100L

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
explainer <- qshapr::gazer(model)
rsq_contributions <- qshapr::qshap_rsq(explainer, X, y)
t1 <- proc.time()
cat("time:", t1 - t0, "\n")
cat("Q-SHAP R^2 sum:", sum(rsq_contributions$rsq), "\n")
cat("Model R^2 is:", model_rsq, "\n\n")

# Check confidence intervals:
rsq_contributions$sd_rsq
rsq_contributions$ci_lower
rsq_contributions$ci_upper

## xgboost test
max_depth = 2L
nrounds = 50L

model <- xgboost(
  X, y,
  nrounds = nrounds,
  learning_rate = 0.1,
  max_depth = max_depth
)

ypred <- predict(model, X)
sst <- sum((y - mean(y))^2)
sse <- sum((y - ypred)^2)
model_rsq <- 1 - sse / sst

t0 <- proc.time()
explainer <- qshapr::gazer(model)
# parallel computation with 10 cores (would be useful if n_samples is large or depth is high)
rsq_cons <- qshapr::qshap_rsq(explainer, X, y, local=TRUE, ncore = 10)
t1 <- proc.time()
cat("time:", t1 - t0, "\n")
rsq_contributions <- rsq_cons[[1]]
cat("Q-SHAP R^2 sum:", sum(rsq_contributions), "\n")
cat("Model R^2 is:", model_rsq, "\n\n")


# if you would like to use sampling

t0 <- proc.time()
rsq_contributions_sample <- qshapr::qshap_rsq(explainer, X, y, nsample=512)
t1 <- proc.time()
cat("time:", t1 - t0, "\n")

# Let's check the real R^2
ypred <- predict(model, X)
sst <- sum((y - mean(y))^2)
sse <- sum((y - ypred)^2)
model_rsq <- 1 - sse / sst

cat("Q-SHAP R^2 sum is:", sum(rsq_contributions_sample$rsq), "\n")
cat("Model R^2 is:", model_rsq, "\n")


# Visualization

# rsq bar plot
plot(rsq_contributions)

# change palette
plot(rsq_contributions, color_map_name = "viridis")
plot(rsq_contributions, color_map_name = "inferno")

# custom labels
feature_names <- paste0("f", seq_along(rsq_contributions$rsq))
plot(rsq_contributions, label = feature_names, rotation = 45)

# horizontal plot and save
plot(rsq_contributions, horizontal = TRUE, model_rsq = FALSE, max_feature = 15, save_name = "rsq_eg")

# elbow plot
top_idx <- plot(rsq_contributions, type = "elbow", max_comp = 15, label=feature_names)

# cumulative explained
plot(rsq_contributions, type = "cumu", max_comp = 15, save_name = "cumu_eg")

# interactive loss explorer
# (this launches a small shiny app)
# Note: vis$loss still uses the old interface
qshapr::vis$loss(rsq_cons[[2]])