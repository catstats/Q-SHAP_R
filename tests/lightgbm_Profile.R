# devtools::load_all()
suppressPackageStartupMessages({
  library(lightgbm)
  library(qshapr)
})

set.seed(0)

# --------------------------
# 1) Data
# --------------------------
n_samples <- 1000
p <- 10
X <- matrix(runif(n_samples * p), n_samples, p)
y <- X[,1] + 2 * X[,2] + 0.5 * (X[,3]^2) + rnorm(n_samples, sd=0.1)

dtrain <- lgb.Dataset(data = X, label = y)

# --------------------------
# 2) Train
# --------------------------
num_tree <- 50
params <- list(
  objective = "regression",
  metric = "l2",
  learning_rate = 0.01,
  max_depth = 3,
  num_threads = 1,
  verbosity = -1
)

model <- lgb.train(
  params = params,
  data = dtrain,
  nrounds = num_tree
)

# --------------------------
# 3) True R^2
# --------------------------
ypred <- predict(model, X)
true_rsq <- 1 - sum((y - ypred)^2) / sum((y - mean(y))^2)

# --------------------------
# 4) Q-SHAP (your package)
# --------------------------


t_qshap <- system.time({
  explainer <- qshapr::create_tree_explainer(model)
  rsq_contributions <- qshapr::qshap_rsq(explainer, X, y)
})
print(t_qshap)

cat("Q-SHAP R^2 sum:", sum(rsq_contributions), "\n")
cat("True model R^2:", true_rsq, "\n")

# --------------------------
# 5) Profiling (base Rprof)
# --------------------------
Rprof("rprof_lgb.out", interval = 0.001, append = FALSE)
rsq_contributions2 <- qshapr::qshap_rsq(explainer, X, y)
Rprof(NULL)

print(summaryRprof("rprof_lgb.out"))
