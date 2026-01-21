# devtools::load_all()
library(xgboost)
library(qshapr)

set.seed(0)
n_samples <- 1000
p <- 1000
X <- matrix(runif(n_samples * p), n_samples, p)
y <- X[,1] + 2 * X[,2] + 0.5 * (X[,3] ^ 2) + rnorm(n_samples, sd=0.1)

dtrain <- xgb.DMatrix(data = X, label = y)

# ---- train ----
model <- xgb.train(
  data = dtrain,
  nrounds = 50,
  params = list(
    objective = "reg:squarederror",
    eta = 0.01,
    base_score = mean(y),
    max_depth = 3
  ),
  verbose = 0
)

# ---- true R^2 ----
ypred <- predict(model, X)
true_rsq <- 1 - sum((y - ypred)^2) / sum((y - mean(y))^2)

# ---- Q-SHAP ----
explainer <- qshapr::create_tree_explainer(model)

# Ensure single-threaded operation for fair timing
Sys.setenv(
  OMP_NUM_THREADS = "1",
  OPENBLAS_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1",
  VECLIB_MAXIMUM_THREADS = "1",
  NUMEXPR_NUM_THREADS = "1"
)

# timing
t_qshap <- system.time({
  rsq_contributions <- qshapr::qshap_rsq(explainer, X, y)
})
print(t_qshap)

cat("Q-SHAP R^2 sum: ", sum(rsq_contributions), "\n", sep="")
cat("True Model R^2:", true_rsq, "\n", sep="")

# -----------------------------
# Profiling
# -----------------------------

# A) profvis 
# profvis::profvis({
#   rsq_contributions <- qshapr::qshap_rsq(explainer, X, y)
# })

# B) base Rprof (works everywhere)
Rprof("rprof.out", interval = 0.001, append=FALSE)
rsq_contributions <- qshapr::qshap_rsq(explainer, X, y)
Rprof(NULL)
print(summaryRprof("rprof.out"))