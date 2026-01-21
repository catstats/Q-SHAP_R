# devtools::load_all() 
library(xgboost)
library(qshapr)

# Create synthetic data  
set.seed(0)
n_samples <- 1000
p <- 100
X <- matrix(runif(n_samples * p), n_samples, p)
y <- X[,1] + 2 * X[,2] + 0.5 * X[,3] ** 2 + rnorm(n_samples, sd=0.1)


dtrain <- xgb.DMatrix(data = X, label = y)

# using default base_score=0.5 can result in very poor prediction for small number of trees and eta.
model <- xgb.train(
  data = dtrain,
  nrounds = 5,
  params = list(
    objective = "reg:squarederror",
    eta = 0.01,
    base_score = mean(y),
    max_depth = 3
  ),
  verbose = 0
)

# dump_txt <- xgb.dump(model, with_stats = TRUE)
# head(xgb.model.dt.tree(text = dump_txt, use_int_id = TRUE))

# Calculate true model R²
ypred <- predict(model, X)
true_rsq <- 1 - sum((y - ypred)^2) / sum((y - mean(y))^2)

# Calculate Q-SHAP R² contributions
explainer <- qshapr::create_tree_explainer(model)
t_qshap <- system.time({
rsq_contributions <- qshapr::qshap_rsq(explainer, X, y)
})
t_qshap

# Show the two numbers that should match
print(paste("Q-SHAP R² sum:", sum(rsq_contributions)))
print(paste("True Model R²:", true_rsq))
