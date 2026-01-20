# devtools::load_all() 
library(xgboost)
library(qshapr)

# Create synthetic data  
set.seed(0)
n_samples <- 100
p <- 10
X <- matrix(runif(n_samples * p), n_samples, p)
y <- X[,1] + 2 * X[,2] + 0.5 * X[,3] + rnorm(n_samples, sd=0.1)


# Train XGBoost model 
model <- xgboost(
  data = X,
  label = y,
  objective = "reg:squarederror",
  nrounds = 20,
  eta = 1,
  max_depth = 2,
  verbose = 0
)


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