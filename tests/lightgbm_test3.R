# devtools::load_all() 
library(lightgbm)
library(qshapr)


# Create synthetic data  
set.seed(0)
n_samples <- 1000
p <- 1000
X <- matrix(runif(n_samples * p), n_samples, p)
y <- X[,1] + 2 * X[,2] + 0.5 * X[,3] + rnorm(n_samples, sd=0.1)

# Train LightGBM model# 
dtrain <- lgb.Dataset(data = X, label = y)
model <- lgb.train(
  params = list(objective = "regression", max_depth=3, learning_rate=0.02, min_data_in_leaf = 5, verbose = -1),
  data = dtrain,
  nrounds = 50,
  verbose = -1
)

# Calculate true model R²
ypred <- predict(model, X)
true_rsq <- 1 - sum((y - ypred)^2) / sum((y - mean(y))^2)

# Calculate Q-SHAP R² contributions
explainer <- qshapr::gazer(model)
t_qshap <- system.time({
rsq_contributions <- qshapr::qshap_rsq(explainer, X, y)
})
print(t_qshap)

# Show the two numbers that match
print(paste("Q-SHAP R² sum:", sum(rsq_contributions)))
print(paste("True Model R²:", true_rsq))


# rsq bar plot
vis$rsq(rsq_contributions)

# change palette
vis$rsq(rsq_contributions, color_map_name = "Pastel2")

# custom labels
feature_names <- paste0("feature", seq_along(rsq_contributions))
vis$rsq(rsq_contributions, label = feature_names, rotation = 45)

# horizontal plot and save
vis$rsq(rsq_contributions, horizontal = TRUE, model_rsq = FALSE, max_feature = 15, save_name = "rsq_eg")

# elbow plot
top_idx <- vis$elbow(rsq_contributions, max_comp = 15)

# cumulative explained
vis$cumu(rsq_contributions, max_comp = 15, save_name = "cumu_eg")


# interactive loss explorer
# (this launches a small shiny app)
rsq_contributions <- qshapr::qshap_rsq(explainer, X, y, loss=TRUE)
vis$loss(rsq_contributions[[2]]