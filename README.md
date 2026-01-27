# qshapr: Feature-Specific $R^2$ Values for Tree Ensembles in R

This R package computes feature-specific $R^2$ values using Shapley decomposition of the total $R^2$ for tree ensembles in polynomial time based on the [paper](https://arxiv.org/abs/2407.03515).

Currently supports **XGBoost** and **LightGBM** models.

## Key Features

- **Fast computation**: Polynomial time complexity for Shapley value calculation
- **Multiple models**: Support for XGBoost and LightGBM
- **Parallel processing**: Built-in support for multi-core processing
- **Rich visualizations**: Multiple plot types for interpreting results
- **Memory efficient**: Options for sampling large datasets

## Installation

You can install `qshapr` from GitHub:

```r
# Install devtools if you haven't already
install.packages("devtools")

# Install qshapr from GitHub
devtools::install_github("catstats/Q-SHAP_R")
```

## Quick Start with XGBoost

```r
# Load required libraries
library(xgboost)
library(qshapr)

# Load the California Housing dataset
url <- "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
housing <- read.csv(url)

# Create features
X <- data.frame(
  MedInc = housing$median_income,
  HouseAge = housing$housing_median_age,
  AveRooms = housing$total_rooms / housing$households,
  AveBedrms = housing$total_bedrooms / housing$households,
  Population = housing$population,
  AveOccup = housing$population / housing$households,
  Latitude = housing$latitude,
  Longitude = housing$longitude
)

y <- housing$median_house_value

# Sample 1000 observations for demonstration
set.seed(42)
sample_idx <- sample(nrow(X), 1000)
X_sample <- X[sample_idx, ]
y_sample <- y[sample_idx]

# Train XGBoost model
model <- xgboost(
  data = as.matrix(X_sample),
  label = y_sample,
  nrounds = 50,
  max_depth = 2,
  verbose = 0
)

# Create Q-SHAP explainer
explainer <- create_tree_explainer(model)

# Calculate feature-specific R^2 values
phi_rsq <- qshap_rsq(explainer, X_sample, y_sample)

# Print R^2 values for each feature
print(phi_rsq)
print(paste("Total R^2:", sum(phi_rsq)))

# Visualize feature-specific R^2
vis$rsq(
  phi_rsq, 
  label = colnames(X_sample),
  rotation = 45,
  color_map_name = "Blues",
  title = "Feature-Specific R² (XGBoost)"
)
```

## Example with LightGBM

```r
# Load required libraries
library(lightgbm)
library(qshapr)

# Generate synthetic data
set.seed(42)
n <- 500
X <- matrix(rnorm(n * 8), nrow = n, ncol = 8)
colnames(X) <- paste0("Feature_", 1:8)

# True model: y depends mainly on first 3 features
y <- 2 * X[,1] + 1.5 * X[,2] - 0.8 * X[,3] + rnorm(n, 0, 0.5)

# Create LightGBM dataset
dtrain <- lgb.Dataset(data = X, label = y)

# Set parameters
params <- list(
  objective = "regression",
  metric = "rmse",
  num_leaves = 15,
  learning_rate = 0.1,
  verbose = -1
)

# Train model
lgb_model <- lgb.train(
  params = params,
  data = dtrain,
  nrounds = 50
)

# Create Q-SHAP explainer
explainer <- create_tree_explainer(lgb_model)

# Calculate feature-specific R^2 values
phi_rsq <- qshap_rsq(explainer, X, y)

# Print results
print(phi_rsq)
print(paste("Total R^2:", round(sum(phi_rsq), 4)))

# Visualize
vis$rsq(
  phi_rsq,
  label = colnames(X),
  color_map_name = "Greens",
  title = "Feature-Specific R² (LightGBM)"
)
```

## Advanced Usage

### Model R² Validation

By default, `qshap_rsq` validates that the sum of feature-specific R² contributions matches the model's actual R². This helps catch potential numerical issues or implementation bugs:

```r
# Default behavior includes validation
phi_rsq <- qshap_rsq(explainer, X, y)
# Warning will be issued if R² sum doesn't match model R² within tolerance

# Disable validation if needed
phi_rsq <- qshap_rsq(explainer, X, y, check_model_rsq = FALSE)

# Adjust tolerance for validation (default: 1e-6)
phi_rsq <- qshap_rsq(explainer, X, y, tolerance = 1e-8)
```

### Parallel Processing

For large datasets, use parallel processing to speed up calculations:

```r
# Use 4 cores for parallel processing
phi_rsq <- qshap_rsq(explainer, X, y, ncore = 4)

# Use all available cores
phi_rsq <- qshap_rsq(explainer, X, y, ncore = -1)
```

### Sampling Large Datasets

When working with very large datasets, you can sample a subset:

```r
# Sample 1024 observations
phi_rsq <- qshap_rsq(explainer, X, y, nsample = 1024, random_state = 42)

# Or use a fraction of the data
phi_rsq <- qshap_rsq(explainer, X, y, nfrac = 0.1, random_state = 42)
```

### Visualization Options

The package provides multiple visualization functions:

```r
# Standard bar plot
vis$rsq(phi_rsq, label = feature_names, color_map_name = "Blues")

# Horizontal bar plot
vis$rsq(phi_rsq, label = feature_names, horizontal = TRUE)

# Elbow plot (top features)
vis$elbow(phi_rsq, label = feature_names, max_comp = 10)

# Cumulative explained variance
vis$cumu(phi_rsq, label = feature_names, max_comp = 10)

# Generalized correlation (sqrt of R²)
vis$gcorr(phi_rsq, label = feature_names)
```

## Citation

```bibtex
@inproceedings{jiangfast,
  title={Fast Calculation of Feature Contributions in Boosting Trees},
  author={Jiang, Zhongli and Zhang, Min and Zhang, Dabao},
  booktitle={The 41st Conference on Uncertainty in Artificial Intelligence}
}
```

## API Reference

### Main Functions

- `create_tree_explainer(model)`: Create a Q-SHAP explainer from a trained model
- `qshap_rsq(explainer, X, y, ...)`: Calculate feature-specific R² values
- `qshap_loss(explainer, X, y)`: Calculate feature-specific loss contributions

### Visualization Functions

- `vis$rsq()`: Bar plot of R² values
- `vis$elbow()`: Elbow plot of top features
- `vis$cumu()`: Cumulative explained variance plot
- `vis$gcorr()`: Generalized correlation plot
- `vis$loss()`: Interactive loss explorer (requires shiny)

## References

- Jiang, Z., Zhang, M., & Zhang, D. Fast Calculation of Feature Contributions in Boosting Trees. In The 41st Conference on Uncertainty in Artificial Intelligence.
- Lundberg, Scott M., et al. "From local explanations to global understanding with explainable AI for trees." Nature Machine Intelligence 2.1 (2020): 56-67.
- Karczmarz, Adam, et al. "Improved feature importance computation for tree models based on the Banzhaf value." Uncertainty in Artificial Intelligence. PMLR, 2022.
- Bifet, Albert, Jesse Read, and Chao Xu. "Linear tree shap." Advances in Neural Information Processing Systems 35 (2022): 25818-25828.
- Chen, Tianqi, and Carlos Guestrin. "Xgboost: A scalable tree boosting system." Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2016.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Task List

- [ ] Task 1: Add Catboost support
- [ ] Task 2: Add more visualization examples
