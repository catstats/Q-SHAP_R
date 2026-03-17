# qshap: Feature-Specific $R^2$ Values for Boosting Trees in R

[![CRAN status](https://www.r-pkg.org/badges/version/misspi)](https://CRAN.R-project.org/package=qshap)
[![](http://cranlogs.r-pkg.org/badges/grand-total/misspi?color=blue)](https://cran.r-project.org/package=qshap)
[![R-hub](https://github.com/catstats/Q-SHAP_R/actions/workflows/rhub.yaml/badge.svg)](https://github.com/catstats/Q-SHAP_R/actions/workflows/rhub.yaml)

This R package computes feature-specific $R^2$ values using Shapley decomposition of the total $R^2$ for Boosting Trees in polynomial time based on the [paper](https://dl.acm.org/doi/10.5555/3762387.3762469)

Currently supports **XGBoost** and **LightGBM** models.

## Key Features

- **Fast computation**: Polynomial time complexity for Shapley value calculation
- **Multiple models**: Support for XGBoost and LightGBM
- **Parallel processing**: Built-in support for multi-core processing
- **Rich visualizations**: Multiple plot types for interpreting results
- **Memory efficient**: Options for sampling large datasets

## Installation

You can install `qshap` from GitHub:

```r
# Install devtools if you haven't already
install.packages("devtools")

# Install qshap from GitHub
devtools::install_github("catstats/Q-SHAP_R")
```

## Quick Start with XGBoost

```r
# Load required libraries
library(xgboost)
library(qshap)
library(ggplot2)

# Load the Boston Housing dataset
data(Boston, package = "MASS")

X <- Boston[, -14]  # All columns except medv (target)
y <- Boston$medv


# Train XGBoost model
model <- xgboost(
  x = as.matrix(X),
  y = y,
  nrounds = 50,
  max_depth = 2,
  learning_rate = 0.1,
  objective = "reg:squarederror",
)

# Create Q-SHAP explainer
explainer <- gazer(model)

# Calculate feature-specific R^2 values using the rsq() wrapper
# This returns a qshap_result object with enhanced formatting
result <- rsq(explainer, X, y)

# Print shows top 10 features automatically
print(result)

# Get detailed summary with custom number of top features
summary(result, n = 5)

# Convert to data frame for further analysis
df <- as.data.frame(result)

# Calculate loss contributions directly using loss() alias
loss_matrix <- loss(explainer, X, y)

# Calculate model R^2 for verification
ypred <- predict(model, as.matrix(X))
sst <- sum((y - mean(y))^2)
sse <- sum((y - ypred)^2)
model_rsq <- 1 - sse/sst

print(paste("Total R^2:", round(sum(result$rsq), 4)))
print(paste("Model R^2:", round(model_rsq, 4)))

# Visualize feature-specific R^2
plot(
  result,
  label = colnames(X),
  rotation = 45,
  color_map_name = "Blues",
  title = "Feature-Specific R² (XGBoost)"
)
```

## Example with LightGBM

```r
# Load required libraries
library(lightgbm)
library(qshap)

# Generate synthetic data with high dimension
set.seed(42)
n <- 1000
p <- 1000
X <- matrix(rnorm(n * p), nrow = n, ncol = p)
colnames(X) <- paste0("Feature_", 1:p)

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
explainer <- gazer(lgb_model)

# Calculate feature-specific R^2 values
result <- rsq(explainer, X, y)

# Calculate model R^2 for verification
ypred <- predict(lgb_model, X)
sst <- sum((y - mean(y))^2)
sse <- sum((y - ypred)^2)
model_rsq <- 1 - sse/sst

# Print results
print(result)
print(paste("Total R^2:", round(sum(result$rsq), 4)))
print(paste("Model R^2:", round(model_rsq, 4)))

# Visualize
plot(
  result,
  label = colnames(X),
  rotation=45,
  color_map_name = "Greens",
  title = "Feature-Specific R² (LightGBM)"
)
```

## Advanced Usage

### Parallel Processing

For large datasets, use parallel processing to speed up calculations:

```r
# Use 4 cores for parallel processing
rsq.result <- rsq(explainer, X, y, ncore = 4)

# Use all available cores
rsq.result <- rsq(explainer, X, y, ncore = -1)
```

### Sampling Large Datasets

When working with very large datasets, you can sample a subset:

```r
# Sample 512 observations
rsq.result <- rsq(explainer, X, y, nsample = 512, random_state = 42)

# Or use a fraction of the data
rsq.result <- rsq(explainer, X, y, nfrac = 0.1, random_state = 42)
```

### Visualization Options

The package provides multiple visualization functions accessible through the `plot()` method:

```r
# Standard bar plot
feature_names <- colnames(X)

plot(rsq.result, label = feature_names, color_map_name = "Blues", rotation=45)

# Horizontal bar plot
plot(rsq.result, label = feature_names, horizontal = TRUE)

# Elbow plot (top features)
plot(rsq.result, type = "elbow", label = feature_names, max_comp = 10, rotation=45)

# Cumulative explained variance
plot(rsq.result, type = "cumu", label = feature_names, max_comp = 10)

# Generalized correlation (sqrt of R²)
plot(rsq.result, type = "gcorr", label = feature_names, rotation=45)
```

## API Reference

### Main Functions

- `gazer(model)`: Create a Q-SHAP explainer from a trained model
  - Returns a `qshap_tree_explainer` object with `print()` and `summary()` methods
- `rsq(explainer, X, y, ...)`: Calculate feature-specific R² values
  - Returns a `qshap_result` object with enhanced formatting and methods
  - Automatically extracts feature names and includes metadata
  - Provides `print()`, `summary()`, and `as.data.frame()` methods
- `qshap_result(rsq, feature_names, ...)`: Create a Q-SHAP result object
  - Returns a `qshap_result` object with `print()`, `summary()`, and `as.data.frame()` methods
- `loss(explainer, X, y)`: Calculate feature-specific loss contributions


### S3 Classes

The package uses a formal S3 class system for better structure and usability:

#### `qshap_tree_explainer`

Created by `gazer()`. Contains the preprocessed model information for fast SHAP computation.

```r
explainer <- gazer(model)

# Print summary information
print(explainer)
#> <qshap_tree_explainer>
#>   Model type: xgboost
#>   Number of trees: 50
#>   Max depth: 2
#>   Base score: 22.5328 

# Detailed summary
summary(explainer)
```

#### `qshap_result`

Stores Q-SHAP R² results with rich metadata and convenient methods.

```r
# Use rsq() to calculate feature-specific R² values
result <- rsq(explainer, X, y)

# Print top contributing features (default: top 10)
print(result)
#> <qshap_result>
#>  Total R²: 0.9082 
#>  Number of features: 13 
#>  Number of samples: 506 

#> Top 10 features by R²:
#> Feature R_squared
#>   lstat 0.4606137
#>      rm 0.3148459
#>     ...

# Get detailed statistics with custom number of top features
summary(result, n = 5)  # Show top 5 features

# Convert to data frame for further analysis
df <- as.data.frame(result)
```

### Visualization Functions

The recommended way to visualize Q-SHAP results is using the standard R `plot()` method:

- `plot(x, type = "rsq", ...)`: Bar plot of R² values (default)
- `plot(x, type = "elbow", ...)`: Elbow plot of top features
- `plot(x, type = "cumu", ...)`: Cumulative explained variance plot
- `plot(x, type = "gcorr", ...)`: Generalized correlation plot
- `plot(x, type = "hist", ...)`: Histogram of R² values



## Citation

```bibtex
@inproceedings{10.5555/3762387.3762469,
author = {Jiang, Zhongli and Zhang, Min and Zhang, Dabao},
title = {Fast calculation of feature contributions in boosting trees},
year = {2025},
publisher = {JMLR.org},
numpages = {17},
location = {Rio de Janeiro, Brazil},
series = {UAI '25}
}

```

## Reference
- Jiang, Z., Zhang, M., & Zhang, D. (2025). Fast calculation of feature contributions in boosting trees. *Proceedings of the 41st Conference on Uncertainty in Artificial Intelligence (UAI)*, 82:1859 - 1875


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
