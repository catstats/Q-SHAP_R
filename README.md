# qshap: Feature-Specific $R^2$ Values for Boosting Trees in R

[![CRAN status](https://www.r-pkg.org/badges/version/qshap)](https://CRAN.R-project.org/package=qshap)
[![Total downloads](https://cranlogs.r-pkg.org/badges/grand-total/qshap?color=blue)](https://cran.r-project.org/package=qshap)
[![Downloads per month](https://cranlogs.r-pkg.org/badges/qshap)](https://cran.r-project.org/package=qshap)
[![R-hub](https://github.com/catstats/Q-SHAP_R/actions/workflows/rhub.yaml/badge.svg)](https://github.com/catstats/Q-SHAP_R/actions/workflows/rhub.yaml)
[![License: GPL v2+](https://img.shields.io/badge/License-GPL%20v2+-blue.svg)](https://www.gnu.org/licenses/gpl-2.0)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2407.03515-B31B1B.svg)](https://doi.org/10.48550/arXiv.2407.03515)

This R package computes feature-specific $R^2$ values using Shapley decomposition of the total $R^2$ for Boosting Trees in polynomial time based on the [paper](https://dl.acm.org/doi/10.5555/3762387.3762469)

Currently supports **XGBoost**, **LightGBM**, and **CatBoost** models.

## Key Features

- **Fast computation**: Polynomial time complexity for Shapley value calculation
- **Multiple models**: Support for XGBoost, LightGBM, and CatBoost
- **Parallel processing**: Built-in support for multi-core processing
- **Rich visualizations**: Multiple plot types for interpreting results
- **Memory efficient**: Options for sampling large datasets

## Installation

You can install the released version of `qshap` from CRAN:

```r
install.packages("qshap")
```

You can also install the development version from GitHub:

```r
# Install remotes if you haven't already
install.packages("remotes")

remotes::install_github("catstats/Q-SHAP_R")
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
  nthread = 1,
  verbosity = 0
)

# Create Q-SHAP explainer
explainer <- gazer(model)

# Return the first tree already stored by gazer
tree <- get_tree(explainer, 1)
print(tree)

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

# Request both local decompositions
local_result <- rsq(explainer, X, y, local = TRUE)
raw_loss <- local_result$loss

# Observation-level contributions to the global R² decomposition
local_rsq <- local_result$local_rsq
stopifnot(isTRUE(all.equal(
  colSums(local_rsq),
  unname(local_result$rsq),
  check.attributes = FALSE
)))

# The heatmap displays local_rsq by default; n_show controls the number of observations
plot(
  local_result,
  type = "heatmap",
  feature_names = colnames(X),
  n_show = 20
)

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

# Load the same Boston Housing dataset used above
data(Boston, package = "MASS")

X <- as.matrix(Boston[, -14])  # All columns except medv (target)
y <- Boston$medv

# Create LightGBM dataset
dtrain <- lgb.Dataset(data = X, label = y)

# Set parameters
params <- list(
  objective = "regression",
  metric = "rmse",
  max_depth = 2,
  num_leaves = 4,
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

## Example with CatBoost

CatBoost support in `qshap` is an optional runtime backend. It uses the official
R `catboost` package directly; it does not call the Python package. Because
`catboost` for R is not distributed on CRAN, install it from the official
CatBoost R instructions before running the example:

```r
install.packages("remotes")

# Choose the OS-specific binary URL from:
# https://catboost.ai/docs/en/concepts/r-installation
remotes::install_url(
  "BINARY_URL",
  INSTALL_opts = c("--no-multiarch", "--no-test-load")
)

e.g.
remotes::install_url(

  "https://github.com/catboost/catboost/releases/download/v1.2.10/catboost-R-darwin-universal2-1.2.10.tgz",

  INSTALL_opts = c("--no-multiarch", "--no-test-load")

)

```

After installation, train a `catboost.Model` in R, pass it to `gazer()`, and
compute feature-specific R-squared values with `rsq()`:

Numeric scalar regression models support all three CatBoost grow policies.
`SymmetricTree` uses the optimized cached backend, while `Depthwise` and
`Lossguide` automatically fall back to the general-tree implementation.
Float32 split boundaries and `nan_mode` routing are preserved. Models that
contain categorical/CTR splits are rejected explicitly because those splits
cannot be represented as raw numeric input-column thresholds.

```r
# Load required libraries
library(catboost)
library(qshap)

# Load the same Boston Housing dataset used above
data(Boston, package = "MASS")

X <- as.matrix(Boston[, -14])  # All columns except medv (target)
y <- Boston$medv

# Create CatBoost dataset and train model
pool <- catboost.load_pool(data = X, label = y)

params <- list(
  loss_function = "RMSE",
  iterations = 50,
  depth = 2,
  learning_rate = 0.1,
  verbose = 0,
  allow_writing_files = FALSE
)

model <- catboost.train(pool, params = params)

# Create Q-SHAP explainer. qshap dispatches to gazer.catboost.Model(),
# exports the fitted CatBoost model to JSON, and parses the tree structure.
explainer <- gazer(model)

# Calculate feature-specific R^2 values
result <- rsq(explainer, X, y)

# Print results
print(result)

# Calculate model R^2 for verification
ypred <- catboost.predict(model, pool)
sst <- sum((y - mean(y))^2)
sse <- sum((y - ypred)^2)
model_rsq <- 1 - sse/sst

print(paste("Total R²:", round(sum(result$rsq), 4)))
print(paste("Model R²:", round(model_rsq, 4)))

# Visualize
plot(
  result,
  label = colnames(X),
  rotation = 45,
  color_map_name = "Oranges",
  title = "Feature-Specific R² (CatBoost)"
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

- `gazer(model)`: Create a Q-SHAP explainer from a trained model (XGBoost, LightGBM, or CatBoost)
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
