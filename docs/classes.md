# qshapr S3 Class System

## Overview

The `qshapr` package uses a formal S3 class system to provide a professional and user-friendly interface.

## Core Classes

1. **`qshapr_tree_explainer`** - Main explainer object for computing SHAP values
2. **`qshap_result`** - Container for Q-SHAP R² results with rich metadata
3. **`simple_tree`** - Internal representation of a single tree structure
4. **`tree_summary`** - Summarized tree structure with computed statistics

## Usage Examples

### Working with Explainers

```r
library(xgboost)
library(qshapr)

# Train model
model <- xgboost(data = X, label = y, nrounds = 50, verbose = 0)

# Create explainer
explainer <- create_tree_explainer(model)

# Quick summary
print(explainer)
#> <qshapr_tree_explainer>
#>   Model type: xgboost
#>   Number of trees: 50
#>   Max depth: 6
#>   Base score: 0.5

# Detailed summary
summary(explainer)
```

### Working with Results

```r
# Calculate Q-SHAP values
phi_rsq <- qshap_rsq(explainer, X, y)

# Create result object for better presentation
result <- qshap_result(
  rsq = phi_rsq,
  feature_names = colnames(X),
  n_samples = nrow(X)
)

# Display top features
print(result)

# Get detailed statistics
summary(result)

# Convert to data frame for further analysis
df <- as.data.frame(result)
```

## Benefits

1. **Better error messages** - Validation catches problems early
2. **Self-documenting** - `print()` and `summary()` methods
3. **Easier analysis** - `as.data.frame()` method
4. **Professional interface** - Consistent with modern R packages
5. **Full backward compatibility** - Existing code continues to work
