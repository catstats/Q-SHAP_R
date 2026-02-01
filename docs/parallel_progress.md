# Progress Bar for Parallel Computing

## Overview

This document describes the progress bar feature added for parallel computing in the `qshap_rsq` function.

## Background

When processing large datasets with multiple cores, it's useful to see the progress of the computation. The `qshap_rsq` function now includes a progress bar that displays real-time updates when using parallel processing.

## How It Works

### Serial Processing (ncore = 1)

When using a single core, the progress bar is shown at the tree level (iterating through each tree in the model). This is the existing behavior in the `qshap_loss_xgboost` and `qshap_loss_lightgbm` functions.

```r
# Serial processing with tree-level progress
phi_rsq <- qshap_rsq(explainer, X, y, ncore = 1)
```

Output:
```
Progress [==========] 50/50 (100%)
```

### Parallel Processing (ncore > 1)

When using multiple cores, the data is divided into chunks and processed in parallel. The progress bar shows the completion of chunks rather than individual trees.

```r
# Parallel processing with chunk-level progress
phi_rsq <- qshap_rsq(explainer, X, y, ncore = 4)
```

Output:
```
Processing 1000 samples using 4 cores in 10 chunks...
  [==========] 10/10 chunks (100%) ETA:  0s
```

## Implementation Details

### Chunking Strategy

The implementation creates more chunks than cores to enable better progress tracking:

- **For 1-4 cores**: At least 2 chunks per core, minimum 10 chunks total
- **For 5+ cores**: At least 4 chunks per core, minimum 20 chunks total
- **Maximum**: Not more chunks than samples

For example:
- With 2 cores and 50 samples: Creates 10 chunks (minimum threshold)
- With 4 cores and 1000 samples: Creates 10 chunks (max of 8 from 2×4 cores or 10 minimum)
- With 8 cores and 1000 samples: Creates 32 chunks (4 per core for better progress granularity)
- With 16 cores and 100 samples: Creates 64 chunks, but limited to 100 (sample limit)

### Batched Processing

To balance parallelism with progress visibility:

1. Chunks are grouped into batches of size `ncore`
2. Each batch is processed in parallel (all cores working simultaneously)
3. Progress bar updates after each batch completes
4. This ensures true parallel processing while providing regular progress updates

### Progress Bar Format

The progress bar displays:
- Visual progress bar with percentage
- Current chunk / Total chunks
- Estimated time remaining (ETA)

## Examples

### Example 1: Basic Usage

```r
library(xgboost)
library(qshapr)

# Generate data
set.seed(42)
X <- matrix(rnorm(1000 * 10), nrow = 1000, ncol = 10)
y <- rowSums(X[, 1:3]) + rnorm(1000, 0, 0.5)

# Train model
model <- xgboost(data = X, label = y, nrounds = 100, verbose = 0)

# Create explainer
explainer <- gazer(model)

# Compute with parallel processing and progress bar
phi_rsq <- qshap_rsq(explainer, X, y, ncore = 4)
```

### Example 2: Large Dataset

```r
# For very large datasets
set.seed(42)
X <- matrix(rnorm(10000 * 50), nrow = 10000, ncol = 50)
y <- rowSums(X[, 1:10]) + rnorm(10000, 0, 0.5)

model <- xgboost(data = X, label = y, nrounds = 200, verbose = 0)
explainer <- gazer(model)

# Use all available cores
phi_rsq <- qshap_rsq(explainer, X, y, ncore = -1)
```

Output will show processing across all available cores with chunk progress:
```
Processing 10000 samples using 8 cores in 20 chunks...
  [====------] 8/20 chunks (40%) ETA: 15s
```

### Example 3: With Sampling

Progress bar also works when sampling a subset of the data:

```r
# Sample 1000 observations from a large dataset
phi_rsq <- qshap_rsq(explainer, X, y, nsample = 1000, ncore = 4, random_state = 42)
```

## Performance Considerations

### Trade-offs

1. **Progress Granularity**: More chunks = more frequent updates but slight overhead
2. **Batch Size**: Batches of `ncore` chunks balance parallelism with progress visibility
3. **Overhead**: Progress tracking adds minimal overhead (< 1% typically)

### When Progress Bar is Shown

- **Serial mode (ncore = 1)**: Progress shown at tree level (existing behavior)
- **Parallel mode (ncore > 1)**: Progress shown at chunk level (new behavior)
- **Very small datasets**: Progress may complete very quickly

## Technical Notes

### Dependencies

The progress bar uses the `progress` package, which is already a dependency of qshapr.

### Cluster Type

The implementation uses PSOCK clusters (via `parallel::makeCluster`), which are:
- Cross-platform compatible (Windows, Mac, Linux)
- CRAN-friendly
- Don't require forking support

### Thread Safety

The progress bar is updated in the main thread after each batch completes, avoiding thread safety issues common in parallel progress tracking.

## Comparison with Serial Processing

While parallel processing provides speedup, the progress bar may update less frequently than in serial mode:

**Serial mode**:
- Updates after each tree (e.g., 100 updates for 100 trees)
- Fine-grained progress
- No parallelism

**Parallel mode**:
- Updates after each batch of chunks (e.g., 10 updates for 10 chunks with 4 cores)
- Coarser-grained but still informative
- Full parallelism maintained

## Disabling Progress Output

If you want to suppress all output including the progress bar, you can use `suppressMessages()`:

```r
phi_rsq <- suppressMessages(
  qshap_rsq(explainer, X, y, ncore = 4)
)
```

## Future Enhancements

Potential improvements for future versions:

1. **Real-time progress**: Use asynchronous processing to update progress as chunks complete
2. **Tree-level progress in parallel**: Show detailed tree progress within each chunk
3. **Adaptive chunking**: Automatically determine optimal number of chunks based on data size and complexity
4. **Progress callbacks**: Allow users to provide custom progress reporting functions
