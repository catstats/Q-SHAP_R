# Installing CatBoost for R

Q-SHAP calls the R `catboost` package directly. Installing Python's CatBoost
does not install the R package. For these replication scripts, install the
official R release binary before running CatBoost examples or benchmarks.

Install `remotes` and choose the asset matching your operating system and CPU
from the [official release instructions](https://catboost.ai/docs/en/installation/r-installation-binary-installation)
and [release assets](https://github.com/catboost/catboost/releases).
The examples below pin CatBoost 1.2.10:

```r
install.packages("remotes")

# macOS: Apple Silicon and Intel
remotes::install_url(
  "https://github.com/catboost/catboost/releases/download/v1.2.10/catboost-R-darwin-universal2-1.2.10.tgz"
)

# Linux x86_64: use this instead of the macOS command
# remotes::install_url(
#   "https://github.com/catboost/catboost/releases/download/v1.2.10/catboost-R-linux-x86_64-1.2.10.tgz"
# )

# Linux ARM64 uses catboost-R-linux-aarch64-1.2.10.tgz.
# Windows x86_64 uses catboost-R-windows-x86_64-1.2.10.tgz.
```

Run only the command matching your platform. If your R installation requires
single-architecture installation, add `INSTALL_opts = "--no-multiarch"` to
`install_url()`. Check installation by loading the package:

```r
library(catboost)
packageVersion("catboost")
sessionInfo()
```

If a release binary cannot be used on your system, follow the official
[source-build instructions](https://catboost.ai/docs/en/installation/r-installation-github-installation).
Use a separate local library if you need to preserve an existing installation.
Record R, CatBoost, and Q-SHAP versions with replication results.

Q-SHAP supports numeric, scalar CatBoost regression models with `SymmetricTree`,
`Depthwise`, or `Lossguide` growth policies. Categorical/CTR splits, text, and
embedding splits require additional adapters and are rejected by the current
implementation.
