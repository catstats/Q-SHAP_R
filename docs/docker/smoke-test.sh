set -euo pipefail

test_dir=$(mktemp -d)
Rscript - "$test_dir" <<'RS'
output_dir <- commandArgs(trailingOnly = TRUE)[[1L]]
stopifnot(packageVersion("qshap") == "3.0.0",
          packageVersion("xgboost") == "3.1.3.1",
          packageVersion("catboost") == "1.2.10",
          requireNamespace("lightgbm", quietly = TRUE))
housing <- OpenML::getOMLDataSet(data.id = 43939)[["data"]]
stopifnot(nrow(housing) == 20640L, "median_house_value" %in% names(housing))

x <- data.frame(a = seq_len(32L), b = sin(seq_len(32L)))
y <- 0.3 * x[["a"]] + 2 * x[["b"]]
xgb <- xgboost::xgboost(x = x, y = y, nrounds = 4L, max_depth = 2L,
                        nthreads = 1L, verbosity = 0L)
pool <- catboost::catboost.load_pool(as.matrix(x), label = y)
catboost_model <- catboost::catboost.train(pool, NULL, params = list(
    iterations = 4L, depth = 2L, thread_count = 1L, loss_function = "RMSE",
    logging_level = "Silent", allow_writing_files = FALSE
))
for (model in list(xgb, catboost_model)) {
    result <- qshap::rsq(qshap::gazer(model), x, y, local = TRUE)
    stopifnot(inherits(result, "qshap_result"),
              isTRUE(all.equal(unname(colSums(qshap::local_rsq(result))),
                               unname(result[["rsq"]]), tolerance = 1e-8)))
}
xgboost::xgb.save(xgb, file.path(output_dir, "model.json"))
write.csv(x, file.path(output_dir, "X.csv"), row.names = FALSE)
write.csv(data.frame(y = y), file.path(output_dir, "y.csv"), row.names = FALSE)
RS

python - "$test_dir" <<'PY'
from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from qshap import _qshap_cpp, gazer, vis
from xgboost import XGBRegressor

assert sys.version_info[:2] == (3, 13)
assert version("qshap") == "3.0.0"
test_dir = Path(sys.argv[1])
X = pd.read_csv(test_dir / "X.csv")
y = pd.read_csv(test_dir / "y.csv")["y"]
model = XGBRegressor(n_jobs=1)
model.load_model(test_dir / "model.json")
result = gazer(model).rsq(X.to_numpy(), y.to_numpy(), local=True, progress_bar=False)
np.testing.assert_allclose(result.local_rsq.sum(axis=0), result.rsq,
                           rtol=1e-8, atol=1e-10)
vis.rsq(result.rsq, label=X.columns, save_name=str(test_dir / "python-rsq"))
assert (test_dir / "python-rsq.pdf").is_file()
PY

python -m pip check
