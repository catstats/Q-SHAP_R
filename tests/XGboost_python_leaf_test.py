import numpy as np
import xgboost as xgb
import json, tempfile, os

def mean_abs_leaf_from_json(booster, tree_idx=1):
    # tree_idx is 0-based here
    fd, fn = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    booster.save_model(fn)
    with open(fn, "r") as f:
        mj = json.load(f)
    os.remove(fn)

    trees = mj["learner"]["gradient_booster"]["model"]["trees"]
    tr = trees[tree_idx]

    base_w = np.array(tr["base_weights"], dtype=float)
    left = np.array(tr["left_children"], dtype=int)
    is_leaf = (left == -1)

    return float(np.mean(np.abs(base_w[is_leaf])))

def run(eta):
    rng = np.random.default_rng(0)
    n, p = 200, 10
    X = rng.random((n, p))
    y = X[:,0] + 2*X[:,1] + 0.5*X[:,2] + rng.normal(0, 0.1, n)

    dtrain = xgb.DMatrix(X, label=y)
    booster = xgb.train(
        params={"objective":"reg:squarederror", "max_depth":3, "eta":eta},
        dtrain=dtrain,
        num_boost_round=20,
        verbose_eval=False
    )

    m = mean_abs_leaf_from_json(booster, tree_idx=1)
    print(f"eta={eta}  mean(|json leaf_value|)={m:.6f}")

for eta in [1.0, 0.3, 0.1]:
    run(eta)