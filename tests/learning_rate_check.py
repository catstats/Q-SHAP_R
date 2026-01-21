import numpy as np
import xgboost as xgb
import shap

# --- data ---
np.random.seed(0)
n, p = 200, 5
X = np.random.randn(n, p)
y = 3*X[:,0] - 2*X[:,1] + 0.5*X[:,2] + np.random.randn(n)*0.1

dX = xgb.DMatrix(X, label=y)

def check_eta(eta):
    params = {
        "objective": "reg:squarederror",
        "max_depth": 3,
        "eta": eta,
        "seed": 0,
    }
    model = xgb.train(params, dX, num_boost_round=20)

    # 1) global additivity check (SHAP + base == predict)
    expl = shap.TreeExplainer(model)
    shap_vals = expl.shap_values(X)               # (n, p)
    base = expl.expected_value                    # scalar for regression
    pred = model.predict(dX)                      # (n,)

    err = np.max(np.abs(shap_vals.sum(1) + base - pred))

    # 2) marginal tree SHAP magnitude check (tree 2 - tree 1)
    c1 = model.predict(dX, pred_contribs=True, iteration_range=(0, 1))
    c2 = model.predict(dX, pred_contribs=True, iteration_range=(0, 2))
    marg2 = c2 - c1
    marg_mean = np.mean(np.abs(marg2[:, :p]))

    return model, err, marg_mean

for eta in [1.0, 0.3, 0.1]:
    model, err, marg_mean = check_eta(eta)
    print(f"eta={eta:0.1f}  additivity_max_err={err:.2e}  mean(|marg_tree2_shap|)={marg_mean:.4f}")
