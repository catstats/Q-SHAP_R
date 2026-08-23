# Q-SHAP Algorithm Notes

## General Trees

The stable generic backend remains `T2()` / `T2_sample()`. It uses the existing Q-SHAP complex root-of-unity polynomial machinery:

- `store_complex_root()`
- `store_complex_v_invc()`
- `complex_dot_v2()`
- `store_z`
- `store_v_invc`

The experimental `T2_ol2d()` interface is present so the depth-linear product-tree kernel can be validated without replacing the stable backend. It uses the same complex root-of-unity direction as the stable code. This backend is for comparison and validation; the stable `T2()` backend remains the default.

Current comparison status:

- Stable `T2()`: complex-root unordered leaf-pair backend with `O(L^2 D^2)` pair arithmetic.
- Experimental `T2_ol2d()`: complex-root product-tree backend with `O(L^2 D)` asymptotic pair/path arithmetic.
- The product-tree backend has larger constants and can be slower for moderate depths.

## CatBoost Symmetric Trees

The fast CatBoost global backend is separate from the generic `T2_ol2d()` interface.
It groups by parsed leaf/path, not raw prediction value.

For one tree, reached leaf/path, and feature `j`, the backend caches:

- `T0_leaf[j]`
- `T2_leaf[j]`

For samples `G` reaching the same leaf/path:

- `m = |G|`
- `sum_r = sum residual_i`
- `sum_r2 = sum residual_i^2`

With CatBoost JSON leaf values already scaled by learning rate:

```text
a = T2_leaf[j]
b = 2 * T0_leaf[j]

loss_sum[j]   += m*a - b*sum_r
loss_sumsq[j] += m*a*a - 2*a*b*sum_r + b*b*sum_r2
```

The global coefficient of determination is:

```text
R_j^2 = -loss_sum[j] / SST
```

First-order TreeSHAP contribution:

The same bottom-up subset DP computes the ordinary TreeSHAP term `T0_leaf[j]`
and the quadratic Q-SHAP term `T2_leaf[j]` together for each reached CatBoost
leaf/path. This means the CatBoost global backend does not need to materialize
the native `n x (p + 1)` TreeSHAP matrix. CatBoost's native attribution
convention can differ slightly from the parsed path-dependent backend on deeper
trees. The Q-SHAP backend aggregates the leaf/path-level `T0` terms directly
with residual sums:

```text
loss_sum[j] += m*T2_leaf[j] - 2*T0_leaf[j]*sum_r
```

The R CatBoost API used here does not expose per-tree leaf indexes directly, so
the backend routes each sample through each symmetric tree using the parsed
split features and thresholds. Prediction values are not used for grouping:
equal numeric predictions can come from different paths. Leaf-level `T0`/`T2`
contributions are stored in compact active-feature vectors rather than dense
`p`-length temporary vectors, which matters most for high-dimensional settings.

For depth `D`, leaves `L = 2^D`, trees `T`, and reached leaves `U`, the batch complexity is:

```text
O(T [nD + ULD])
```

Worst case:

```text
O(T [nD + L^2D])
```

The local T0 routine returns an explicit `n x p` matrix, so it also pays the
output allocation cost. The implementation updates only active split features
when broadcasting grouped T0 values, giving:

```text
O(np + T [nD + U(LD + p)])
```

The `U p` term comes from the current dense leaf-vector container; a purely
global first-order aggregation can avoid both the `np` output matrix and dense
leaf-vector materialization.
