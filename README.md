# End-to-End Conformal Calibration for Optimization Under Uncertainty

## Installation instructions

Code from this repo has been tested on Ubuntu 22.04.

Running code from this repo requires:
- python 3.12
- cvxpy 1.4
- cvxpylayers 0.1.6
- numpy 1.26
- pandas 2.2
- pytorch 2.4
- scikit-learn 1.5

We recommend using the [conda](https://docs.conda.io/) package manager.

1. Install [miniconda](https://docs.anaconda.com/miniconda/miniconda-install/).

2. Install the packages from the `env.yml` file:
```bash
conda env update --file env.yml --prune
```

3. Get a license for [MOSEK](https://www.mosek.com/). If you are an academic, you may request a free [academic license](https://www.mosek.com/products/academic-licenses/). Copy the license file to `~/mosek/mosek.lic`.


## Running code

1. Pre-train base models

    These scripts save CSVs file out to `out/{problem}_{model}/hyperparams*.csv`.

    ```bash
    # portfolio
    python run_portfolio_quantile.py best_hp --dataset synthetic --multiprocess 4 --device cuda
    python run_portfolio_gaussian.py best_hp --dataset synthetic --multiprocess 4 --device cuda

    # storage (no distribution shift)
    python run_storage_quantile.py best_hp --shuffle --multiprocess 4 --device cuda
    python run_storage_gaussian.py best_hp --shuffle --device cuda

    # storage (with distribution shift)
    python run_storage_quantile.py best_hp --multiprocess 4 --device cuda
    python run_storage_gaussian.py best_hp --device cuda
    ```

    Use the `analysis/{problem}.ipynb` notebooks to read the CSVs.

2. Run ETO

    ```bash
    # portfolio
    python run_portfolio_quantile.py eto --dataset synthetic --device cuda --multiprocess 4
    python run_portfolio_gaussian.py eto --dataset synthetic --device cuda --multiprocess 4
    python run_portfolio_picnn.py eto --dataset synthetic --lr 1e-2 1e-3 1e-4 --l2reg 1e-2 1e-3 1e-4 --multiprocess 4

    # storage (no distribution shift)
    python run_storage_quantile.py eto --shuffle --multiprocess 4 --device cuda
    python run_storage_gaussian.py eto --shuffle --multiprocess 4 --device cuda
    python run_storage_picnn.py eto --shuffle --lr 1e-2 1e-3 1e-4 --l2reg 1e-2 1e-3 1e-4 --multiprocess 4

    # storage (with distribution shift)
    python run_storage_quantile.py eto --multiprocess 4 --device cuda
    python run_storage_gaussian.py eto --multiprocess 4 --device cuda
    python run_storage_picnn.py eto --lr 1e-2 1e-3 1e-4 --l2reg 1e-2 1e-3 1e-4 --multiprocess 4
    ```

3. Run E2E

    ```bash
    # portfolio
    python run_portfolio_quantile.py e2e --dataset synthetic --lr 1e-2 1e-3 1e-4 --multiprocess 4
    python run_portfolio_gaussian.py e2e --dataset synthetic --lr 1e-2 1e-3 1e-4 --multiprocess 4
    python run_portfolio_picnn.py e2e --dataset synthetic --lr 1e-3 1e-4 --multiprocess 4

    # storage (no distribution shift)
    python run_storage_quantile.py e2e --shuffle --lr 1e-2 1e-3 1e-4 --multiprocess 4
    python run_storage_gaussian.py e2e --shuffle --lr 1e-2 1e-3 1e-4 --multiprocess 4
    python run_storage_picnn.py e2e --shuffle --lr 1e-3 1e-4 --multiprocess 4

    # storage (with distribution shift)
    python run_storage_quantile.py e2e --lr 1e-2 1e-3 1e-4 --multiprocess 4
    python run_storage_gaussian.py e2e --lr 1e-3 1e-4 --multiprocess 4
    python run_storage_picnn.py e2e --lr 1e-3 1e-4 --multiprocess 4
    ```

4. Run PTC baselines

    ```bash
    # portfolio optimization
    python run_portfolio_ptc.py best_hp --dataset synthetic --device cuda
    python run_portfolio_ptc.py ptc_box --dataset synthetic --multiprocess 4 --device cuda
    python run_portfolio_ptc.py ptc_ellipse --dataset synthetic --multiprocess 4 --device cuda
    python run_portfolio_ptc.py ptc_ellipse_johnstone --dataset synthetic --multiprocess 4 --device cuda

    # storage (with distribution shift)
    python run_storage_ptc.py best_hp --device cuda
    python run_storage_ptc.py ptc_box --multiprocess 4 --device cuda
    python run_storage_ptc.py ptc_ellipse --multiprocess 4 --device cuda
    python run_storage_ptc.py ptc_ellipse_johnstone --multiprocess 4 --device cuda

    # storage (no distribution shift)
    python run_storage_ptc.py best_hp --shuffle --device cuda
    python run_storage_ptc.py ptc_box --shuffle --multiprocess 4 --device cuda
    python run_storage_ptc.py ptc_ellipse --shuffle --multiprocess 4 --device cuda
    python run_storage_ptc.py ptc_ellipse_johnstone --shuffle --multiprocess 4 --device cuda
    ```

## Problems structure

This section describes how the underlying mathematical problem is organized in code. Consider the generic problem:

$$
\begin{aligned}
    \min_{z \in \mathbb{R}^p} \max_{\hat{y} \in \Omega(x)} \quad
    & \hat{y}^\top Fz + \tilde{f}(x,z) \\
    \text{s.t.} \quad
    & g(x,z) \leq 0.
\end{aligned}
$$

For the specific types of uncertainty sets $\Omega(x)$ considered in our paper, the inner maximization problem $\max_{\hat{y} \in \Omega(x)} \ \hat{y}^\top Fz$ has a strong dual problem:

$$
\begin{aligned}
    \min_{\nu \in \mathbb{R}^d} \quad
    & h(\nu, Fz, \Omega(x)) \\
    \text{s.t.} \quad
    & A(\nu, Fz, \Omega(x)) \leq 0,\quad \nu \geq 0.
\end{aligned}
$$

Thus, the whole problem can be written as

$$
\begin{aligned}
    \min_{z \in \mathbb{R}^p, \nu \in \mathbb{R}^d} \quad
    & h(\nu, Fz, \Omega(x)) + \tilde{f}(x,z) \\
    \text{s.t.} \quad
    & A(\nu, Fz, \Omega(x)) \leq 0, \quad \nu \geq 0, \quad g(x,z) \leq 0
\end{aligned}
$$

There are 4 parts to every problem.

1. `{Task}ProblemBase`: abstract class that implements the "primal" problem components
    - variable $z$
    - parameter $F$
    - objective term $\tilde{f}(x,z)$
    - constraint $g(x,z) \leq 0$.

    This class also implements the task loss function with a `numpy` and a `torch` version.

2. `{UncertaintySet}Problem`: abstract class that implements the "dual" problem components
    - variable $\nu$
    - objective term $h(\nu, Fz, \Omega(x))$
    - constraints $A(\nu, Fz, \Omega(x)) \leq 0$ and $\nu \geq 0$

    This class also implements the `solve()` method.

3. `{UncertaintySet}ProblemProtocol`: protocol class that implements the "glue" for combining the primal and dual problem components into a single optimization problem.
    - objective: sum of the primal and dual objectives
    - constraints: combination of the primal and dual constraints

    This class also implements the `get_cvxpylayer()` method.

4. `{Task}Problem{UncertaintySet}`: inherits from three parent classes (`{Task}ProblemBase`, `{UncertaintySet}Problem`, and `{UncertaintySet}ProblemProtocol`). This class is what is actually instantiated.

In our experiments:
- the choices for `{Task}` are `Storage` or `Portfolio`
- the choices for `{UncertaintySet}` are `NonRobust`, `Box`, `Ellipsoid`, or `PICNN`


### Example: battery storage optimization problem

In this problem, we have

- $z = (z^\text{in} \in \mathbb{R}^{24},\ z^\text{out} \in \mathbb{R}^{24},\ z^\text{state} \in \mathbb{R}^{25})$
- $Fz = z^\text{in} - z^\text{out}$
- $\tilde{f}(x,z) = \lambda \left\|z^\text{state} - \frac{B}{2} \mathbf{1}\right\|^2 + \epsilon \|z^\text{in}\|^2 + \epsilon \|z^\text{out}\|^2$
- $g(x,z) \leq 0$ is given by

    $$
    \begin{aligned}
    & z^\text{state}_0 = B/2,
    \qquad
    z^\text{state}_t = z^\text{state}_{t-1} - z^\text{out}_t + \gamma z^\text{in}_t \qquad\forall t=1,\dotsc,T \\
    & 0 \leq z^\text{in} \leq c^\text{in},
    \qquad
    0 \leq z^\text{out} \leq c^\text{out},
    \qquad
    0 \leq z^\text{state} \leq B.
    \end{aligned}
    $$

These components are implemented by

```python
class StorageProblemBase:
    # instance variables
    constraints: list[cp.Constraint]
    f_tilde: cp.Expression | float
    Fz: cp.Expression
    primal_vars: dict[str, cp.Variable]

    ...
```

### Example: box uncertainty

The box uncertainty set has $\Omega(x) = [\underline{y}(x), \overline{y}(x)]$, which results in the inner maximization problem

$$
\begin{aligned}
\max_{\hat{y} \in \mathbb{R}^n} \quad & \hat{y}^\top Fz \\
\text{s.t.} \quad & \underline{y}(x) \leq \hat{y} \leq \overline{y}.
\end{aligned}
$$

This has dual problem

$$
\begin{aligned}
\min_{\nu \in \mathbb{R}^n} \quad
& \left(\overline{y}(x) - \underline{y}(x)\right)^\top \nu + \underline{y}(x)^\top Fz \\
\text{s.t.} \quad
& \nu \geq 0,\quad \nu - Fz \geq 0.
\end{aligned}
$$

These components are implemented by

```python
class BoxProblem:
    # instance variables
    dual_constraints: list[cp.Constraint]
    dual_obj: cp.Expression
    dual_vars: dict[str, cp.Variable]
    params: dict[str, cp.Parameter]

    ...
```