"""
This module implements optimization over box uncertainty.
"""
import cvxpy as cp
import numpy as np


class BoxProblem:
    # instance variables
    dual_constraints: list[cp.Constraint]
    dual_obj: cp.Expression
    dual_vars: dict[str, cp.Variable]
    params: dict[str, cp.Parameter]

    # subclass must provide these instance variables
    prob: cp.Problem

    def __init__(
        self, y_dim: int, y_mean: np.ndarray, y_std: np.ndarray, Fz: cp.Expression
    ):
        self.y_mean = y_mean
        self.y_std = y_std

        nu = cp.Variable(2 * y_dim, nonneg=True)
        self.dual_vars = {'nu': nu}

        # parameters
        y_min = cp.Parameter(y_dim, name='y_min')
        y_max = cp.Parameter(y_dim, name='y_max')
        self.params = {
            'y_min': y_min,
            'y_max': y_max,
        }

        # objective
        self.dual_obj = (
            y_max @ nu[:y_dim] - y_min @ nu[y_dim:]
            + y_mean @ Fz
        )

        # constraints
        self.dual_constraints = [
            nu[:y_dim] - nu[y_dim:] == cp.multiply(y_std, Fz)
        ]

    def solve(self, pred_lo: np.ndarray, pred_hi: np.ndarray) -> cp.Problem:
        """
        Args:
            pred_lo: shape [y_dim], lower bound on standardized y
            pred_hi: shape [y_dim], upper bound on standardized y

        Returns:
            prob: solved problem
        """
        self.params['y_min'].value = pred_lo
        self.params['y_max'].value = pred_hi
        prob = self.prob
        prob.solve()
        if prob.status != 'optimal':
            print('Problem status:', prob.status)
        return prob


class BoxProblemV2:
    # instance variables
    dual_constraints: list[cp.Constraint]
    dual_obj: cp.Expression
    dual_vars: dict[str, cp.Variable]
    params: dict[str, cp.Parameter]

    # subclass must provide these instance variables
    prob: cp.Problem

    def __init__(
        self, y_dim: int, y_mean: np.ndarray, y_std: np.ndarray, Fz: cp.Expression
    ):
        self.y_mean = y_mean
        self.y_std = y_std

        nu = cp.Variable(y_dim, nonneg=True)
        self.dual_vars = {'nu': nu}

        # parameters
        y_min = cp.Parameter(y_dim, name='y_min')
        y_max = cp.Parameter(y_dim, name='y_max')
        self.params = {
            'y_min': y_min,
            'y_max': y_max,
        }

        c = cp.multiply(y_std, Fz)

        # objective
        self.dual_obj = (
            (y_max - y_min) @ nu
            + y_min @ c
            + y_mean @ Fz
        )

        # constraints and problem
        self.dual_constraints = [nu - c >= 0]

    def solve(self, pred_lo: np.ndarray, pred_hi: np.ndarray) -> cp.Problem:
        """
        Args:
            pred_lo: shape [y_dim], lower bound on standardized y
            pred_hi: shape [y_dim], upper bound on standardized y

        Returns:
            prob: solved problem
        """
        self.params['y_min'].value = pred_lo
        self.params['y_max'].value = pred_hi
        prob = self.prob
        prob.solve()
        if prob.status != 'optimal':
            print('Problem status:', prob.status)
        return prob
