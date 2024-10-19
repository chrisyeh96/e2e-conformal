"""
This module implements optimization over ellipsoid uncertainty.
"""
import cvxpy as cp
import numpy as np


class EllipsoidProblem:
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

        # parameters
        loc = cp.Parameter(y_dim, name='loc')
        scale_tril = cp.Parameter((y_dim, y_dim), name='scale_tril')
        self.params = {
            'loc': loc,
            'scale_tril': scale_tril,
        }

        Fz_ystd = cp.multiply(y_std, Fz)

        # objective
        self.dual_obj = (
            cp.norm(scale_tril.T @ Fz_ystd)
            + loc @ Fz_ystd
            + y_mean @ Fz
        )

        # ellipsoid problem has no dual variables or constraints
        self.dual_vars = {}
        self.dual_constraints = []

    def solve(self, loc: np.ndarray, scale_tril: np.ndarray) -> cp.Problem:
        """
        Args:
            loc: shape [y_dim], predicted mean
            scale_tril: shape [y_dim, y_dim], Cholesky factor of covariance matrix

        Returns:
            prob: solved problem
        """
        self.params['loc'].value = loc
        self.params['scale_tril'].value = scale_tril
        prob = self.prob
        prob.solve()
        if prob.status != 'optimal':
            print('Problem status:', prob.status)
        return prob
