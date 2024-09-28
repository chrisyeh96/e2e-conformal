"""
This module implements non-robust optimization.
"""
import cvxpy as cp
import numpy as np


class NonRobustProblem:
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
        y = cp.Parameter(y_dim, name='y')
        self.params = {'y': y}

        # not really a "dual" objective, because we're not handling robustness here
        self.dual_obj = (cp.multiply(y, y_std) + y_mean) @ Fz

        # non-robust problem has no dual variables or constraints
        self.dual_vars = {}
        self.dual_constraints = []

    def solve(self, y: np.ndarray) -> cp.Problem:
        """
        Args:
            y: shape [y_dim], predicted y

        Returns:
            prob: solved problem
        """
        self.params['y'].value = y
        prob = self.prob
        prob.solve(solver=cp.CLARABEL)
        if prob.status != 'optimal':
            print('Problem status:', prob.status)
        return prob
