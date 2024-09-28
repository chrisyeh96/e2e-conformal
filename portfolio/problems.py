"""
This module implements the portfolio problems.
"""
from collections.abc import Sequence

import cvxpy as cp
import numpy as np
import torch
from torch import Tensor

from problems.box import BoxProblemV2
from problems.ellipsoid import EllipsoidProblem
from problems.nonrobust import NonRobustProblem
from problems.picnn import PICNNProblem
from problems.protocols import (
    BoxProblemProtocol,
    EllipsoidProblemProtocol,
    NonRobustProblemProtocol,
    PICNNProblemProtocol
)


class PortfolioProblemBase:
    """
    Base class for the portfolio optimization problem. This base class should be
    subclassed to implement specific uncertainty sets. The base class defines the primal
    problem variables, constraints, and the task loss.
    """
    # instance variables
    constraints: list[cp.Constraint]
    f_tilde: cp.Expression | float
    Fz: cp.Expression
    primal_vars: dict[str, cp.Variable]

    # to be implemented in subclass
    prob: cp.Problem
    y_mean: np.ndarray
    y_std: np.ndarray

    def __init__(self, N: int):
        """N is the dimension of y (or z)"""
        self.N = N

        z = cp.Variable(N, name='z', nonneg=True)
        self.primal_vars = {'z': z}

        self.constraints = [
            cp.sum(z) == 1,
        ]

        self.Fz = -z
        self.f_tilde = 0.

    def task_loss_np(self, y: np.ndarray, is_standardized: bool) -> float:
        """Computes task loss for a single example.

        Args:
            y: shape [N], energy price
            is_standardized: whether y is standardized

        Returns:
            task_loss: task loss for one example
        """
        assert self.prob.value is not None, 'Problem must be solved first'
        if is_standardized:
            y = y * self.y_std + self.y_mean
        z = self.primal_vars['z'].value
        task_loss = y @ -z
        return task_loss

    def task_loss_torch(
        self, y: Tensor, is_standardized: bool, solution: Sequence[Tensor]
    ) -> Tensor:
        """Computes task loss for a single example or a batch of examples.

        Args:
            y: shape [..., N], returns
            is_standardized: whether y is standardized
            solution: tuple of (z, and possibly dual variables)

        Returns:
            task_loss: shape [...], task loss for each example
        """
        if is_standardized:
            y = y * torch.from_numpy(self.y_std) + torch.from_numpy(self.y_mean)

        z = solution[0]
        assert y.shape == z.shape
        task_loss = (y * -z).sum(dim=-1)
        return task_loss


class PortfolioProblemNonRobust(PortfolioProblemBase, NonRobustProblem, NonRobustProblemProtocol):
    def __init__(self, N: int, y_mean: np.ndarray, y_std: np.ndarray):
        PortfolioProblemBase.__init__(self, N=N)
        NonRobustProblem.__init__(self, y_dim=N, y_mean=y_mean, y_std=y_std, Fz=self.Fz)
        NonRobustProblemProtocol.__init__(self)


class PortfolioProblemBox(PortfolioProblemBase, BoxProblemV2, BoxProblemProtocol):
    def __init__(self, N: int, y_mean: np.ndarray, y_std: np.ndarray):
        PortfolioProblemBase.__init__(self, N=N)
        BoxProblemV2.__init__(self, y_dim=N, y_mean=y_mean, y_std=y_std, Fz=self.Fz)
        BoxProblemProtocol.__init__(self)


class PortfolioProblemEllipsoid(PortfolioProblemBase, EllipsoidProblem, EllipsoidProblemProtocol):
    def __init__(self, N: int, y_mean: np.ndarray, y_std: np.ndarray):
        PortfolioProblemBase.__init__(self, N=N)
        EllipsoidProblem.__init__(self, y_dim=N, y_mean=y_mean, y_std=y_std, Fz=self.Fz)
        EllipsoidProblemProtocol.__init__(self)


class PortfolioProblemPICNN(PortfolioProblemBase, PICNNProblem, PICNNProblemProtocol):
    def __init__(self, N: int, L: int, d: int, y_mean: np.ndarray, y_std: np.ndarray,
                 epsilon: float = 0.):
        PortfolioProblemBase.__init__(self, N=N)
        PICNNProblem.__init__(self, y_dim=N, L=L, d=d, y_mean=y_mean, y_std=y_std,
                              Fz=self.Fz, epsilon=epsilon)
        PICNNProblemProtocol.__init__(self)
