"""
This module implements the battery storage problems.
"""
from collections.abc import Sequence
from typing import NamedTuple

import cvxpy as cp
import numpy as np
import torch
from torch import Tensor

from problems.box import BoxProblem, BoxProblemV2
from problems.ellipsoid import EllipsoidProblem
from problems.nonrobust import NonRobustProblem
from problems.picnn import PICNNProblem
from problems.protocols import (
    BoxProblemProtocol,
    EllipsoidProblemProtocol,
    NonRobustProblemProtocol,
    PICNNProblemProtocol
)


class Constants(NamedTuple):
    lam: float
    eps: float
    eff: float
    c_in: float
    c_out: float
    B: float


DEFAULT_CONSTANTS = Constants(
    lam=0.1,
    eps=0.05,
    eff=0.9,
    c_in=0.5,
    c_out=0.2,
    B=1
)


class StorageProblemBase:
    """
    Base class for the battery storage problem. This base class should be subclassed
    to implement specific uncertainty sets. The base class defines the primal problem
    variables, constraints, and the task loss.
    """
    # instance variables
    const: Constants
    constraints: list[cp.Constraint]
    f_tilde: cp.Expression | float
    Fz: cp.Expression
    primal_vars: dict[str, cp.Variable]

    # to be implemented in subclass
    prob: cp.Problem
    y_mean: np.ndarray
    y_std: np.ndarray

    def __init__(self, T: int, const: Constants = DEFAULT_CONSTANTS):
        self.const = const
        lam, eps, eff, c_in, c_out, B = self.const

        z_state = cp.Variable(T + 1, name='z_state', nonneg=True)
        z_in = cp.Variable(T, name='z_in', nonneg=True)
        z_out = cp.Variable(T, name='z_out', nonneg=True)
        self.primal_vars = {
            'z_in': z_in,
            'z_out': z_out,
            'z_state': z_state,
        }

        self.constraints = [
            # SOC constraints
            z_state[0] == B/2,
            z_state[1:] == z_state[:-1] + eff * z_in - z_out,
            z_state <= B,

            # ramp constraints
            z_in <= c_in,
            z_out <= c_out,
        ]

        self.Fz = z_in - z_out

        self.f_tilde = (
            lam * cp.norm2(z_state - B/2)**2
            + eps * cp.norm2(z_in)**2
            + eps * cp.norm2(z_out)**2
        )

    def task_loss_np(self, y: np.ndarray, is_standardized: bool) -> float:
        """Computes task loss for a single example.

        Args:
            y: shape [T], energy price
            is_standardized: whether y is standardized

        Returns:
            task_loss: task loss for one example
        """
        assert self.prob.value is not None, 'Problem must be solved first'
        lam = self.const.lam
        eps = self.const.eps
        B = self.const.B

        z_in = self.primal_vars['z_in'].value
        z_out = self.primal_vars['z_out'].value
        z_state = self.primal_vars['z_state'].value

        if is_standardized:
            y = y * self.y_std + self.y_mean

        task_loss = (
            y @ (z_in - z_out)
            + lam * np.linalg.norm(z_state - B/2)**2
            + eps * np.linalg.norm(z_in)**2
            + eps * np.linalg.norm(z_out)**2
        )
        return task_loss

    def task_loss_torch(
        self, y: Tensor, is_standardized: bool, solution: Sequence[Tensor]
    ) -> Tensor:
        """Computes task loss for a single example or a batch of examples.

        Args:
            y: shape [..., T], energy price
            is_standardized: whether y is standardized
            solution: tuple of (z_in, z_out, z_state, and additional dual variables)

        Returns:
            task_loss: shape [...], task loss for each example
        """
        z_in, z_out, z_state = solution[:3]

        lam = self.const.lam
        eps = self.const.eps
        B = self.const.B

        if is_standardized:
            y = y * torch.from_numpy(self.y_std) + torch.from_numpy(self.y_mean)

        task_loss = (
            torch.sum(y * (z_in - z_out), dim=-1)
            + lam * torch.norm(z_state - B/2, dim=-1)**2
            + eps * torch.norm(z_in, dim=-1)**2
            + eps * torch.norm(z_out, dim=-1)**2
        )
        assert task_loss.shape == y.shape[:-1]
        return task_loss


class StorageProblemNonRobust(StorageProblemBase, NonRobustProblem, NonRobustProblemProtocol):
    def __init__(self, T: int, y_mean: np.ndarray, y_std: np.ndarray):
        StorageProblemBase.__init__(self, T=T)
        NonRobustProblem.__init__(self, y_dim=T, y_mean=y_mean, y_std=y_std, Fz=self.Fz)
        NonRobustProblemProtocol.__init__(self)


class StorageProblemBox(StorageProblemBase, BoxProblem, BoxProblemProtocol):
    def __init__(self, T: int, y_mean: np.ndarray, y_std: np.ndarray):
        StorageProblemBase.__init__(self, T=T)
        BoxProblem.__init__(self, y_dim=T, y_mean=y_mean, y_std=y_std, Fz=self.Fz)
        BoxProblemProtocol.__init__(self)


class StorageProblemBoxV2(StorageProblemBase, BoxProblemV2, BoxProblemProtocol):
    def __init__(self, T: int, y_mean: np.ndarray, y_std: np.ndarray):
        StorageProblemBase.__init__(self, T=T)
        BoxProblemV2.__init__(self, y_dim=T, y_mean=y_mean, y_std=y_std, Fz=self.Fz)
        BoxProblemProtocol.__init__(self)


class StorageProblemEllipsoid(StorageProblemBase, EllipsoidProblem, EllipsoidProblemProtocol):
    def __init__(self, T: int, y_mean: np.ndarray, y_std: np.ndarray):
        StorageProblemBase.__init__(self, T=T)
        EllipsoidProblem.__init__(self, y_dim=T, y_mean=y_mean, y_std=y_std, Fz=self.Fz)
        EllipsoidProblemProtocol.__init__(self)


class StorageProblemPICNN(StorageProblemBase, PICNNProblem, PICNNProblemProtocol):
    def __init__(self, T: int, L: int, d: int, y_mean: np.ndarray, y_std: np.ndarray,
                 epsilon: float = 0.):
        StorageProblemBase.__init__(self, T=T)
        PICNNProblem.__init__(self, y_dim=T, L=L, d=d, y_mean=y_mean, y_std=y_std,
                              Fz=self.Fz, epsilon=epsilon)
        PICNNProblemProtocol.__init__(self)
