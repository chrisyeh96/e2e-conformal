from collections.abc import Sequence
from typing import Protocol

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np
from torch import Tensor

from models.picnn import PICNN


class BaseProblemProtocol(Protocol):
    # from primal problem
    constraints: list[cp.Constraint]
    f_tilde: cp.Expression | float
    Fz: cp.Expression
    primal_vars: dict[str, cp.Variable]

    # from dual problem
    dual_constraints: list[cp.Constraint]
    dual_obj: cp.Expression
    dual_vars: dict[str, cp.Variable]
    params: dict[str, cp.Parameter]

    # implemented in protocol
    vars: dict[str, cp.Variable]
    prob: cp.Problem

    def __init__(self):
        self.vars = self.primal_vars | self.dual_vars
        self.constraints.extend(self.dual_constraints)
        obj = self.f_tilde + self.dual_obj

        prob = cp.Problem(cp.Minimize(obj), self.constraints)
        assert prob.is_dpp()
        self.prob = prob

    def task_loss_np(self, y: np.ndarray, is_standardized: bool) -> float:
        ...

    def task_loss_torch(
        self, y: Tensor, is_standardized: bool, solution: Sequence[Tensor]
    ) -> Tensor:
        ...

    def get_cvxpylayer(self) -> CvxpyLayer:
        return CvxpyLayer(
            self.prob,
            parameters=list(self.params.values()),
            variables=list(self.vars.values())
        )


class NonRobustProblemProtocol(BaseProblemProtocol, Protocol):
    def solve(self, y: np.ndarray) -> cp.Problem:
        ...


class BoxProblemProtocol(BaseProblemProtocol, Protocol):
    def solve(self, pred_lo: np.ndarray, pred_hi: np.ndarray) -> cp.Problem:
        ...


class EllipsoidProblemProtocol(BaseProblemProtocol, Protocol):
    def solve(self, loc: np.ndarray, scale_tril: np.ndarray) -> cp.Problem:
        ...


class PICNNProblemProtocol(BaseProblemProtocol, Protocol):
    def solve(self, x: np.ndarray, model: PICNN, q: float) -> cp.Problem:
        ...

    def solve_cvxpylayers(
        self, x: Tensor, model: PICNN, q: Tensor, layer: CvxpyLayer
    ) -> list[Tensor]:
        ...
