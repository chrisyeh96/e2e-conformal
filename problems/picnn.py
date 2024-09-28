"""
This module implements optimization over PICNN sublevel set uncertainty.
"""
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import torch
from torch import nn, Tensor

from models.picnn import PICNN


def einsum_batch_diag(A: Tensor, B: Tensor) -> Tensor:
    """
    Computes A @ batch_diag(B) using einsum.

    Equivalent to
        torch.stack([
            A @ torch.diag(B[i]) for i in range(B.shape[0])
        ])

    Args:
        A: shape (m, n)
        B: shape (batch_size, n)

    Returns:
        result: shape (batch_size, m, n)
    """
    # einsum expression explanation:
    # 'ik,kj->ij' represents the matrix multiplication of A and diag(B_i)
    return torch.einsum('ij,bj->bij', A, B)


class PICNNProblem:
    """
    Args:
        y_dim: dimension of y
        L: # of hidden layers in PICNN
        d: dimension of the hidden layers in PICNN
        y_mean: array of shape [y_dim], entrywise mean of y
        y_std: array of shape [y_dim], entrywise standard deviation of y
        Fz: array of shape [y_dim], cost vector
        epsilon: weight for ‖y‖∞ term in output layer of PICNN
    """
    # instance variables
    L: int
    d: int
    dual_constraints: list[cp.Constraint]
    dual_obj: cp.Expression
    dual_vars: dict[str, cp.Variable]
    params: dict[str, cp.Parameter]

    # subclass must provide these instance variables
    prob: cp.Problem

    def __init__(
        self, y_dim: int, L: int, d: int, y_mean: np.ndarray, y_std: np.ndarray,
        Fz: cp.Expression, epsilon: float = 0.
    ):
        assert epsilon >= 0.
        self.epsilon = epsilon

        self.L = L
        self.d = d
        self.y_mean = y_mean
        self.y_std = y_std

        nu = cp.Variable(2*L*d + 1, nonneg=True)
        nus = [
            nu[d*l: d*(l+1)] for l in range(2*L + 1)
        ]
        mu = nu[-1]
        self.dual_vars = {'nu': nu}

        # there are L+1 Vs, L Ws, L+1 bs, and 1 q
        V_stacked = cp.Parameter((L*d + 1, y_dim))
        W_stacked = cp.Parameter(((L-1)*d + 1, d))
        Ws = [W_stacked[d*l: d*(l+1)] for l in range(L)]
        bs = cp.Parameter((L, d))
        b_fin = cp.Parameter()
        q = cp.Parameter()
        self.params = {
            'V_stacked': V_stacked,
            'W_stacked': W_stacked,
            'bs': bs,
            'b_fin': b_fin,
            'q': q,
        }

        obj = (q - b_fin) * mu + y_mean @ Fz
        for l in range(L):
            obj -= bs[l] @ nus[L + l]
        self.dual_obj = obj

        # constraints
        constr: list[cp.Constraint] = [
            Ws[l].T @ nus[L+l+1] - nus[L+l] - nus[l] == 0
            for l in range(L)
        ]
        c = cp.multiply(y_std, Fz)
        if self.epsilon == 0:
            constr.append(V_stacked.T @ nu[L*d:] == c)
        else:
            rho = cp.Variable(2*y_dim, nonneg=True)
            self.dual_vars['rho'] = rho
            constr.extend([
                V_stacked.T @ nu[L*d:] + rho[:y_dim] - rho[y_dim:] == c,
                mu == cp.sum(rho)
            ])

        self.dual_constraints = constr

    def solve_primal_max(
        self, x: np.ndarray, model: PICNN, q: float, c: np.ndarray | None = None
    ) -> np.ndarray | None:
        """Solve argmax_y (c @ y) subject to PICNN(x, y) <= q.

        Args:
            x: input to the PICNN, shape [x_dim]
            model: PICNN model
            q: upper bound on PICNN(x, y)
            c: optional cost vector, shape [y_dim]. If None, the cost vector
                is sampled from a standard normal distribution.

        Returns:
            argmax_y: optimal y, shape [y_dim], or None if the problem is unbounded
        """
        assert model.L == self.L
        assert model.hidden_dim == self.d
        assert model.epsilon == self.epsilon

        L = self.L
        d = self.d

        if not hasattr(self, 'prob_primal_max'):
            # create self.prob_primal_max here
            c_param = cp.Parameter(model.y_dim, name='c')

            sigma = cp.Variable((L, d), nonneg=True)
            y = cp.Variable(model.y_dim, name='y')

            Vs = [self.params['V_stacked'][d*l: d*(l+1)] for l in range(L + 1)]
            Ws = [self.params['W_stacked'][d*l: d*(l+1)] for l in range(L)]
            bs = self.params['bs']
            b_fin = self.params['b_fin']

            constraints = []
            for l in range(L):
                if l == 0:
                    constr = (sigma[l] >= Vs[0] @ y + bs[l])
                else:
                    constr = (sigma[l] >= Ws[l-1] @ sigma[l-1] + Vs[l] @ y + bs[l])
                constraints.append(constr)

            if self.epsilon == 0:
                constraints.append(Ws[L-1] @ sigma[L-1] + Vs[L] @ y + b_fin <= q)
            else:
                kappa = cp.Variable()
                constraints.extend([
                    Ws[L-1] @ sigma[L-1] + Vs[L] @ y + b_fin + self.epsilon * kappa <= q,
                    kappa >= y,
                    kappa >= -y
                ])

            obj = c_param @ y
            self.prob_primal_max = cp.Problem(cp.Maximize(obj), constraints=constraints)

        self.populate_params(x=x, model=model)
        c_param = self.prob_primal_max.param_dict['c']
        if c is None:
            c = np.random.randn(model.y_dim)
        c_param.value = c

        try:
            self.prob_primal_max.solve(solver=cp.CLARABEL)
        except Exception as e:
            print(e)
            self.prob_primal_max.solve(solver=cp.MOSEK)

        if self.prob_primal_max.status == 'unbounded':
            argmax_y = None
        elif self.prob_primal_max.status == 'infeasible':
            raise Exception('PICNN primal max problem infeasible')
        else:
            y = self.prob_primal_max.var_dict['y']
            argmax_y = y.value

        return argmax_y

    def solve_PICNN_min_q(
        self, x_batch: np.ndarray, model: PICNN, use_existing_params: bool = False
    ) -> np.ndarray:
        """Computes min_y PICNN(x, y) for each x.

        Used to ensure that the sublevel set of the PICNN used for robust
        optimization is non-empty.

        Args:
            x_batch: partial inputs to the PICNN, shape [batch_size, x_dim]
            model: PICNN model

        Returns:
            picnn_min: min_y PICNN(x, y) for each x, shape [batch_size]
        """
        assert model.L == self.L
        assert model.hidden_dim == self.d
        assert model.epsilon == self.epsilon
        assert x_batch.ndim == 2

        if x_batch.shape[0] > 1:
            assert not use_existing_params, "should not use same params for multiple x's"

        L = self.L
        d = self.d

        if not hasattr(self, 'prob_picnn_min'):
            # create self.prob_picnn_min here
            sigma = cp.Variable((L, d), nonneg=True)
            y = cp.Variable(model.y_dim)

            Vs = [self.params['V_stacked'][d*l: d*(l+1)] for l in range(L + 1)]
            Ws = [self.params['W_stacked'][d*l: d*(l+1)] for l in range(L)]
            bs = self.params['bs']
            b_fin = self.params['b_fin']

            constraints = []
            for l in range(L):
                if l == 0:
                    constr = (sigma[l] >= Vs[0] @ y + bs[l])
                else:
                    constr = (sigma[l] >= Ws[l-1] @ sigma[l-1] + Vs[l] @ y + bs[l])
                constraints.append(constr)

            obj = Ws[L-1] @ sigma[L-1] + Vs[L] @ y + b_fin
            if self.epsilon > 0:
                obj += self.epsilon * cp.norm(y, p='inf')
            self.prob_picnn_min = cp.Problem(cp.Minimize(obj), constraints=constraints)

        picnn_min = np.zeros(x_batch.shape[0])
        for i, x in enumerate(x_batch):
            if not use_existing_params:
                self.populate_params(x=x, model=model)
            try:
                self.prob_picnn_min.solve(solver=cp.CLARABEL)
            except Exception as e:
                print(e)
                self.prob_picnn_min.solve(solver=cp.MOSEK)

            # if prob.status != 'optimal':
            #   print('PICNN Minimization Problem status:', prob.status)
            #   raise Exception('PICNN Minimization Problem status:', prob.status)
            if self.prob_picnn_min.status == 'unbounded':
                picnn_min[i] = -np.inf
            elif self.prob_picnn_min.status == 'infeasible':
                raise Exception('PICNN minimization problem infeasible')
            else:
                picnn_min[i] = self.prob_picnn_min.value
        return picnn_min

    def populate_params(self, x: np.ndarray, model: PICNN) -> None:
        L = self.L
        d = self.d
        ReLU = nn.ReLU()

        V_stacked = np.zeros((L*d + 1, model.y_dim))
        Vs = [
            V_stacked[d*l: d*(l+1)] for l in range(L + 1)
        ]
        W_stacked = np.zeros(((L-1)*d + 1, d))
        Ws = [
            W_stacked[d*l: d*(l+1)] for l in range(L)
        ]
        bs = np.zeros((L, d))

        with torch.no_grad():
            u = torch.from_numpy(x).to(torch.float32)
            for l in range(L):
                if l > 0:
                    W_hat_vec = ReLU(model.W_hat_layers[l](u))
                    W = model.W_bar_layers[l].weight @ torch.diag(W_hat_vec)
                    Ws[l-1][:] = W.numpy()
                V_hat_vec = model.V_hat_layers[l](u)
                V = model.V_bar_layers[l].weight @ torch.diag(V_hat_vec)
                b = model.b_layers[l](u)
                u = ReLU(model.u_layers[l](u))
                Vs[l][:] = V.numpy()
                bs[l] = b.numpy()
            l = L
            W_hat_vec = ReLU(model.W_hat_layers[l](u))
            W = model.W_bar_layers[l].weight @ torch.diag(W_hat_vec)
            Ws[l-1][:] = W.numpy()  # shape [1, d]
            V_hat_vec = model.V_hat_layers[l](u)
            V = model.V_bar_layers[l].weight @ torch.diag(V_hat_vec)
            Vs[-1][:] = V.numpy()  # shape [1, y_dim]
            b_fin = model.b_layers[l](u)[0].numpy()

        self.params['V_stacked'].value = V_stacked
        self.params['W_stacked'].value = W_stacked
        self.params['bs'].value = bs
        self.params['b_fin'].value = b_fin

    def solve(self, x: np.ndarray, model: PICNN, q: float) -> cp.Problem:
        assert model.L == self.L
        assert model.hidden_dim == self.d
        assert model.epsilon == self.epsilon

        self.populate_params(x=x, model=model)

        # we add small constant to picnn_min in case numerical issues cause
        # picnn_min to be slightly below the true min_y picnn(x, y)
        picnn_min = self.solve_PICNN_min_q(
            x[None], model, use_existing_params=True)[0] + 1e-5
        q_feas = max(q, picnn_min)
        self.params['q'].value = q_feas

        prob = self.prob
        try:
            prob.solve(solver=cp.CLARABEL)
        except Exception as e:
            print(e)
            prob.solve(solver=cp.MOSEK)

        if prob.status == 'infeasible':
            msg = (f'PICNNProblem.solve() problem status: {prob.status}. '
                   f'q: {q:.3g}, picnn_min: {picnn_min:.3g}')
            print(msg)
        elif prob.status != 'optimal':
            print('PICNNProblem.solve() problem status:', prob.status)
        return prob

    def solve_cvxpylayers(
        self, x: Tensor, model: PICNN, q: Tensor, layer: CvxpyLayer
    ) -> list[Tensor]:
        assert model.L == self.L
        assert model.hidden_dim == self.d
        assert model.epsilon == self.epsilon

        ReLU = nn.ReLU()
        L = model.L
        d = model.hidden_dim
        batch_size = x.shape[0]
        V_stacked = torch.zeros((batch_size, L*d + 1, model.y_dim))
        Vs = [
            V_stacked[:, d*l: d*(l+1)] for l in range(L + 1)
        ]
        W_stacked = torch.zeros((batch_size, (L-1)*d + 1, d))
        Ws = [
            W_stacked[:, d*l: d*(l+1)] for l in range(L)
        ]
        bs = torch.zeros((batch_size, L, d))

        u = x
        for l in range(L):
            if l > 0:
                W_hat_vec = ReLU(model.W_hat_layers[l](u))
                Ws[l-1][:] = einsum_batch_diag(model.W_bar_layers[l].weight, W_hat_vec)
            V_hat_vec = model.V_hat_layers[l](u)
            Vs[l][:] = einsum_batch_diag(model.V_bar_layers[l].weight, V_hat_vec)
            bs[:, l] = model.b_layers[l](u)
            u = ReLU(model.u_layers[l](u))
        l = L
        W_hat_vec = ReLU(model.W_hat_layers[l](u))
        Ws[l-1][:] = einsum_batch_diag(model.W_bar_layers[l].weight, W_hat_vec)
        V_hat_vec = model.V_hat_layers[l](u)
        Vs[l][:] = einsum_batch_diag(model.V_bar_layers[l].weight, V_hat_vec)
        b_fin = model.b_layers[l](u)[:, 0]

        # we add small constant to picnn_min in case numerical issues cause
        # picnn_min to be slightly below the true min_y picnn(x, y)
        picnn_min = self.solve_PICNN_min_q(x.detach().numpy(), model) + 1e-5
        q_feas = torch.maximum(q, torch.from_numpy(picnn_min).to(dtype=q.dtype))

        return layer(V_stacked, W_stacked, bs, b_fin, q_feas)
