from collections.abc import Mapping, Sequence
import io
from typing import Any

import numpy as np
import torch.distributions as tdist
import torch.utils.data
from tqdm.auto import tqdm

from problems.protocols import EllipsoidProblemProtocol
from .gaussian import (
    GaussianRegressor, calc_q, conformal_q, mahalanobis_dist2)
from .gaussian_eto import optimize


def train_e2e_epoch(
    model: GaussianRegressor,
    prob: EllipsoidProblemProtocol,
    loader: torch.utils.data.DataLoader,
    alpha: float,
    nll_loss_frac: float,
    rng: np.random.Generator,
    optimizer: torch.optim.Optimizer,
    show_pbar: bool = False
) -> tuple[float, float, float]:
    """Trains decision-aware quantile regression model for 1 epoch.

    Args:
        model: model to train
        prob: robust optimization problem to solve (with box constraints)
        loader: data loader
        alpha: desired coverage level
        y_info: either string 'log' or a tuple (mean, std) of the label y
        nll_loss_frac: weight for the quantile loss in [0, 1],
            (1-nll_loss_frac) is the weight for the task loss
        rng: random number generator
        optimizer: optimizer to use for training
        show_pbar: whether to show a progress bar

    Returns:
        avg_loss: total loss averaged over training examples
        avg_nll_loss: total nll loss averaged over training examples
        avg_task_loss: total task loss averaged over training examples for which
            the task loss was computed
    """
    assert 0 < alpha < 1
    model.train()

    cvxpylayer = prob.get_cvxpylayer()

    total_loss = 0.
    total_nll_loss = 0.
    total_task_loss = 0.
    total_num_tasks = 0
    for x, y in tqdm(loader) if show_pbar else loader:
        batch_size = x.shape[0]
        loc, scale_tril = model(x)

        loss = torch.tensor(0.)
        msgs = []

        # compute nll loss if desired
        if nll_loss_frac > 0:
            nll_loss = -tdist.MultivariateNormal(loc, scale_tril=scale_tril).log_prob(y).mean()
            loss += nll_loss_frac * nll_loss
            total_nll_loss += nll_loss.item() * batch_size
            msgs.append(f'nll_loss: {nll_loss.item()}')

        # compute task loss if desired
        if nll_loss_frac < 1:
            perm = rng.permutation(batch_size)

            # calculate q using half of the batch
            cal_inds = perm[:batch_size//2]
            scores_cal = mahalanobis_dist2(loc[cal_inds], scale_tril[cal_inds], y[cal_inds])
            q = calc_q(scores_cal, alpha)
            if q == torch.inf:
                tqdm.write('Batch is too small, skipping')
                continue

            # predict on the other half of the batch and compute task loss
            task_inds = perm[batch_size//2:]
            y_task = y[task_inds]
            loc_task = loc[task_inds]
            scale_tril_task = scale_tril[task_inds] * torch.sqrt(q)

            try:
                # solution is a tuple of Tensors, each with shape [batch_size/2, var_dim]
                solution = cvxpylayer(loc_task, scale_tril_task)
            except Exception as e:
                print(e)
                raise e
                # import pdb
                # pdb.set_trace()

            task_loss = prob.task_loss_torch(y_task, is_standardized=True, solution=solution).mean()
            loss += (1 - nll_loss_frac) * task_loss

            total_task_loss += task_loss.item() * task_inds.shape[0]
            total_num_tasks += task_inds.shape[0]

            msgs.append(f'task_loss: {task_loss.item()}')

        if show_pbar:
            tqdm.write(','.join(msgs))
        total_loss += loss.item() * batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(loader.dataset)
    avg_nll_loss = total_nll_loss / len(loader.dataset)
    avg_task_loss = total_task_loss / total_num_tasks
    return avg_loss, avg_nll_loss, avg_task_loss


def train_e2e(
    model: GaussianRegressor,
    loaders: Mapping[str, torch.utils.data.DataLoader],
    alpha: float,
    max_epochs: int,
    lr: float,
    l2reg: float,
    prob: EllipsoidProblemProtocol,
    rng: np.random.Generator,
    nll_loss_frac: float | Sequence[float],
    saved_model_path: str = ''
) -> dict[str, Any]:
    """Trains a quantile regression model end-to-end (e2e) with conformal calibration.

    Uses early-stopping based on task loss on calibration set.
    Everything in this function is done on CPU.

    Args:
        model: model to train, on CPU

    Returns:
        result: dict of training results
    """
    # Initialize model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2reg)
    if saved_model_path != '':
        tqdm.write(f'Loading saved model: {saved_model_path}')
        model.load_state_dict(torch.load(saved_model_path, weights_only=True))

    # Train model
    result: dict[str, Any] = {
        'train_e2e_losses': [],
        'train_nll_losses': [],
        'train_task_losses': [],
        'val_task_losses': [],
        'best_epoch': 0,
        'val_task_loss': np.inf,  # best loss on val set
    }
    steps_since_decrease = 0
    buffer = io.BytesIO()

    pbar = tqdm(range(max_epochs))
    for epoch in pbar:
        if isinstance(nll_loss_frac, Sequence):
            weight = nll_loss_frac[min(epoch, len(nll_loss_frac) - 1)]
        else:
            weight = nll_loss_frac

        train_e2e_loss, train_nll_loss, train_task_loss = train_e2e_epoch(
            model, prob=prob, loader=loaders['train'], alpha=alpha,
            nll_loss_frac=weight, rng=rng, optimizer=optimizer)
        result['train_e2e_losses'].append(train_e2e_loss)
        result['train_nll_losses'].append(train_nll_loss)
        result['train_task_losses'].append(train_task_loss)

        # here, we use the calibration set to select its own q
        # (not ideal, but our datasets are small enough that we can't afford to
        # split it further)
        with torch.no_grad():
            model.eval()
            q = conformal_q(model, loaders['calib'], alpha=alpha).item()
            val_task_loss = optimize(model, prob=prob, loader=loaders['calib'], q=q)
            result['val_task_losses'].append(val_task_loss)

        msg = (f'Epoch {epoch}, train_task_loss {train_task_loss:.3f}, '
               f'val_task_loss {val_task_loss:.3f}, q {q:.3f}')
        pbar.set_description(msg)

        steps_since_decrease += 1

        if val_task_loss < result['val_task_loss']:
            result['best_epoch'] = epoch
            result['val_task_loss'] = val_task_loss
            steps_since_decrease = 0
            buffer.seek(0)
            torch.save(model.state_dict(), buffer)

        if steps_since_decrease > 10:
            break

    buffer.seek(0)
    model.load_state_dict(torch.load(buffer, weights_only=True))
    return result
