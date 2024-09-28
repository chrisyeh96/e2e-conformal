from collections.abc import Mapping, Sequence
import io
from typing import Any

import numpy as np
import torch.utils.data
from tqdm.auto import tqdm

from problems.protocols import BoxProblemProtocol
from .quantile import (
    PinballLoss, QuantileRegressor, calc_q, conformal_q, multidim_quantile_score)
from .quantile_eto import optimize


def train_e2e_epoch(
    model: QuantileRegressor,
    prob: BoxProblemProtocol,
    loader: torch.utils.data.DataLoader,
    y_dim: int,
    alpha: float,
    y_info: str | tuple[np.ndarray, np.ndarray],
    quantile_loss_frac: float,
    rng: np.random.Generator,
    optimizer: torch.optim.Optimizer,
    show_pbar: bool = False
) -> tuple[float, float, float]:
    """Trains decision-aware quantile regression model for 1 epoch.

    Args:
        model: model to train
        prob: robust optimization problem to solve (with box constraints)
        loader: data loader
        y_dim: dimension of each label y
        alpha: desired coverage level
        y_info: either string 'log' or a tuple (mean, std) of the label y
        quantile_loss_frac: weight for the quantile loss in [0, 1],
            (1-quantile_loss_frac) is the weight for the task loss
        rng: random number generator
        optimizer: optimizer to use for training
        show_pbar: whether to show a progress bar

    Returns:
        avg_loss: total loss averaged over training examples
        avg_pinball_loss: total pinball loss averaged over training examples
        avg_task_loss: total task loss averaged over training examples for which
            the task loss was computed
    """
    assert 0 < alpha < 1
    model.train()

    cvxpylayer = prob.get_cvxpylayer()

    lo_quantile_loss_fn = PinballLoss(alpha/2)
    hi_quantile_loss_fn = PinballLoss(1-alpha/2)

    total_loss = 0.
    total_pinball_loss = 0.
    total_task_loss = 0.
    total_num_tasks = 0
    for x, y in tqdm(loader) if show_pbar else loader:
        batch_size = x.shape[0]
        pred = model(x)
        pred_lo = pred[..., :y_dim]
        pred_hi = pred[..., y_dim:]

        # compute quantile loss if desired
        loss = torch.tensor(0.)
        if quantile_loss_frac > 0:
            lo_quantile_loss = lo_quantile_loss_fn(pred_lo, y)
            hi_quantile_loss = hi_quantile_loss_fn(pred_hi, y)
            pinball_loss = lo_quantile_loss + hi_quantile_loss
            loss += quantile_loss_frac * pinball_loss
            total_pinball_loss += pinball_loss.item() * batch_size

        # compute task loss if desired
        if quantile_loss_frac < 1:
            perm = rng.permutation(batch_size)

            # calculate q using half of the batch
            cal_inds = perm[:batch_size//2]
            scores_cal = multidim_quantile_score(pred[cal_inds], y[cal_inds], y_dim)
            q = calc_q(scores_cal, alpha)
            if q == torch.inf:
                tqdm.write('Batch is too small, skipping')
                continue

            # predict on the other half of the batch and compute task loss
            task_inds = perm[batch_size//2:]
            y_task = y[task_inds]
            pred_lo_calib = pred_lo[task_inds] - q  # shape [batch_size/2, y_dim]
            pred_hi_calib = pred_hi[task_inds] + q

            if y_info == 'log':
                y_task = torch.exp(y_task)
                pred_lo_calib = torch.exp(pred_lo_calib)
                pred_hi_calib = torch.exp(pred_hi_calib)
                is_standardized = False
            else:
                is_standardized = True

            try:
                # solution is a tuple of Tensors, each with shape [batch_size/2, var_dim]
                solution = cvxpylayer(pred_lo_calib, pred_hi_calib)
            except Exception as e:
                print(e)
                if (pred_hi_calib < pred_lo_calib).any():
                    print('ERROR: pred_hi < pred_lo')
                for i in range(len(task_inds)):
                    try:
                        res = prob.solve(pred_hi_calib[i].detach().numpy(), pred_lo_calib[i].detach().numpy())
                        # solution = cvxpylayer(pred_lo_calib[i], pred_hi_calib[i])
                        if res.status != 'optimal':
                            print(res.status)
                    except Exception as e:
                        print(e)
                        import pdb
                        pdb.set_trace()

            task_loss = prob.task_loss_torch(y_task, is_standardized=is_standardized, solution=solution).mean()
            loss += (1 - quantile_loss_frac) * task_loss

            total_task_loss += task_loss.item() * task_inds.shape[0]
            total_num_tasks += task_inds.shape[0]

        if show_pbar:
            tqdm.write(f'pinball_loss: {pinball_loss.item()}, task_loss: {task_loss.item()}')
        total_loss += loss.item() * batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(loader.dataset)
    avg_pinball_loss = total_pinball_loss / len(loader.dataset)
    avg_task_loss = total_task_loss / total_num_tasks
    return avg_loss, avg_pinball_loss, avg_task_loss


def train_e2e(
    loaders: Mapping[str, torch.utils.data.DataLoader],
    input_dim: int,
    y_dim: int,
    alpha: float,
    max_epochs: int,
    lr: float,
    l2reg: float,
    y_info: str | tuple[np.ndarray, np.ndarray],
    prob: BoxProblemProtocol,
    rng: np.random.Generator,
    quantile_loss_frac: float | Sequence[float],
    saved_model_path: str = ''
) -> tuple[QuantileRegressor, dict[str, Any]]:
    """Trains a quantile regression model end-to-end (e2e) with conformal calibration.

    Uses early-stopping based on task loss on calibration set.
    Everything in this function is done on CPU.

    Returns:
        model: trained model, on CPU
        result: dict of training results
    """
    # Initialize model
    model = QuantileRegressor(input_dim=input_dim, y_dim=y_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2reg)
    if saved_model_path is not None:
        model.load_state_dict(torch.load(saved_model_path, weights_only=True))

    # Train model
    result: dict[str, Any] = {
        'train_e2e_losses': [],
        'train_pinball_losses': [],
        'train_task_losses': [],
        'val_task_losses': [],
        'best_epoch': 0,
        'val_task_loss': np.inf,  # best loss on val set
    }
    steps_since_decrease = 0
    buffer = io.BytesIO()

    pbar = tqdm(range(max_epochs))
    for epoch in pbar:
        if isinstance(quantile_loss_frac, Sequence):
            weight = quantile_loss_frac[min(epoch, len(quantile_loss_frac) - 1)]
        else:
            weight = quantile_loss_frac

        train_e2e_loss, train_pinball_loss, train_task_loss = train_e2e_epoch(
            model, prob=prob, loader=loaders['train'], y_dim=y_dim, alpha=alpha,
            y_info=y_info, quantile_loss_frac=weight, rng=rng,
            optimizer=optimizer)
        result['train_e2e_losses'].append(train_e2e_loss)
        result['train_pinball_losses'].append(train_pinball_loss)
        result['train_task_losses'].append(train_task_loss)

        # here, we use the calibration set to select its own q
        # (not ideal, but our datasets are small enough that we can't afford to
        # split it further)
        with torch.no_grad():
            model.eval()
            q = conformal_q(model, loaders['calib'], y_dim=y_dim, alpha=alpha).item()
            val_task_loss = optimize(
                model, prob=prob, loader=loaders['calib'], y_dim=y_dim, q=q, y_info=y_info)
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
    return model.cpu(), result
