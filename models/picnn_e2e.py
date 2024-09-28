from collections.abc import Mapping
import io
from typing import Any

import cvxpy.error
import diffcp.cone_program
import numpy as np
import torch.utils.data
from tqdm.auto import tqdm

from models.picnn import PICNN, conformal_q
from problems.protocols import PICNNProblemProtocol
from utils.conformal import calc_q


def optimize(
    model: PICNN,
    prob: PICNNProblemProtocol,
    loader: torch.utils.data.DataLoader,
    q: float,
    show_pbar: bool = False
) -> tuple[float, float]:
    """
    Args:
        model: model to evaluate
        prob: robust optimization problem to solve (with box constraints)
        loader: data loader
        q: score threshold for the desired coverage level
        show_pbar: whether to show a progress bar

    Returns:
        task_loss: average task loss per training example
        coverage: proportion of examples whose scores are ≤ q

    Raises:
        cvxpy.error.SolverError: if optimization problem is infeasible
    """
    model.eval()

    all_scores = []
    task_losses = []
    if show_pbar:
        pbar = tqdm(total=len(loader.dataset))

    with torch.no_grad():
        for x_batch, y_batch in loader:
            scores = model(x_batch, y_batch)
            all_scores.append(scores)

            for x, y in zip(x_batch.numpy(), y_batch.numpy()):
                # model.clamp_weights()
                # print(f's(x,y): {scores[i].item()}, q={q:.3f}')
                # prob.solve_primal_max(x, model, q)
                prob.solve(x, model, q)
                task_loss = prob.task_loss_np(y, is_standardized=True)
                task_losses.append(task_loss)
                if show_pbar:
                    pbar.update(1)

    scores = torch.cat(all_scores)
    coverage = (scores <= q).to(torch.float64).mean().item()
    avg_task_loss = np.mean(task_losses).item()
    return avg_task_loss, coverage


def optimize_wrapper(
    model: PICNN,
    prob: PICNNProblemProtocol,
    loaders: Mapping[str, torch.utils.data.DataLoader],
    alpha: float,
    show_pbar: bool = False
) -> dict[str, float]:
    """Runs the "optimize" part of ETO on both train and test splits.

    Returns:
        result: dictionary of results
    """
    model.eval()
    with torch.no_grad():
        q = conformal_q(model, loaders['calib'], alpha=alpha).item()
        result = {'q': q}

        for split in ('train', 'test'):
            task_loss, coverage = optimize(
                model, prob=prob, loader=loaders[split], q=q, show_pbar=show_pbar)
            result |= {
                f'{split}_coverage': coverage,
                f'{split}_task_loss': task_loss,
            }
    return result


def train_e2e_epoch(
    model: PICNN,
    prob: PICNNProblemProtocol,
    loader: torch.utils.data.DataLoader,
    alpha: float,
    rng: np.random.Generator,
    optimizer: torch.optim.Optimizer,
    large_q_penalty: float = 1e-2,
    show_pbar: bool = False
) -> float:
    """Trains decision-aware quantile regression model for 1 epoch.

    Args:
        model: model to train
        prob: robust optimization problem to solve (with box constraints)
        loader: data loader
        alpha: desired coverage level
        rng: random number generator
        optimizer: optimizer to use for training
        large_q_penalty: weight for penalizing large q values
        show_pbar: whether to show a progress bar

    Returns:
        avg_task_loss: total task loss averaged over training examples for which
            the task loss was computed

    Raises:
        diffcp.cone_program.SolverError if e2e optimization problem is infeasible
    """
    assert 0 < alpha < 1
    model.train()
    cvxpylayer = prob.get_cvxpylayer()

    total_task_loss = 0.
    total_num_tasks = 0
    for x, y in tqdm(loader) if show_pbar else loader:
        batch_size = x.shape[0]

        # compute task loss if desired
        perm = rng.permutation(batch_size)

        # calculate q using half of the batch
        cal_inds = perm[:batch_size//2]
        scores_cal = model(x[cal_inds], y[cal_inds])  # standardized y
        q = calc_q(scores_cal, alpha)
        if q == torch.inf:
            tqdm.write('Batch is too small, skipping')
            continue

        # predict on the other half of the batch and compute task loss
        task_inds = perm[batch_size//2:]
        y_task = y[task_inds]
        solution = prob.solve_cvxpylayers(x[task_inds], model, q, cvxpylayer)

        task_loss = prob.task_loss_torch(
            y_task, is_standardized=True, solution=solution).mean()
        loss = task_loss + large_q_penalty * q**2  # penalize large q
        total_task_loss += task_loss.item() * task_inds.shape[0]
        total_num_tasks += task_inds.shape[0]

        if show_pbar:
            tqdm.write(f'task_loss: {task_loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.clamp_weights()

    avg_task_loss = total_task_loss / total_num_tasks
    return avg_task_loss


def train_e2e(
    model: PICNN,
    loaders: Mapping[str, torch.utils.data.DataLoader],
    alpha: float,
    max_epochs: int,
    lr: float,
    l2reg: float,
    prob: PICNNProblemProtocol,
    rng: np.random.Generator,
    large_q_penalty: float = 1e-2,
    saved_model_path: str = ''
) -> tuple[PICNN, dict[str, Any]]:
    """Trains a PICNN model end-to-end (e2e) with conformal calibration.

    Uses early-stopping based on task loss on calibration set.

    Returns:
        model: trained model, on CPU
        result: dict of training results
    """
    # Initialize model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2reg)
    if saved_model_path != '':
        model.load_state_dict(torch.load(saved_model_path, weights_only=True))
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)

    # Train model
    result: dict[str, Any] = {
        'train_task_losses': [],
        'val_task_losses': [],
        'best_epoch': 0,
        'val_task_loss': np.inf,  # best loss on val set
    }
    steps_since_decrease = 0

    pbar = tqdm(range(max_epochs))
    try:
        for epoch in pbar:
            train_task_loss = train_e2e_epoch(
                model, prob=prob, loader=loaders['train'], alpha=alpha,
                rng=rng, optimizer=optimizer, large_q_penalty=large_q_penalty)
            result['train_task_losses'].append(train_task_loss)

            # here, we use the calibration set to select its own q
            # (not ideal, but our datasets are small enough that we can't afford to
            # split it further)
            with torch.no_grad():
                model.eval()
                q = conformal_q(model, loaders['calib'], alpha=alpha).item()
                val_task_loss, _ = optimize(
                    model, prob=prob, loader=loaders['calib'], q=q)
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
    except (diffcp.cone_program.SolverError, cvxpy.error.SolverError) as e:
        print(f'Error: {e}')

    buffer.seek(0)
    model.load_state_dict(torch.load(buffer, weights_only=True))
    return model.cpu(), result
