from collections.abc import Mapping
import io
import itertools
from typing import Any, Literal

import numpy as np
import scipy.stats
import torch.utils.data
from torch import distributions as tdist, nn, Tensor
import torch.nn.functional as F
from tqdm.auto import tqdm

from utils.conformal import calc_q

MIN_VALUE = 1e-8
LossName = Literal['mse', 'msenll', 'nll', 'nlldiag']


class GaussianRegressor(nn.Module):
    """
    For each input example x of shape [input_dim], the model outputs a vector
    pred of shape [y_dim + y_dim*(y_dim+1)/2].
    - The first y_dim elements of pred are the predicted mean.
    - The remaining y_dim*(y_dim+1)/2 elements are a Cholesky factor for the
        covariance matrix of a Gaussian distribution. A Cholesky factor is a
        lower-triangular matrix.
      - Of these elements, the first y_dim elements are the diagonal of the Cholesky.
        We enforce strict positive definiteness of the covariance matrix by passing
        these diagonal elements through a softplus function.
      - The remaining y_dim*(y_dim-1)/2 elements are the off-diagonal elements of the
        Cholesky factor. These are unconstrained.
    """
    def __init__(self, input_dim: int, y_dim: int):
        super().__init__()
        self.y_dim = y_dim
        ReLU = nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            ReLU,
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            ReLU,
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            ReLU,
            nn.Linear(256, y_dim + y_dim * (y_dim + 1) // 2)
        )
        # indices below diag
        self.tril_inds_r, self.tril_inds_c = torch.tril_indices(y_dim, y_dim, offset=-1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: shape [n, input_dim]

        Returns:
            loc: shape [n, y_dim], predicted means
            scale_tril: shape [n, y_dim, y_dim], lower-triangular factor of covariance,
                with positive-valued diagonal
        """
        pred = self.net(x)  # shape [n, y_dim + y_dim*(y_dim+1)/2]
        loc = pred[:, :self.y_dim]

        # populate the diagonal
        diag = F.softplus(pred[:, self.y_dim:2*self.y_dim]) + MIN_VALUE
        scale_tril = torch.diag_embed(diag)

        # populate the lower triangle (excluding the diagonal)
        scale_tril[:, self.tril_inds_r, self.tril_inds_c] = pred[:, 2*self.y_dim:]
        return loc, scale_tril


class GaussianRegressorSplit(GaussianRegressor):
    """
    This class breaks up the monolithic GaussianRegressor final layer
    into separate heads for the mean, scale_tril diagonal, and scale_tril
    off-diagonal.

    Args:
        input_dim: dimension of each input example x
        y_dim: dimension of each label y
        zero_init_scale_tril: if True, initialize the scale_tril head's
            weights and bias to zero. This seems to be crucial for getting good
            NLL loss.
        diag_cov: if True, the covariance matrix is diagonal
    """
    def __init__(self, input_dim: int, y_dim: int, zero_init_scale_tril: bool = True,
                 diag_cov: bool = False):
        nn.Module.__init__(self)
        self.y_dim = y_dim
        self.diag_cov = diag_cov
        # act = nn.LeakyReLU()
        act = nn.ReLU()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            act,
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            act,
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            act
        )
        self.loc_net = nn.Linear(256, y_dim)
        self.diag_net = nn.Linear(256, y_dim)
        self.scale_tril_net = nn.Linear(256, y_dim * (y_dim - 1) // 2)

        if zero_init_scale_tril:
            self.zero_scale_tril_net()

        # indices below diag
        self.tril_inds_r, self.tril_inds_c = torch.tril_indices(y_dim, y_dim, offset=-1)

    def initialize_diag_bias(self, bias: float) -> None:
        with torch.no_grad():
            self.diag_net.bias.fill_(bias)

    def zero_scale_tril_net(self) -> None:
        with torch.no_grad():
            self.scale_tril_net.weight.fill_(0.)
            self.scale_tril_net.bias.fill_(0.)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: shape [..., input_dim]

        Returns:
            loc: shape [..., y_dim], predicted means
            scale_tril: shape [..., y_dim, y_dim], lower-triangular factor of covariance,
                with positive-valued diagonal
        """
        emb = self.embed(x)

        # predict mean
        loc = self.loc_net(emb)

        # populate the strictly positive diagonal
        diag = F.softplus(self.diag_net(emb)) + MIN_VALUE
        scale_tril = torch.diag_embed(diag)

        if not self.diag_cov:
            # populate the lower triangle (excluding the diagonal)
            scale_tril[:, self.tril_inds_r, self.tril_inds_c] = self.scale_tril_net(emb)
        return loc, scale_tril


def mahalanobis_dist2(loc: Tensor, scale_tril: Tensor, y: Tensor) -> Tensor:
    """Computes squared Mahalanobis distance
        (y-loc)^T @ inv(S) @ (y-loc)
    where S = scale_tril @ scale_tril^T for batches of loc, scale_tril, y.

    Args:
        mean: shape [n, y_dim], predicted means
        scale_tril: shape [n, y_dim, y_dim], lower-triangular factor of covariance,
            with positive-valued diagonal
        y: shape [n, y_dim], true labels

    Returns:
        dists: shape [n], squared Mahalanobis distance for each example
    """
    return tdist.multivariate_normal._batch_mahalanobis(bL=scale_tril, bx=y - loc)


def conformal_q(
    model: GaussianRegressor,
    loader: torch.utils.data.DataLoader,
    alpha: float,
    device: str = 'cpu'
) -> Tensor:
    """
    Args:
        model: model to evaluate
        loader: data loader
        alpha: desired coverage level in (0, 1)
        device: device to use

    Returns:
        q: scalar score threshold for the desired coverage level
    """
    model.to(device)
    all_scores = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        loc, scale_tril = model(x)
        scores = mahalanobis_dist2(loc, scale_tril, y)
        all_scores.append(scores)

    scores = torch.cat(all_scores)
    q = calc_q(scores, alpha)
    assert q != torch.inf, 'Size of calibration set is too small'
    return q


def run_epoch_gaussian_regressor(
    model: GaussianRegressor,
    loader: torch.utils.data.DataLoader,
    y_dim: int,
    loss_name: LossName = 'nll',
    alpha: float | None = None,
    q: Tensor | float | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = 'cpu'
) -> tuple[float, float | None]:
    """Runs 1 epoch of training or evaluation with a Gaussian regression model
    and negative log-likelihood loss.

    Args:
        model: model to train
        loader: data loader
        y_dim: dimension of each label
        loss_name: either 'mse', 'nll', 'nllmse', 'nlldiag'
        alpha: optional risk level for calculating coverage
        q: optional score threshold for the desired coverage level,
            q=None means not doing conformal
        optimizer: optimizer to use for training, None for evaluation
        device: either 'cpu' or 'cuda'

    Returns:
        avg_loss: average loss per training example
        coverage: proportion of targets that fall within predicted quantile range
    """
    if optimizer is None:
        model.eval().to(device)
    else:
        model.train().to(device)

    if alpha is not None:
        assert (0 < alpha < 1) and (q is None)
        q = scipy.stats.chi2.ppf(q=1-alpha, df=y_dim).item()

    total_loss = 0.
    num_covered = 0.
    num_total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        loc, scale_tril = model(x)
        if loss_name == 'nll':
            loss = -tdist.MultivariateNormal(loc, scale_tril=scale_tril).log_prob(y).mean()
        elif loss_name == 'mse':
            loss = F.mse_loss(loc, y)
        elif loss_name == 'msenll':
            mse_loss = F.mse_loss(loc, y)
            nll_loss = -tdist.MultivariateNormal(loc, scale_tril=scale_tril).log_prob(y).mean()
            loss = mse_loss + nll_loss
        elif loss_name == 'nlldiag':
            std = scale_tril.diagonal(dim1=-2, dim2=-1)
            loss = -tdist.Normal(loc=loc, scale=std).log_prob(y).sum(dim=1).mean()
        total_loss += loss.item() * x.shape[0]
        num_total += x.shape[0]

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if q is not None:
            with torch.no_grad():
                covered = (mahalanobis_dist2(loc, scale_tril, y) <= q)  # shape [batch]
                num_covered += covered.sum().item()

    if q is None:
        coverage = None
    else:
        coverage = num_covered / num_total

    avg_loss = total_loss / num_total
    return avg_loss, coverage


def train_gaussian_regressor(
    loaders: Mapping[str, torch.utils.data.DataLoader],
    input_dim: int,
    y_dim: int,
    max_epochs: int,
    lr: float,
    l2reg: float,
    show_pbar: bool = False,
    return_best_model: bool = False,
    device: str = 'cpu'
) -> tuple[GaussianRegressor, dict[str, Any]]:
    """Trains a Gaussian regression model.

    Uses early-stopping based on negative log-likelihood loss on the calibration set.

    Args:
        loaders: dictionary of data loaders
        input_dim: dimension of each input example x
        y_dim: dimension of each label y
        max_epochs: maximum number of epochs to train
        lr: learning rate
        l2reg: L2 regularization strength
        show_pbar: if True, show a progress bar
        return_best_model: if True, return the model with the best validation loss,
            otherwise returns the model from the last training epoch
        device: either 'cpu' or 'cuda'

    Returns:
        model: trained model on CPU, either the model from the last training epoch, or
            the model that achieves result['val_nll_loss'] on the calibration set,
            see return_best_model
        result: dict of performance metrics
    """
    # Initialize model
    model = GaussianRegressor(input_dim=input_dim, y_dim=y_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2reg)

    result: dict[str, Any] = {
        'train_nll_losses': [],
        'val_nll_losses': [],
        'best_epoch': 0,
        'val_nll_loss': np.inf,  # best loss on val set
    }
    steps_since_decrease = 0
    buffer = io.BytesIO()

    # Train model
    pbar = tqdm(range(max_epochs)) if show_pbar else range(max_epochs)
    for epoch in pbar:
        # train
        train_loss, _ = run_epoch_gaussian_regressor(
            model, loaders['train'], y_dim=y_dim, optimizer=optimizer, device=device)
        result['train_nll_losses'].append(train_loss)

        # calculate loss on calibration set
        with torch.no_grad():
            val_loss, _ = run_epoch_gaussian_regressor(
                model, loaders['calib'], y_dim=y_dim, device=device)
            result['val_nll_losses'].append(val_loss)

        if show_pbar:
            msg = f'Train loss: {train_loss:.3g}, Calib loss: {val_loss:.3g}'
            pbar.set_description(msg)

        steps_since_decrease += 1

        if val_loss < result['val_nll_loss']:
            result['best_epoch'] = epoch
            result['val_nll_loss'] = val_loss
            steps_since_decrease = 0
            if return_best_model:
                buffer.seek(0)
                torch.save(model.state_dict(), buffer)

        if steps_since_decrease > 10:
            break

    # load best model
    if return_best_model:
        buffer.seek(0)
        model.load_state_dict(torch.load(buffer, weights_only=True))

    return model.cpu(), result


def train_gaussian_regressor_custom(
    model: GaussianRegressorSplit,
    loaders: Mapping[str, torch.utils.data.DataLoader],
    loss_name: LossName,
    max_epochs: int,
    lr: float,
    l2reg: float,
    freeze: tuple[Literal['embed', 'loc_net', 'diag_net', 'scale_tril_net'], ...] = (),
    cutoff: int = 20,
    show_pbar: bool = False,
    return_best_model: bool = False,
    device: str = 'cpu'
) -> dict[str, Any]:
    """Trains a Gaussian regression model using specified loss function.

    Uses early-stopping based on loss on the calibration set.

    Args:
        model: model to train
        loaders: maps split to dataloader
        loss_name: either 'mse', 'msenll', 'nll', 'nlldiag'
        max_epochs: maximum number of epochs to train
        lr: learning rate
        l2reg: L2 regularization strength
        freeze: names of parts of model to freeze during training
        cutoff: number of epochs without improvement to stop training
        show_pbar: if True, show a progress bar
        return_best_model: if True, return the model with the best validation loss,
            otherwise returns the model from the last training epoch
        device: either 'cpu' or 'cuda'

    Returns:
        result: dict of performance metrics
    """
    model.train()
    model.to(device)

    all_params_dict = {
        'embed': model.embed.parameters(),
        'loc_net': model.loc_net.parameters(),
        'diag_net': model.diag_net.parameters(),
        'scale_tril_net': model.scale_tril_net.parameters()
    }

    possible_params: tuple[str, ...]
    if loss_name == 'mse':
        possible_params = ('embed', 'loc_net')
    elif loss_name == 'nlldiag':
        possible_params = ('embed', 'loc_net', 'diag_net')
    elif loss_name == 'nll' or loss_name == 'msenll':
        possible_params = ('embed', 'loc_net', 'diag_net', 'scale_tril_net')

    param_names = set(possible_params) - set(freeze)
    params = itertools.chain(*(all_params_dict[name] for name in param_names))
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=l2reg)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=15)
    current_lr = lr

    result: dict[str, Any] = {
        'train_losses': [],
        'val_losses': [],
        'best_epoch': 0,
        'val_loss': np.inf,  # best loss on val set
    }
    steps_since_decrease = 0
    buffer = io.BytesIO()

    pbar = tqdm(total=max_epochs) if show_pbar else None
    for epoch in range(max_epochs):
        # train
        train_loss, _ = run_epoch_gaussian_regressor(
            model, loaders['train'], y_dim=model.y_dim, loss_name=loss_name,
            optimizer=optimizer, device=device)
        result['train_losses'].append(train_loss)

        # calculate loss on calibration set
        with torch.no_grad():
            val_loss, _ = run_epoch_gaussian_regressor(
                model, loaders['calib'], y_dim=model.y_dim, loss_name=loss_name,
                device=device)
            result['val_losses'].append(val_loss)

        if pbar is not None:
            msg = (f'Train {loss_name} loss: {train_loss:.3g}, '
                   f'Calib {loss_name} loss: {val_loss:.3g}')
            pbar.set_description(msg)
            pbar.update()

        steps_since_decrease += 1

        if val_loss < result['val_loss']:
            result['best_epoch'] = epoch
            result['val_loss'] = val_loss
            steps_since_decrease = 0
            buffer.seek(0)
            torch.save(model.state_dict(), buffer)

        if steps_since_decrease > cutoff:
            break

        lr_scheduler.step(val_loss)
        if lr_scheduler.get_last_lr()[0] < current_lr:
            current_lr = lr_scheduler.get_last_lr()[0]
            tqdm.write(f'Finished epoch {epoch}, reducing lr to {current_lr}')

    # load best model
    if return_best_model:
        buffer.seek(0)
        model.load_state_dict(torch.load(buffer, weights_only=True))

    return result
