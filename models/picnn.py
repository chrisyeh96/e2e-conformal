import torch.utils.data
from torch import nn, Tensor
from tqdm.auto import tqdm

from utils.conformal import calc_q


class PICNN(nn.Module):
    """
    Partially input-convex neural network mapping from input example
    x of shape [input_dim] and y of shape [y_dim] to a scalar score output
    s(x, y)

    Args:
        input_dim: dimension of the input x
        y_dim: dimension of the input y
        hidden_dim: dimension of the hidden layers (i.e., d)
        n_layers: # of hidden layers (i.e., L)
        y_in_output_layer: if False, sets V_L = 0
        epsilon: weight for ‖y‖∞ term in output layer
    """
    def __init__(self, input_dim: int, y_dim: int, hidden_dim: int, n_layers: int,
                 y_in_output_layer: bool = True, epsilon: float = 0.):
        super().__init__()
        self.input_dim = input_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        L = n_layers
        self.L = L

        # bag of tricks for feasibility
        self.epsilon = epsilon
        self.y_in_output_layer = y_in_output_layer

        null_module = nn.Module()

        self.W_hat_layers = nn.ModuleList(
            [null_module]
            + [nn.Linear(hidden_dim, hidden_dim) for _ in range(1, L+1)]
        )
        self.W_bar_layers = nn.ModuleList(
            [null_module]
            + [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(1, L)]
            + [nn.Linear(hidden_dim, 1, bias=False)]
        )
        self.V_hat_layers = nn.ModuleList(
            [nn.Linear(input_dim, y_dim)]
            + [nn.Linear(hidden_dim, y_dim) for _ in range(1, L+1)]
        )
        self.V_bar_layers = nn.ModuleList(
            [nn.Linear(y_dim, hidden_dim, bias=False) for _ in range(L)]
            + [nn.Linear(y_dim, 1, bias=False)]
        )
        self.b_layers = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)]
            + [nn.Linear(hidden_dim, hidden_dim) for _ in range(1, L)]
            + [nn.Linear(hidden_dim, 1)]
        )
        self.u_layers = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)]
            + [nn.Linear(hidden_dim, hidden_dim) for _ in range(1, L)]  # Used to be L+1
        )
        self.clamp_weights()

    def clamp_weights(self) -> None:
        """Clamps weights of all the W_bar layers to be ≥ 0.
        Always call this function after loss.backward().
        """
        with torch.no_grad():
            for layer in self.W_bar_layers:
                if isinstance(layer, nn.Linear):
                    layer.weight.clamp_(min=0)
            if not self.y_in_output_layer:
                self.V_bar_layers[-1].weight.fill_(0.)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Computes the score for the given input examples (x, y).

        Args:
            x: shape [batch_size, input_dim], input
            y: shape [batch_size, y_dim], labels

        Returns:
            s: shape [batch_size], score
        """
        ReLU = nn.ReLU()
        u = x
        sigma = 0
        for l in range(self.L):
            if l > 0:
                W_hat_vec = ReLU(self.W_hat_layers[l](u))
                W_sigma = self.W_bar_layers[l](W_hat_vec * sigma)
            else:
                W_sigma = 0
            V_hat_vec = self.V_hat_layers[l](u)        # shape [batch, d]
            V_y = self.V_bar_layers[l](V_hat_vec * y)
            b = self.b_layers[l](u)
            sigma = ReLU(W_sigma + V_y + b)
            u = ReLU(self.u_layers[l](u))
        l = self.L
        W_hat_vec = ReLU(self.W_hat_layers[l](u))
        W_sigma = self.W_bar_layers[l](W_hat_vec * sigma)

        V_y = 0.
        if self.y_in_output_layer:
            V_hat_vec = self.V_hat_layers[l](u)
            V_y = self.V_bar_layers[l](V_hat_vec * y)
        b = self.b_layers[l](u)

        output = W_sigma + V_y + b
        if self.epsilon != 0:
            output += self.epsilon * torch.norm(y, p=float('inf'), dim=1)

        return output[..., 0]


def MALA_PICNN(
    model: PICNN,
    num_samples: int,
    burnin: int,
    x_init: Tensor,
    y_init: Tensor,
    std: float
) -> Tensor:
    """MALA MCMC sampling for PICNNs.

    See Algorithm 2 in Appendix B.2 of
    "How to Train Your FALCON: Learning Log-Concave Densities with Energy-Based
    Neural Networks" (Lin and Ba, 2023)

    Args:
        model: PICNN
        num_samples: number of samples
        burnin: number of burn-in steps
        x_init: initial sample, shape [num_chains, x_dim], on same device as model
        y_init: initial sample, shape [num_chains, y_dim], on same device as model
        std: standard deviation of the Gaussian proposal

    Returns:
        samples: shape [num_samples, num_chains, x_dim + y_dim], on same device as model
    """
    model.eval()

    x_dim = x_init.shape[1]
    xy_init = torch.cat([x_init, y_init], dim=1)

    # Initial gradient computation
    x_init.requires_grad_(True)
    y_init.requires_grad_(True)
    output_init = model(x_init, y_init)

    init_grad_x, init_grad_y = torch.autograd.grad(
        outputs=output_init, inputs=(x_init, y_init),
        grad_outputs=torch.ones_like(output_init))

    # compute mean proposal
    mu = xy_init - 0.5 * std**2 * torch.cat([init_grad_x, init_grad_y], dim=1)

    samples = []
    for t in range(burnin + num_samples):
        with torch.no_grad():
            sample_candidate = mu + torch.randn_like(xy_init) * std
        sample_candidate.requires_grad_(True)

        sample_x = sample_candidate[:, :x_dim]
        sample_y = sample_candidate[:, x_dim:]

        output = model(sample_x, sample_y)
        grad_x, grad_y = torch.autograd.grad(
            outputs=output, inputs=(sample_x, sample_y),
            grad_outputs=torch.ones_like(output))

        if grad_x is None or grad_y is None:
            raise RuntimeError("One of the gradients is None. Ensure the inputs are "
                               "used in the model's forward pass.")

        with torch.no_grad():
            nu = sample_candidate - 0.5 * std**2 * torch.cat([grad_x, grad_y], dim=1)

            acceptance = torch.exp(
                - output
                - torch.norm(xy_init - nu, dim=1)**2 / (2 * std**2)
                + output_init
                + torch.norm(sample_candidate - mu, dim=1)**2 / (2 * std**2)
            )
            acceptance.clamp_(0., 1.)
            accept = torch.bernoulli(acceptance).to(dtype=torch.bool)

            mu[accept] = nu[accept]
            xy_init[accept] = sample_candidate[accept]
            output_init[accept] = output[accept]

            if t >= burnin:
                samples.append(xy_init)

    return torch.stack(samples)


def fit_picnn_FALCON(
    model: PICNN,
    lr: float,
    l2reg: float,
    epochs: int,
    loader: torch.utils.data.DataLoader,
    device: str = 'cpu',
    num_chains: int = 1,
    num_samples: int = 1,
    burnin: int = 100,
    std: float = 0.25,
    warm_start: bool = False,
    weight_towards_zero: float = 0.
) -> tuple[PICNN, list[float]]:
    """
    Uses the FALCON algorithm to fit a log-concave distribution
    to the data using the PICNN in an energy-based model following
    the algorithm of https://openreview.net/forum?id=UP1r5-t-ov.

    Args:
        model: model to fit
        lr: learning rate
        l2reg: L2 regularization strength
        epochs number of epochs to train
        loader: data loader
        device: either 'cpu' or 'cuda'
        num_chains: # of MCMC chains to run in parallel
        num_samples: # of samples to draw from each chain
        burnin: # of MCMC steps before drawing samples
        std: standard deviation of MCMC proposal distribution
        warm_start: if True, warm starts the MCMC chains uniformly at random
            from the final samples of the chains from the previous epoch;
            otherwise, the MCMC chains start at the origin
        weight_towards_zero: weight for the loss term that encourages model(x,y) to
            be close to 0; set to 0 to ignore this loss term

    Returns:
        model: fitted model
        train_losses: list of losses for each epoch
    """
    model = model.eval().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2reg)

    xdim = model.input_dim
    ydim = model.y_dim
    x_init = torch.zeros((num_chains, xdim), device=device)
    y_init = torch.zeros((num_chains, ydim), device=device)

    pbar = tqdm(range(epochs))
    train_losses = []
    for epoch in pbar:
        total_loss = 0.
        num_examples = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # don't really know how to isolate the gradients
            optimizer.zero_grad()

            # collect the samples from MALA
            # shape [num_samples, num_chains, xdim + ydim]
            samples = MALA_PICNN(
                model, num_samples=num_samples, burnin=burnin,
                x_init=x_init, y_init=y_init, std=std)

            # zero the grad again just for good measure
            optimizer.zero_grad()
            samples = samples.detach()

            # compute the loss (model evaluated on batch - model evaluated on samples)
            data_out = model(x, y)
            samples_out = model(samples[:, :, :xdim].reshape(-1, xdim),
                                samples[:, :, xdim:].reshape(-1, ydim))
            loss = data_out.mean() - samples_out.mean()
            if weight_towards_zero > 0:
                loss += weight_towards_zero * (data_out**2).mean()
            loss.backward()
            optimizer.step()
            model.clamp_weights()

            total_loss += loss.item() * x.shape[0]
            num_examples += x.shape[0]

            if warm_start:
                # update the x_init and y_init by randomly sampling from the final samples
                indices = torch.randint(0, num_chains, (num_chains,))
                x_init = samples[-1, indices, :xdim]
                y_init = samples[-1, indices, xdim:]

        avg_loss = total_loss / num_examples
        msg = f'Epoch {epoch}, loss: {avg_loss:.3f}'
        pbar.set_description(msg)

        train_losses.append(avg_loss)
    return model, train_losses


def conformal_q(
    model: PICNN,
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
        scores = model(x, y)
        all_scores.append(scores)

    scores = torch.cat(all_scores)
    q = calc_q(scores, alpha)
    assert q != torch.inf, 'Size of calibration set is too small'
    return q
