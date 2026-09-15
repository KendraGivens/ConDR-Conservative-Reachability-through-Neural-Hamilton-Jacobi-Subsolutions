"""Models for conservative neural HJ reachability.

`DeepReachExact` is the ansatz all four experiment models share:

    V(x, tau) = g(x) + tau * (value_var / value_normto) * rho(x, tau)

with `correction_mode: free` reproducing DeepReach's 'exact' model, and
`correction_mode: negative_softplus` giving rho = -softplus(.) <= 0, so
V <= g holds by construction -- the structural half of the conservative
formulation.

The LatentReach / bottleneck-ROM models (INR_PNODE, MixtureINR_PNODE, the
PNODEs, and the conditioned/basis decoders) were removed on 2026-08-27: that
thread is superseded and the code is in archive/. Recover it from git history
or archive/scripts_offthesis/ if it is ever needed again.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, omega0=30.0, is_first=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.omega0 = omega0
        self.is_first = is_first

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.omega0
            self.weight.uniform_(-bound, bound)
            self.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega0 * F.linear(x, self.weight, self.bias))

class InvariantEncoder(nn.Module):
    """Learned invariant bottleneck for joint-state systems.

    Maps the full (network-scaled) state to a small feature vector that the
    conditioned decoder consumes in place of raw coordinates, so the
    dimensionality reduction is learned rather than hand-coded (e.g. the
    Air6D relative-coordinate transform). Trained jointly with the PDE loss
    plus an invariance penalty under joint symmetry transforms
    (see INR_PNODE encoder_invariance_weight). Angles are lifted to sin/cos
    so heading wraps cannot break invariance, and the output is tanh-bounded
    so the downstream SIREN decoder sees inputs in (-1, 1).
    """

    def __init__(
        self,
        coordinate_dim,
        out_dim,
        hidden_dim=128,
        num_layers=3,
        angle_indices=(),
        angle_scale=math.pi,
    ):
        super().__init__()
        self.coordinate_dim = int(coordinate_dim)
        self.out_dim = int(out_dim)
        self.angle_indices = tuple(int(i) for i in angle_indices)
        self.angle_scale = float(angle_scale)

        in_dim = self.coordinate_dim + len(self.angle_indices)
        layers = []
        dim = in_dim
        for _ in range(max(int(num_layers), 1)):
            layers.extend([nn.Linear(dim, hidden_dim), nn.SiLU()])
            dim = hidden_dim
        layers.append(nn.Linear(dim, self.out_dim))
        self.net = nn.Sequential(*layers)

    def features(self, x_net):
        feats = []
        angle_set = set(self.angle_indices)
        for idx in range(self.coordinate_dim):
            val = x_net[..., idx : idx + 1]
            if idx in angle_set:
                angle_phys = val * self.angle_scale
                feats.extend([torch.sin(angle_phys), torch.cos(angle_phys)])
            else:
                feats.append(val)
        return torch.cat(feats, dim=-1)

    def forward(self, x_net):
        return torch.tanh(self.net(self.features(x_net)))

class DeepReachModel(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=512, num_layers=4, omega0=30.0):
        super().__init__()
        layers = [SirenLayer(in_dim, hidden_dim, omega0=omega0, is_first=True)]
        for _ in range(num_layers-1):
            layers.append(SirenLayer(hidden_dim, hidden_dim, omega0=omega0, is_first=False))
        self.hidden = nn.Sequential(*layers)
        self.final = nn.Linear(hidden_dim, 1)

    def forward(self, xt):
        out = self.hidden(xt)
        return self.final(out)

class DeepReachExact(nn.Module):
    def __init__(self, backbone, residual, coordinate_dim=3, value_var=0.5, value_normto=0.02,
                 correction_mode="free", correction_softplus_shift=-8.0,
                 encoder=None, encoder_invariance_weight=0.0, encoder_invariance_translation=0.0):
        super().__init__()
        self.backbone = backbone
        self.residual = residual
        self.coordinate_dim = coordinate_dim
        self.value_var = float(value_var)
        self.value_normto = float(value_normto)
        # Nonpositive correction (same as INR_PNODE): structurally enforces
        # V <= g for tau > 0, giving the conservative-DeepReach variant.
        self.correction_mode = str(correction_mode).lower()
        self.correction_softplus_shift = float(correction_softplus_shift)
        # Optional learned invariant bottleneck (same module as LatentReach):
        # the backbone consumes [E(x), tau] instead of [x, tau]. Autodiff still
        # yields dV/dx correctly because E is inside forward().
        self.encoder = encoder
        self.encoder_invariance_weight = float(encoder_invariance_weight)
        self.encoder_invariance_translation = float(encoder_invariance_translation)

    def raw_output(self, xt):
        x_net = xt[:, :self.coordinate_dim]
        tau_net = xt[:, self.coordinate_dim:self.coordinate_dim + 1]
        feat = self.encoder(x_net) if self.encoder is not None else x_net
        return self.backbone(torch.cat([feat, tau_net], dim=-1)).squeeze(-1)

    def transform_raw_correction(self, raw):
        if self.correction_mode in {"free", "signed", "none"}:
            return raw
        if self.correction_mode in {"negative_softplus", "nonpositive_softplus"}:
            return -F.softplus(raw + self.correction_softplus_shift)
        if self.correction_mode in {"negative_square", "nonpositive_square"}:
            return -raw.pow(2)
        raise ValueError(f"Unknown correction_mode: {self.correction_mode}")

    def forward(self, xt):
        raw = self.transform_raw_correction(self.raw_output(xt))
        x_net = xt[:, :self.coordinate_dim]
        tau_net = xt[:, self.coordinate_dim]
        tau_phys = tau_net * self.residual.T if self.residual.scale_time_to_01 else tau_net
        x_phys = self.residual._unscale_x(x_net)
        boundary = self.residual.target_function(x_phys)
        V = boundary + tau_phys * (self.value_var / self.value_normto) * raw
        return V.unsqueeze(-1)

    def compute_encoder_invariance_loss(self, xt_interior):
        residual = self.residual
        if self.encoder is None or not hasattr(residual, "random_joint_se2"):
            return torch.zeros((), device=xt_interior.device)
        x_net = xt_interior[:, : residual.coordinate_dim].detach()
        x_sym, valid = residual.random_joint_se2(
            x_net, max_translation=self.encoder_invariance_translation)
        if not valid.any():
            return torch.zeros((), device=xt_interior.device)
        return (self.encoder(x_net[valid]) - self.encoder(x_sym[valid])).pow(2).sum(-1).mean()
