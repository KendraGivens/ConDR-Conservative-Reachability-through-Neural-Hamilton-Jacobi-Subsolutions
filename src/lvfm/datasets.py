import torch
import math
from torch.utils.data import Dataset

class LinearOscillator2DDataset(Dataset):
    def __init__(
        self,
        num_batches,
        num_interior,
        num_terminal,
        T,
        x1_bounds=(-1.0, 1.0),
        x2_bounds=(-1.0, 1.0),
        tau_max=None,
        scale_to_minus1_1=True,
        scale_time_to_0_1=True,
        efficient=False,
        num_unique_taus=8,
    ):
        super().__init__()

        self.num_batches = int(num_batches)
        self.num_interior = int(num_interior)
        self.num_terminal = int(num_terminal)
        self.T = float(T)

        self.x1_bounds = tuple(x1_bounds)
        self.x2_bounds = tuple(x2_bounds)

        self.tau_max = float(self.T if tau_max is None else tau_max)

        self.scale_to_minus1_1 = bool(scale_to_minus1_1)
        self.scale_time_to_0_1 = bool(scale_time_to_0_1)

        self.efficient = bool(efficient)
        self.num_unique_taus = num_unique_taus

        self.coordinate_dim = 2

    def __len__(self):
        return self.num_batches

    def set_tau_max(self, tau_max):
        tau_max = float(tau_max)
        self.tau_max = max(0.0, min(self.T, tau_max))

    def _sample_uniform(self, n, bounds):
        low, high = bounds
        return low + (high - low) * torch.rand(n, 1)

    def _sample_states_phys(self, n):
        x1 = self._sample_uniform(n, self.x1_bounds)
        x2 = self._sample_uniform(n, self.x2_bounds)
        return torch.cat([x1, x2], dim=-1)

    def _sample_tau_phys(self, n):
        if self.tau_max <= 0.0:
            return torch.zeros(n, 1)

        # Non-efficient: independent random tau for every point.
        if not self.efficient:
            return self.tau_max * torch.rand(n, 1)

        # Efficient: reuse only num_unique_taus different tau values.
        if self.num_unique_taus is None or self.num_unique_taus >= n:
            return self.tau_max * torch.rand(n, 1)

        k = max(1, int(self.num_unique_taus))

        edges = torch.linspace(0.0, self.tau_max, steps=k + 1)

        tau_unique = []
        for i in range(k):
            low = edges[i]
            high = edges[i + 1]
            tau_unique.append(low + (high - low) * torch.rand(1))

        tau_unique = torch.stack(tau_unique).reshape(k, 1)

        idx = torch.randint(0, k, (n,))
        return tau_unique[idx]

    def _scale_tau(self, tau_phys):
        if not self.scale_time_to_0_1:
            return tau_phys
        return tau_phys / self.T

    def _scale_states(self, x_phys):
        if not self.scale_to_minus1_1:
            return x_phys

        bounds = torch.tensor(
            [self.x1_bounds, self.x2_bounds],
            dtype=x_phys.dtype,
            device=x_phys.device,
        )

        low = bounds[:, 0]
        high = bounds[:, 1]

        return 2.0 * (x_phys - low) / (high - low) - 1.0

    def __getitem__(self, idx):
        x_interior_phys = self._sample_states_phys(self.num_interior)
        x_interior_net = self._scale_states(x_interior_phys)

        tau_interior_phys = self._sample_tau_phys(self.num_interior)
        tau_interior_net = self._scale_tau(tau_interior_phys)

        xt_interior = torch.cat([x_interior_net, tau_interior_net], dim=-1)

        x_terminal_phys = self._sample_states_phys(self.num_terminal)
        x_terminal_net = self._scale_states(x_terminal_phys)

        tau_terminal_phys = torch.zeros(self.num_terminal, 1)
        tau_terminal_net = self._scale_tau(tau_terminal_phys)

        xt_terminal = torch.cat([x_terminal_net, tau_terminal_net], dim=-1)

        return {
            "xt_interior": xt_interior.float(),
            "xt_terminal": xt_terminal.float(),
            "x_interior_phys": x_interior_phys.float(),
            "tau_interior_phys": tau_interior_phys.squeeze(-1).float(),
            "x_terminal_phys": x_terminal_phys.float(),
        }


class Air3DDataset(Dataset):
    def __init__(
        self,
        num_batches,
        num_interior,
        num_terminal,
        T,
        x_bounds=(-2.0, 2.0),
        y_bounds=(-2.0, 2.0),
        psi_bounds=(-math.pi, math.pi),
        tau_max=None,
        scale_to_minus1_1=True,
        scale_time_to_0_1=True,
        efficient=False,
        num_unique_taus=8,
        angle_alpha_factor=1.2,
        boundary_oversample_frac=0.0,
        boundary_delta=0.4,
        unsafe_radius=0.25,
    ):
        super().__init__()

        self.num_batches = int(num_batches)
        self.num_interior = int(num_interior)
        self.num_terminal = int(num_terminal)
        self.T = float(T)

        self.x_bounds = tuple(x_bounds)
        self.y_bounds = tuple(y_bounds)
        self.psi_bounds = tuple(psi_bounds)

        self.tau_max = float(self.T if tau_max is None else tau_max)

        self.scale_to_minus1_1 = bool(scale_to_minus1_1)
        self.scale_time_to_0_1 = bool(scale_time_to_0_1)

        self.efficient = bool(efficient)
        self.num_unique_taus = num_unique_taus

        self.coordinate_dim = 3

        # DeepReach-style angle scaling:
        # psi_net = psi_phys / (angle_alpha_factor * pi)
        self.angle_alpha_factor = float(angle_alpha_factor)
        self.angle_scale = self.angle_alpha_factor * math.pi

        # Zero-level-set oversampling: draw boundary_oversample_frac of
        # interior points from a disk of radius (unsafe_radius + boundary_delta)
        # centred at the origin, concentrating samples near the BRT boundary.
        self.boundary_oversample_frac = float(boundary_oversample_frac)
        self.boundary_delta = float(boundary_delta)
        self.unsafe_radius = float(unsafe_radius)

    def __len__(self):
        return self.num_batches

    def set_tau_max(self, tau_max):
        tau_max = float(tau_max)
        self.tau_max = max(0.0, min(self.T, tau_max))

    def _sample_uniform(self, n, bounds):
        low, high = bounds
        return low + (high - low) * torch.rand(n, 1)

    def _sample_uniform_states(self, n):
        x = self._sample_uniform(n, self.x_bounds)
        y = self._sample_uniform(n, self.y_bounds)
        psi = self._sample_uniform(n, self.psi_bounds)
        return torch.cat([x, y, psi], dim=-1)

    def _sample_boundary_states(self, n):
        # Uniform over a disk of radius (unsafe_radius + boundary_delta).
        # This concentrates points near the origin, oversampling near g(x)=0.
        r_max = self.unsafe_radius + self.boundary_delta
        r = r_max * torch.sqrt(torch.rand(n))
        theta = 2 * math.pi * torch.rand(n)
        x = (r * torch.cos(theta)).clamp(self.x_bounds[0], self.x_bounds[1]).unsqueeze(1)
        y = (r * torch.sin(theta)).clamp(self.y_bounds[0], self.y_bounds[1]).unsqueeze(1)
        psi = self._sample_uniform(n, self.psi_bounds)
        return torch.cat([x, y, psi], dim=-1)

    def _sample_states_phys(self, n):
        if self.boundary_oversample_frac <= 0.0:
            return self._sample_uniform_states(n)
        n_boundary = int(n * self.boundary_oversample_frac)
        n_uniform = n - n_boundary
        parts = []
        if n_uniform > 0:
            parts.append(self._sample_uniform_states(n_uniform))
        if n_boundary > 0:
            parts.append(self._sample_boundary_states(n_boundary))
        return torch.cat(parts, dim=0)

    def _sample_tau_phys(self, n):
        if self.tau_max <= 0.0:
            return torch.zeros(n, 1)

        # Non-efficient: independent random tau for every point.
        if not self.efficient:
            return self.tau_max * torch.rand(n, 1)

        # Efficient: reuse only num_unique_taus different tau values.
        if self.num_unique_taus is None or self.num_unique_taus >= n:
            return self.tau_max * torch.rand(n, 1)

        k = max(1, int(self.num_unique_taus))

        edges = torch.linspace(0.0, self.tau_max, steps=k + 1)

        tau_unique = []
        for i in range(k):
            low = edges[i]
            high = edges[i + 1]
            tau_unique.append(low + (high - low) * torch.rand(1))

        tau_unique = torch.stack(tau_unique).reshape(k, 1)

        idx = torch.randint(0, k, (n,))
        return tau_unique[idx]

    def _scale_tau(self, tau_phys):
        if not self.scale_time_to_0_1:
            return tau_phys
        return tau_phys / self.T

    def _scale_states(self, x_phys):
        if not self.scale_to_minus1_1:
            return x_phys

        x_net = x_phys.clone()

        x_net[:, 0] = 2.0 * (x_phys[:, 0] - self.x_bounds[0]) / (
            self.x_bounds[1] - self.x_bounds[0]
        ) - 1.0

        x_net[:, 1] = 2.0 * (x_phys[:, 1] - self.y_bounds[0]) / (
            self.y_bounds[1] - self.y_bounds[0]
        ) - 1.0

        # Keep DeepReach-style angle scaling.
        x_net[:, 2] = x_phys[:, 2] / self.angle_scale

        return x_net

    def __getitem__(self, idx):
        x_interior_phys = self._sample_states_phys(self.num_interior)
        x_interior_net = self._scale_states(x_interior_phys)

        tau_interior_phys = self._sample_tau_phys(self.num_interior)
        tau_interior_net = self._scale_tau(tau_interior_phys)

        xt_interior = torch.cat([x_interior_net, tau_interior_net], dim=-1)

        x_terminal_phys = self._sample_states_phys(self.num_terminal)
        x_terminal_net = self._scale_states(x_terminal_phys)

        tau_terminal_phys = torch.zeros(self.num_terminal, 1)
        tau_terminal_net = self._scale_tau(tau_terminal_phys)

        xt_terminal = torch.cat([x_terminal_net, tau_terminal_net], dim=-1)

        return {
            "xt_interior": xt_interior.float(),
            "xt_terminal": xt_terminal.float(),
            "x_interior_phys": x_interior_phys.float(),
            "tau_interior_phys": tau_interior_phys.squeeze(-1).float(),
            "x_terminal_phys": x_terminal_phys.float(),
        }

class Air6DJointDataset(Dataset):
    def __init__(
        self,
        num_batches,
        num_interior,
        num_terminal,
        T,
        x_bounds=(-1.0, 1.0),
        y_bounds=(-1.0, 1.0),
        theta_bounds=(-math.pi, math.pi),
        tau_max=None,
        scale_to_minus1_1=True,
        scale_time_to_0_1=True,
        efficient=False,
        num_unique_taus=16,
        angle_alpha_factor=1.2,
        boundary_oversample_frac=0.0,
        boundary_delta=0.4,
        unsafe_radius=0.25,
    ):
        super().__init__()

        self.num_batches = int(num_batches)
        self.num_interior = int(num_interior)
        self.num_terminal = int(num_terminal)
        self.T = float(T)

        self.x_bounds = tuple(x_bounds)
        self.y_bounds = tuple(y_bounds)
        self.theta_bounds = tuple(theta_bounds)

        self.tau_max = float(self.T if tau_max is None else tau_max)

        self.scale_to_minus1_1 = bool(scale_to_minus1_1)
        self.scale_time_to_0_1 = bool(scale_time_to_0_1)

        self.efficient = bool(efficient)
        self.num_unique_taus = num_unique_taus

        self.coordinate_dim = 6

        self.angle_alpha_factor = float(angle_alpha_factor)
        self.angle_scale = self.angle_alpha_factor * math.pi
        self.boundary_oversample_frac = float(boundary_oversample_frac)
        self.boundary_delta = float(boundary_delta)
        self.unsafe_radius = float(unsafe_radius)

    def __len__(self):
        return self.num_batches

    def set_tau_max(self, tau_max):
        tau_max = float(tau_max)
        self.tau_max = max(0.0, min(self.T, tau_max))

    def _sample_uniform(self, n, bounds):
        low, high = bounds
        return low + (high - low) * torch.rand(n, 1)

    def _sample_uniform_states(self, n):
        xp = self._sample_uniform(n, self.x_bounds)
        yp = self._sample_uniform(n, self.y_bounds)
        thp = self._sample_uniform(n, self.theta_bounds)

        xe = self._sample_uniform(n, self.x_bounds)
        ye = self._sample_uniform(n, self.y_bounds)
        the = self._sample_uniform(n, self.theta_bounds)

        return torch.cat([xp, yp, thp, xe, ye, the], dim=-1)

    def _sample_boundary_states(self, n):
        r_max = self.unsafe_radius + self.boundary_delta

        x_low = self.x_bounds[0] + r_max
        x_high = self.x_bounds[1] - r_max
        y_low = self.y_bounds[0] + r_max
        y_high = self.y_bounds[1] - r_max
        xe_bounds = (x_low, x_high) if x_low < x_high else self.x_bounds
        ye_bounds = (y_low, y_high) if y_low < y_high else self.y_bounds

        xe = self._sample_uniform(n, xe_bounds)
        ye = self._sample_uniform(n, ye_bounds)
        the = self._sample_uniform(n, self.theta_bounds)

        r = r_max * torch.sqrt(torch.rand(n, 1))
        phi = 2.0 * math.pi * torch.rand(n, 1)
        x_rel = r * torch.cos(phi)
        y_rel = r * torch.sin(phi)

        c = torch.cos(the)
        s = torch.sin(the)
        xp = xe + c * x_rel - s * y_rel
        yp = ye + s * x_rel + c * y_rel
        xp = xp.clamp(self.x_bounds[0], self.x_bounds[1])
        yp = yp.clamp(self.y_bounds[0], self.y_bounds[1])

        psi_rel = self._sample_uniform(n, self.theta_bounds)
        thp = the + psi_rel
        low, high = self.theta_bounds
        width = high - low
        thp = ((thp - low) % width) + low

        return torch.cat([xp, yp, thp, xe, ye, the], dim=-1)

    def _sample_states_phys(self, n):
        if self.boundary_oversample_frac <= 0.0:
            return self._sample_uniform_states(n)
        n_boundary = int(n * self.boundary_oversample_frac)
        n_uniform = n - n_boundary
        parts = []
        if n_uniform > 0:
            parts.append(self._sample_uniform_states(n_uniform))
        if n_boundary > 0:
            parts.append(self._sample_boundary_states(n_boundary))
        return torch.cat(parts, dim=0)

    def _sample_tau_phys(self, n):
        if self.tau_max <= 0.0:
            return torch.zeros(n, 1)

        if not self.efficient:
            return self.tau_max * torch.rand(n, 1)

        if self.num_unique_taus is None or self.num_unique_taus >= n:
            return self.tau_max * torch.rand(n, 1)

        k = max(1, int(self.num_unique_taus))
        edges = torch.linspace(0.0, self.tau_max, steps=k + 1)

        tau_unique = []
        for i in range(k):
            low = edges[i]
            high = edges[i + 1]
            tau_unique.append(low + (high - low) * torch.rand(1))

        tau_unique = torch.stack(tau_unique).reshape(k, 1)
        idx = torch.randint(0, k, (n,))

        return tau_unique[idx]

    def _scale_tau(self, tau_phys):
        if not self.scale_time_to_0_1:
            return tau_phys
        return tau_phys / self.T

    def _scale_states(self, x_phys):
        if not self.scale_to_minus1_1:
            return x_phys

        x_net = x_phys.clone()

        # pursuer position
        x_net[:, 0] = 2.0 * (x_phys[:, 0] - self.x_bounds[0]) / (
            self.x_bounds[1] - self.x_bounds[0]
        ) - 1.0

        x_net[:, 1] = 2.0 * (x_phys[:, 1] - self.y_bounds[0]) / (
            self.y_bounds[1] - self.y_bounds[0]
        ) - 1.0

        # pursuer heading
        x_net[:, 2] = x_phys[:, 2] / self.angle_scale

        # evader position
        x_net[:, 3] = 2.0 * (x_phys[:, 3] - self.x_bounds[0]) / (
            self.x_bounds[1] - self.x_bounds[0]
        ) - 1.0

        x_net[:, 4] = 2.0 * (x_phys[:, 4] - self.y_bounds[0]) / (
            self.y_bounds[1] - self.y_bounds[0]
        ) - 1.0

        # evader heading
        x_net[:, 5] = x_phys[:, 5] / self.angle_scale

        return x_net

    def __getitem__(self, idx):
        x_interior_phys = self._sample_states_phys(self.num_interior)
        x_interior_net = self._scale_states(x_interior_phys)

        tau_interior_phys = self._sample_tau_phys(self.num_interior)
        tau_interior_net = self._scale_tau(tau_interior_phys)

        xt_interior = torch.cat([x_interior_net, tau_interior_net], dim=-1)

        x_terminal_phys = self._sample_states_phys(self.num_terminal)
        x_terminal_net = self._scale_states(x_terminal_phys)

        tau_terminal_phys = torch.zeros(self.num_terminal, 1)
        tau_terminal_net = self._scale_tau(tau_terminal_phys)

        xt_terminal = torch.cat([x_terminal_net, tau_terminal_net], dim=-1)

        return {
            "xt_interior": xt_interior.float(),
            "xt_terminal": xt_terminal.float(),
            "x_interior_phys": x_interior_phys.float(),
            "tau_interior_phys": tau_interior_phys.squeeze(-1).float(),
            "x_terminal_phys": x_terminal_phys.float(),
        }


class MultiVehicle9DDataset(Dataset):
    """9D three-vehicle collision avoidance. State layout matches official
    DeepReach: [x1,y1, x2,y2, x3,y3, th1,th2,th3]."""

    def __init__(
        self,
        num_batches,
        num_interior,
        num_terminal,
        T,
        x_bounds=(-1.0, 1.0),
        y_bounds=(-1.0, 1.0),
        theta_bounds=(-math.pi, math.pi),
        tau_max=None,
        scale_to_minus1_1=True,
        scale_time_to_0_1=True,
        efficient=False,
        num_unique_taus=16,
        angle_alpha_factor=1.2,
        boundary_oversample_frac=0.0,
        boundary_delta=0.4,
        unsafe_radius=0.25,
    ):
        super().__init__()
        self.num_batches = int(num_batches)
        self.num_interior = int(num_interior)
        self.num_terminal = int(num_terminal)
        self.T = float(T)
        self.x_bounds = tuple(x_bounds)
        self.y_bounds = tuple(y_bounds)
        self.theta_bounds = tuple(theta_bounds)
        self.tau_max = float(self.T if tau_max is None else tau_max)
        self.scale_to_minus1_1 = bool(scale_to_minus1_1)
        self.scale_time_to_0_1 = bool(scale_time_to_0_1)
        self.efficient = bool(efficient)
        self.num_unique_taus = num_unique_taus
        self.coordinate_dim = 9
        self.angle_alpha_factor = float(angle_alpha_factor)
        self.angle_scale = self.angle_alpha_factor * math.pi
        self.boundary_oversample_frac = float(boundary_oversample_frac)
        self.boundary_delta = float(boundary_delta)
        self.unsafe_radius = float(unsafe_radius)
        self._pos_idx = [0, 1, 2, 3, 4, 5]
        self._ang_idx = [6, 7, 8]

    def __len__(self):
        return self.num_batches

    def set_tau_max(self, tau_max):
        self.tau_max = max(0.0, min(self.T, float(tau_max)))

    def _sample_uniform(self, n, bounds):
        low, high = bounds
        return low + (high - low) * torch.rand(n, 1)

    def _sample_uniform_states(self, n):
        cols = []
        for k in range(3):
            cols.append(self._sample_uniform(n, self.x_bounds))
            cols.append(self._sample_uniform(n, self.y_bounds))
        for k in range(3):
            cols.append(self._sample_uniform(n, self.theta_bounds))
        return torch.cat(cols, dim=-1)

    def _sample_boundary_states(self, n):
        # Place vehicle 2 (and 3) near vehicle 1 so some pair is near collision.
        base = self._sample_uniform_states(n)
        r_max = self.unsafe_radius + self.boundary_delta
        for k in [1, 2]:  # vehicles 2 and 3 relative to vehicle 1
            r = r_max * torch.sqrt(torch.rand(n, 1))
            phi = 2.0 * math.pi * torch.rand(n, 1)
            base[:, 2 * k] = (base[:, 0:1] + r * torch.cos(phi)).squeeze(-1).clamp(self.x_bounds[0], self.x_bounds[1])
            base[:, 2 * k + 1] = (base[:, 1:2] + r * torch.sin(phi)).squeeze(-1).clamp(self.y_bounds[0], self.y_bounds[1])
        return base

    def _sample_states_phys(self, n):
        if self.boundary_oversample_frac <= 0.0:
            return self._sample_uniform_states(n)
        n_boundary = int(n * self.boundary_oversample_frac)
        n_uniform = n - n_boundary
        parts = []
        if n_uniform > 0:
            parts.append(self._sample_uniform_states(n_uniform))
        if n_boundary > 0:
            parts.append(self._sample_boundary_states(n_boundary))
        return torch.cat(parts, dim=0)

    def _sample_tau_phys(self, n):
        if self.tau_max <= 0.0:
            return torch.zeros(n, 1)
        if not self.efficient or self.num_unique_taus is None or self.num_unique_taus >= n:
            return self.tau_max * torch.rand(n, 1)
        k = max(1, int(self.num_unique_taus))
        edges = torch.linspace(0.0, self.tau_max, steps=k + 1)
        tau_unique = torch.stack(
            [edges[i] + (edges[i + 1] - edges[i]) * torch.rand(1) for i in range(k)]
        ).reshape(k, 1)
        idx = torch.randint(0, k, (n,))
        return tau_unique[idx]

    def _scale_tau(self, tau_phys):
        if not self.scale_time_to_0_1:
            return tau_phys
        return tau_phys / self.T

    def _scale_states(self, x_phys):
        if not self.scale_to_minus1_1:
            return x_phys
        x_net = x_phys.clone()
        for i in self._pos_idx:
            bnds = self.x_bounds if (i % 2 == 0) else self.y_bounds
            x_net[:, i] = 2.0 * (x_phys[:, i] - bnds[0]) / (bnds[1] - bnds[0]) - 1.0
        for i in self._ang_idx:
            x_net[:, i] = x_phys[:, i] / self.angle_scale
        return x_net

    def __getitem__(self, idx):
        x_interior_phys = self._sample_states_phys(self.num_interior)
        x_interior_net = self._scale_states(x_interior_phys)
        tau_interior_phys = self._sample_tau_phys(self.num_interior)
        tau_interior_net = self._scale_tau(tau_interior_phys)
        xt_interior = torch.cat([x_interior_net, tau_interior_net], dim=-1)

        x_terminal_phys = self._sample_states_phys(self.num_terminal)
        x_terminal_net = self._scale_states(x_terminal_phys)
        tau_terminal_net = self._scale_tau(torch.zeros(self.num_terminal, 1))
        xt_terminal = torch.cat([x_terminal_net, tau_terminal_net], dim=-1)

        return {
            "xt_interior": xt_interior.float(),
            "xt_terminal": xt_terminal.float(),
            "x_interior_phys": x_interior_phys.float(),
            "tau_interior_phys": tau_interior_phys.squeeze(-1).float(),
            "x_terminal_phys": x_terminal_phys.float(),
        }


class Quadrotor13DDataset(Dataset):
    """13D quadrotor obstacle avoidance. State layout matches official DeepReach:
    [x,y,z, qw,qx,qy,qz, vx,vy,vz, wx,wy,wz]. All ranges symmetric about 0, so
    scaling to [-1,1] is a per-dim division by the range half-width."""

    STATE_SCALE = [1.5, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]

    def __init__(
        self,
        num_batches,
        num_interior,
        num_terminal,
        T,
        collision_radius=0.5,
        tau_max=None,
        scale_to_minus1_1=True,
        scale_time_to_0_1=True,
        efficient=False,
        num_unique_taus=16,
        boundary_oversample_frac=0.0,
        boundary_delta=0.5,
    ):
        super().__init__()
        self.num_batches = int(num_batches)
        self.num_interior = int(num_interior)
        self.num_terminal = int(num_terminal)
        self.T = float(T)
        self.tau_max = float(self.T if tau_max is None else tau_max)
        self.scale_to_minus1_1 = bool(scale_to_minus1_1)
        self.scale_time_to_0_1 = bool(scale_time_to_0_1)
        self.efficient = bool(efficient)
        self.num_unique_taus = num_unique_taus
        self.coordinate_dim = 13
        self.collision_radius = float(collision_radius)
        self.boundary_oversample_frac = float(boundary_oversample_frac)
        self.boundary_delta = float(boundary_delta)
        self._scale = torch.tensor(self.STATE_SCALE, dtype=torch.float32)

    def __len__(self):
        return self.num_batches

    def set_tau_max(self, tau_max):
        self.tau_max = max(0.0, min(self.T, float(tau_max)))

    def _sample_uniform_states(self, n):
        # uniform in the symmetric physical box: U(-scale, scale) per dim
        return (2.0 * torch.rand(n, self.coordinate_dim) - 1.0) * self._scale

    def _sample_boundary_states(self, n):
        # place position on a shell near the collision sphere so g(x) ~ 0
        base = self._sample_uniform_states(n)
        r = self.collision_radius + self.boundary_delta * torch.rand(n, 1)
        v = torch.randn(n, 3)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-9)
        base[:, 0:3] = (r * v).clamp(-1.5, 1.5)
        return base

    def _sample_states_phys(self, n):
        if self.boundary_oversample_frac <= 0.0:
            return self._sample_uniform_states(n)
        n_b = int(n * self.boundary_oversample_frac)
        n_u = n - n_b
        parts = []
        if n_u > 0:
            parts.append(self._sample_uniform_states(n_u))
        if n_b > 0:
            parts.append(self._sample_boundary_states(n_b))
        return torch.cat(parts, dim=0)

    def _sample_tau_phys(self, n):
        if self.tau_max <= 0.0:
            return torch.zeros(n, 1)
        if not self.efficient or self.num_unique_taus is None or self.num_unique_taus >= n:
            return self.tau_max * torch.rand(n, 1)
        k = max(1, int(self.num_unique_taus))
        edges = torch.linspace(0.0, self.tau_max, steps=k + 1)
        tau_unique = torch.stack(
            [edges[i] + (edges[i + 1] - edges[i]) * torch.rand(1) for i in range(k)]
        ).reshape(k, 1)
        idx = torch.randint(0, k, (n,))
        return tau_unique[idx]

    def _scale_tau(self, tau_phys):
        if not self.scale_time_to_0_1:
            return tau_phys
        return tau_phys / self.T

    def _scale_states(self, x_phys):
        if not self.scale_to_minus1_1:
            return x_phys
        return x_phys / self._scale

    def __getitem__(self, idx):
        x_interior_phys = self._sample_states_phys(self.num_interior)
        x_interior_net = self._scale_states(x_interior_phys)
        tau_interior_phys = self._sample_tau_phys(self.num_interior)
        tau_interior_net = self._scale_tau(tau_interior_phys)
        xt_interior = torch.cat([x_interior_net, tau_interior_net], dim=-1)

        x_terminal_phys = self._sample_states_phys(self.num_terminal)
        x_terminal_net = self._scale_states(x_terminal_phys)
        tau_terminal_net = self._scale_tau(torch.zeros(self.num_terminal, 1))
        xt_terminal = torch.cat([x_terminal_net, tau_terminal_net], dim=-1)

        return {
            "xt_interior": xt_interior.float(),
            "xt_terminal": xt_terminal.float(),
            "x_interior_phys": x_interior_phys.float(),
            "tau_interior_phys": tau_interior_phys.squeeze(-1).float(),
            "x_terminal_phys": x_terminal_phys.float(),
        }




class SimpleQuadrotor13DDataset(Dataset):
    """Sampler for the MPC-DeepReach 'simpler quadrotor' (pos +-3, quat +-1, vel/omega +-5)."""
    STATE_SCALE = [3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]

    def __init__(self, num_batches, num_interior, num_terminal, T, tau_max=None,
                 scale_to_minus1_1=True, scale_time_to_0_1=True, efficient=False,
                 num_unique_taus=16, boundary_oversample_frac=0.0):
        super().__init__()
        self.num_batches=int(num_batches); self.num_interior=int(num_interior)
        self.num_terminal=int(num_terminal); self.T=float(T)
        self.tau_max=float(self.T if tau_max is None else tau_max)
        self.scale_to_minus1_1=bool(scale_to_minus1_1); self.scale_time_to_0_1=bool(scale_time_to_0_1)
        self.efficient=bool(efficient); self.num_unique_taus=num_unique_taus
        self.coordinate_dim=13
        self._scale=torch.tensor(self.STATE_SCALE, dtype=torch.float32)

    def __len__(self): return self.num_batches
    def set_tau_max(self, tau_max): self.tau_max=max(0.0, min(self.T, float(tau_max)))
    # Quaternion coordinates (indices 3:7) must lie on the unit 3-sphere. Box
    # sampling of all 13 coords put them off-manifold, so residual/coverage/rollout
    # stats were measured at physically invalid orientations (bug P0-D). Sample the
    # quaternion from a normalized Gaussian (uniform on S^3) and canonicalize the
    # double cover q ~ -q to the w>=0 hemisphere.
    QUAT_SLICE = slice(3, 7)
    @staticmethod
    def normalize_quat(states):
        """Project the quaternion block (idx 3:7) of 13D SimpleQuadrotor states
        onto S^3 and canonicalize the double cover q ~ -q to the w>=0 hemisphere.
        Accepts a numpy array or torch tensor (bug P0-D). Use in every evaluator
        that draws states from the raw state box."""
        import numpy as _np
        is_torch = hasattr(states, "detach")
        arr = states.detach().cpu().numpy() if is_torch else _np.asarray(states)
        arr = arr.astype(_np.float32).copy()
        q = arr[:, 3:7]
        q = q / _np.clip(_np.linalg.norm(q, axis=-1, keepdims=True), 1e-12, None)
        q = q * _np.where(q[:, :1] < 0, -1.0, 1.0)
        arr[:, 3:7] = q
        if is_torch:
            import torch as _t
            return _t.as_tensor(arr, dtype=states.dtype, device=states.device)
        return arr
    def _sample_states_phys(self, n):
        x = (2.0*torch.rand(n, self.coordinate_dim)-1.0)*self._scale
        q = torch.randn(n, 4)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        q = q * torch.where(q[..., :1] < 0, -1.0, 1.0)   # w>=0 hemisphere (double cover)
        x[:, self.QUAT_SLICE] = q
        return x
    def _scale_states(self, x): return x/self._scale if self.scale_to_minus1_1 else x
    def _scale_tau(self, t): return t/self.T if self.scale_time_to_0_1 else t

    def _sample_tau_phys(self, n):
        if self.tau_max<=0.0: return torch.zeros(n,1)
        if not self.efficient or self.num_unique_taus is None or self.num_unique_taus>=n:
            return self.tau_max*torch.rand(n,1)
        k=max(1,int(self.num_unique_taus)); edges=torch.linspace(0.0,self.tau_max,steps=k+1)
        tu=torch.stack([edges[i]+(edges[i+1]-edges[i])*torch.rand(1) for i in range(k)]).reshape(k,1)
        return tu[torch.randint(0,k,(n,))]

    def __getitem__(self, idx):
        xi=self._sample_states_phys(self.num_interior); ti=self._sample_tau_phys(self.num_interior)
        xt_interior=torch.cat([self._scale_states(xi), self._scale_tau(ti)], -1)
        xtl=self._sample_states_phys(self.num_terminal)
        xt_terminal=torch.cat([self._scale_states(xtl), self._scale_tau(torch.zeros(self.num_terminal,1))], -1)
        return {"xt_interior":xt_interior.float(),"xt_terminal":xt_terminal.float(),
                "x_interior_phys":xi.float(),"tau_interior_phys":ti.squeeze(-1).float(),
                "x_terminal_phys":xtl.float()}


class Dubins3DDataset(Dataset):
    """Dubins3D avoid sampler: [x,y] in [-1,1]^2, theta in [-pi,pi]."""
    def __init__(self, num_batches, num_interior, num_terminal, T, angle_alpha_factor=1.0,
                 tau_max=None, scale_to_minus1_1=True, scale_time_to_0_1=True,
                 efficient=False, num_unique_taus=16):
        super().__init__()
        self.num_batches=int(num_batches); self.num_interior=int(num_interior)
        self.num_terminal=int(num_terminal); self.T=float(T)
        self.tau_max=float(self.T if tau_max is None else tau_max)
        self.scale_to_minus1_1=bool(scale_to_minus1_1); self.scale_time_to_0_1=bool(scale_time_to_0_1)
        self.efficient=bool(efficient); self.num_unique_taus=num_unique_taus
        self.coordinate_dim=3; self.angle_scale=float(angle_alpha_factor)*math.pi
    def __len__(self): return self.num_batches
    def set_tau_max(self, tau_max): self.tau_max=max(0.0, min(self.T, float(tau_max)))
    def _sample_states_phys(self, n):
        xy = 2.0*torch.rand(n,2)-1.0
        th = (2.0*torch.rand(n,1)-1.0)*math.pi
        return torch.cat([xy, th], -1)
    def _scale_states(self, x):
        if not self.scale_to_minus1_1: return x
        s=x.clone(); s[...,2]=x[...,2]/self.angle_scale; return s
    def _scale_tau(self, t): return t/self.T if self.scale_time_to_0_1 else t
    def _sample_tau_phys(self, n):
        if self.tau_max<=0.0: return torch.zeros(n,1)
        if not self.efficient or self.num_unique_taus is None or self.num_unique_taus>=n:
            return self.tau_max*torch.rand(n,1)
        k=max(1,int(self.num_unique_taus)); edges=torch.linspace(0.0,self.tau_max,steps=k+1)
        tu=torch.stack([edges[i]+(edges[i+1]-edges[i])*torch.rand(1) for i in range(k)]).reshape(k,1)
        return tu[torch.randint(0,k,(n,))]
    def __getitem__(self, idx):
        xi=self._sample_states_phys(self.num_interior); ti=self._sample_tau_phys(self.num_interior)
        xt_interior=torch.cat([self._scale_states(xi), self._scale_tau(ti)], -1)
        xtl=self._sample_states_phys(self.num_terminal)
        xt_terminal=torch.cat([self._scale_states(xtl), self._scale_tau(torch.zeros(self.num_terminal,1))], -1)
        return {"xt_interior":xt_interior.float(),"xt_terminal":xt_terminal.float(),
                "x_interior_phys":xi.float(),"tau_interior_phys":ti.squeeze(-1).float(),
                "x_terminal_phys":xtl.float()}
