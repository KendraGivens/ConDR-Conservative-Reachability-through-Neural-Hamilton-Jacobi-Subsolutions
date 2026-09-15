import torch
import torch.nn as nn
import math
from lvfm.helpers import squeeze_last

class HJVIResidualBase(nn.Module):
    def __init__(
        self,
        coordinate_dim,
        radius,
        T=1.0,
        scale_to_minus1_1=True,
        scale_time_to_01=True,
        pde_only=False,
    ):
        super().__init__()
        self.coordinate_dim = int(coordinate_dim)
        self.radius = float(radius)
        self.T = float(T)
        self.scale_to_minus1_1 = bool(scale_to_minus1_1)
        self.scale_time_to_01 = bool(scale_time_to_01)
        # When True, skip the max(·, V−g) clamp and use the raw PDE residual
        # ∂V/∂τ − H directly.  The min-over-time BRT formulation guarantees
        # V ≤ g everywhere, so the V−g branch of the max is always non-positive
        # and creates dead zones where large PDE errors get masked.
        self.pde_only = bool(pde_only)

    def target_function(self, x_phys):
        return torch.sqrt(
            x_phys[..., 0] ** 2 + x_phys[..., 1] ** 2 + 1e-12
        ) - self.radius

    def _unscale_time_gradient(self, time_grad):
        if not self.scale_time_to_01:
            return time_grad
        return time_grad / self.T

    def _tau_phys_from_xt(self, xt):
        tau = xt[:, self.coordinate_dim]
        if self.scale_time_to_01:
            tau = tau * self.T
        return tau

    def _spatial_grad(self, V_scalar, xt):
        dV_dxt = torch.autograd.grad(
            V_scalar,
            xt,
            grad_outputs=torch.ones_like(V_scalar),
            retain_graph=True,
            create_graph=True,
        )[0]
        return dV_dxt[:, :self.coordinate_dim], dV_dxt[:, self.coordinate_dim]

    def _hjvi(self, V_scalar, tau_grad_phys, xt, spatial_grad_net, return_components=False):
        x_net = xt[:, :self.coordinate_dim]
        x_phys = self._unscale_x(x_net)
        spatial_grad_phys = self._unscale_spatial_gradient(spatial_grad_net)

        boundary = self.target_function(x_phys)
        hamiltonian = self.compute_hamiltonian(x_phys, spatial_grad_phys)

        pde_residual = tau_grad_phys - hamiltonian

        if return_components:
            # Returns (pde_residual, V - g) so the manager can form a one-sided
            # loss: pde_residual² everywhere + relu(V-g)² (only penalise V > g).
            # This eliminates the dead zone while preventing under-approximation.
            return pde_residual, V_scalar - boundary

        if self.pde_only:
            return pde_residual

        return torch.maximum(pde_residual, V_scalar - boundary)

    def compute_deepreach_residual(self, model, xt, return_components=False):
        xt = xt.requires_grad_(True)

        V = squeeze_last(model(xt))
        spatial_grad_net, tau_grad_net = self._spatial_grad(V, xt)

        tau_grad_phys = self._unscale_time_gradient(tau_grad_net)

        return self._hjvi(
            V_scalar=V,
            tau_grad_phys=tau_grad_phys,
            xt=xt,
            spatial_grad_net=spatial_grad_net,
            return_components=return_components,
        )


class LinearOscillator2DResidual(HJVIResidualBase):
    def __init__(
        self,
        oscillation_speed=1.0,
        control_bound=1.0,
        disturbance_bound=0.5,
        radius=0.25,
        T=1.0,
        x1_bounds=(-1.0, 1.0),
        x2_bounds=(-1.0, 1.0),
        scale_to_minus1_1=True,
        scale_time_to_01=True,
    ):
        super().__init__(
            coordinate_dim=2,
            radius=radius,
            T=T,
            scale_to_minus1_1=scale_to_minus1_1,
            scale_time_to_01=scale_time_to_01,
        )

        self.oscillation_speed = float(oscillation_speed)
        self.control_bound = float(control_bound)
        self.disturbance_bound = float(disturbance_bound)

        self.x1_bounds = tuple(x1_bounds)
        self.x2_bounds = tuple(x2_bounds)

    def _unscale_x(self, x_net):
        if not self.scale_to_minus1_1:
            return x_net

        x_phys = x_net.clone()

        x_phys[..., 0] = 0.5 * (x_net[..., 0] + 1.0) * (
            self.x1_bounds[1] - self.x1_bounds[0]
        ) + self.x1_bounds[0]

        x_phys[..., 1] = 0.5 * (x_net[..., 1] + 1.0) * (
            self.x2_bounds[1] - self.x2_bounds[0]
        ) + self.x2_bounds[0]

        return x_phys

    def _scale_x(self, x_phys):
        if not self.scale_to_minus1_1:
            return x_phys

        x_net = x_phys.clone()

        x_net[..., 0] = 2.0 * (x_phys[..., 0] - self.x1_bounds[0]) / (
            self.x1_bounds[1] - self.x1_bounds[0]
        ) - 1.0

        x_net[..., 1] = 2.0 * (x_phys[..., 1] - self.x2_bounds[0]) / (
            self.x2_bounds[1] - self.x2_bounds[0]
        ) - 1.0

        return x_net

    def _unscale_spatial_gradient(self, spatial_grad):
        if not self.scale_to_minus1_1:
            return spatial_grad

        grad = spatial_grad.clone()

        grad[..., 0] = spatial_grad[..., 0] * (
            2.0 / (self.x1_bounds[1] - self.x1_bounds[0])
        )

        grad[..., 1] = spatial_grad[..., 1] * (
            2.0 / (self.x2_bounds[1] - self.x2_bounds[0])
        )

        return grad

    def compute_hamiltonian(self, x_phys, spatial_grad):
        x1 = x_phys[..., 0]
        x2 = x_phys[..., 1]

        p1 = spatial_grad[..., 0]
        p2 = spatial_grad[..., 1]

        base = p1 * x2 + p2 * (-(self.oscillation_speed ** 2) * x1)

        control_term = -self.control_bound * torch.abs(p2)
        disturbance_term = self.disturbance_bound * torch.abs(p2)

        return base + control_term + disturbance_term

class Air3DResidual(HJVIResidualBase):
    def __init__(
        self,
        vp=0.75,
        ve=0.75,
        control_bound=3.0,
        disturbance_bound=3.0,
        radius=0.25,
        T=1.0,
        x_bounds=(-2.0, 2.0),
        y_bounds=(-2.0, 2.0),
        psi_bounds=(-math.pi, math.pi),
        scale_to_minus1_1=True,
        scale_time_to_01=True,
        angle_alpha_factor=1.2,
        pde_only=False,
    ):
        super().__init__(
            coordinate_dim=3,
            radius=radius,
            T=T,
            scale_to_minus1_1=scale_to_minus1_1,
            scale_time_to_01=scale_time_to_01,
            pde_only=pde_only,
        )

        self.vp = float(vp)
        self.ve = float(ve)
        self.control_bound = float(control_bound)
        self.disturbance_bound = float(disturbance_bound)

        self.x_bounds = tuple(x_bounds)
        self.y_bounds = tuple(y_bounds)
        self.psi_bounds = tuple(psi_bounds)

        self.angle_alpha_factor = float(angle_alpha_factor)
        self.angle_scale = self.angle_alpha_factor * math.pi

    def _unscale_x(self, x_net):
        if not self.scale_to_minus1_1:
            return x_net

        x_phys = x_net.clone()

        x_phys[..., 0] = 0.5 * (x_net[..., 0] + 1.0) * (
            self.x_bounds[1] - self.x_bounds[0]
        ) + self.x_bounds[0]

        x_phys[..., 1] = 0.5 * (x_net[..., 1] + 1.0) * (
            self.y_bounds[1] - self.y_bounds[0]
        ) + self.y_bounds[0]

        # DeepReach-style angle scaling.
        x_phys[..., 2] = x_net[..., 2] * self.angle_scale

        return x_phys

    def _scale_x(self, x_phys):
        if not self.scale_to_minus1_1:
            return x_phys

        x_net = x_phys.clone()

        x_net[..., 0] = 2.0 * (x_phys[..., 0] - self.x_bounds[0]) / (
            self.x_bounds[1] - self.x_bounds[0]
        ) - 1.0

        x_net[..., 1] = 2.0 * (x_phys[..., 1] - self.y_bounds[0]) / (
            self.y_bounds[1] - self.y_bounds[0]
        ) - 1.0

        # DeepReach-style angle scaling (inverse of _unscale_x).
        x_net[..., 2] = x_phys[..., 2] / self.angle_scale

        return x_net

    def _unscale_spatial_gradient(self, spatial_grad):
        if not self.scale_to_minus1_1:
            return spatial_grad

        grad = spatial_grad.clone()

        grad[..., 0] = spatial_grad[..., 0] * (
            2.0 / (self.x_bounds[1] - self.x_bounds[0])
        )

        grad[..., 1] = spatial_grad[..., 1] * (
            2.0 / (self.y_bounds[1] - self.y_bounds[0])
        )

        grad[..., 2] = spatial_grad[..., 2] / self.angle_scale

        return grad

    def compute_hamiltonian(self, x_phys, spatial_grad):
        x = x_phys[..., 0]
        y = x_phys[..., 1]
        psi = x_phys[..., 2]

        px = spatial_grad[..., 0]
        py = spatial_grad[..., 1]
        ppsi = spatial_grad[..., 2]

        base = (
            px * (-self.ve + self.vp * torch.cos(psi))
            + py * (self.vp * torch.sin(psi))
        )

        control_coeff = px * y - py * x - ppsi
        control_term = self.control_bound * torch.abs(control_coeff)

        disturbance_term = -self.disturbance_bound * torch.abs(ppsi)

        return base + control_term + disturbance_term

    def compute_control_coeff(self, x_phys, spatial_grad):
        x = x_phys[..., 0]
        y = x_phys[..., 1]

        px = spatial_grad[..., 0]
        py = spatial_grad[..., 1]
        ppsi = spatial_grad[..., 2]

        return px * y - py * x - ppsi

class Air6DJointResidual(HJVIResidualBase):
    def __init__(
        self,
        vp=0.75,
        ve=0.75,
        control_bound=3.0,
        disturbance_bound=3.0,
        radius=0.25,
        T=1.0,
        x_bounds=(-1.0, 1.0),
        y_bounds=(-1.0, 1.0),
        theta_bounds=(-math.pi, math.pi),
        scale_to_minus1_1=True,
        scale_time_to_01=True,
        angle_alpha_factor=1.2,
        pde_only=False,
    ):
        super().__init__(
            coordinate_dim=6,
            radius=radius,
            T=T,
            scale_to_minus1_1=scale_to_minus1_1,
            scale_time_to_01=scale_time_to_01,
            pde_only=pde_only,
        )

        self.vp = float(vp)
        self.ve = float(ve)
        self.control_bound = float(control_bound)
        self.disturbance_bound = float(disturbance_bound)

        self.x_bounds = tuple(x_bounds)
        self.y_bounds = tuple(y_bounds)
        self.theta_bounds = tuple(theta_bounds)

        self.angle_alpha_factor = float(angle_alpha_factor)
        self.angle_scale = self.angle_alpha_factor * math.pi

    def target_function(self, x_phys):
        xp = x_phys[..., 0]
        yp = x_phys[..., 1]

        xe = x_phys[..., 3]
        ye = x_phys[..., 4]

        dist = torch.sqrt((xp - xe) ** 2 + (yp - ye) ** 2 + 1e-12)
        return dist - self.radius

    def _unscale_x(self, x_net):
        if not self.scale_to_minus1_1:
            return x_net

        x_phys = x_net.clone()

        # pursuer x, y
        x_phys[..., 0] = 0.5 * (x_net[..., 0] + 1.0) * (
            self.x_bounds[1] - self.x_bounds[0]
        ) + self.x_bounds[0]

        x_phys[..., 1] = 0.5 * (x_net[..., 1] + 1.0) * (
            self.y_bounds[1] - self.y_bounds[0]
        ) + self.y_bounds[0]

        # pursuer heading
        x_phys[..., 2] = x_net[..., 2] * self.angle_scale

        # evader x, y
        x_phys[..., 3] = 0.5 * (x_net[..., 3] + 1.0) * (
            self.x_bounds[1] - self.x_bounds[0]
        ) + self.x_bounds[0]

        x_phys[..., 4] = 0.5 * (x_net[..., 4] + 1.0) * (
            self.y_bounds[1] - self.y_bounds[0]
        ) + self.y_bounds[0]

        # evader heading
        x_phys[..., 5] = x_net[..., 5] * self.angle_scale

        return x_phys

    def _unscale_spatial_gradient(self, spatial_grad):
        if not self.scale_to_minus1_1:
            return spatial_grad

        grad = spatial_grad.clone()

        # dV/dx_p, dV/dy_p
        grad[..., 0] = spatial_grad[..., 0] * (
            2.0 / (self.x_bounds[1] - self.x_bounds[0])
        )

        grad[..., 1] = spatial_grad[..., 1] * (
            2.0 / (self.y_bounds[1] - self.y_bounds[0])
        )

        # dV/dtheta_p
        grad[..., 2] = spatial_grad[..., 2] / self.angle_scale

        # dV/dx_e, dV/dy_e
        grad[..., 3] = spatial_grad[..., 3] * (
            2.0 / (self.x_bounds[1] - self.x_bounds[0])
        )

        grad[..., 4] = spatial_grad[..., 4] * (
            2.0 / (self.y_bounds[1] - self.y_bounds[0])
        )

        # dV/dtheta_e
        grad[..., 5] = spatial_grad[..., 5] / self.angle_scale

        return grad

    def _scale_x(self, x_phys):
        if not self.scale_to_minus1_1:
            return x_phys

        x_net = x_phys.clone()
        x_net[..., 0] = 2.0 * (x_phys[..., 0] - self.x_bounds[0]) / (
            self.x_bounds[1] - self.x_bounds[0]
        ) - 1.0
        x_net[..., 1] = 2.0 * (x_phys[..., 1] - self.y_bounds[0]) / (
            self.y_bounds[1] - self.y_bounds[0]
        ) - 1.0
        x_net[..., 2] = x_phys[..., 2] / self.angle_scale
        x_net[..., 3] = 2.0 * (x_phys[..., 3] - self.x_bounds[0]) / (
            self.x_bounds[1] - self.x_bounds[0]
        ) - 1.0
        x_net[..., 4] = 2.0 * (x_phys[..., 4] - self.y_bounds[0]) / (
            self.y_bounds[1] - self.y_bounds[0]
        ) - 1.0
        x_net[..., 5] = x_phys[..., 5] / self.angle_scale
        return x_net

    def random_joint_se2(self, x_net, max_translation=0.0):
        """Random joint SE(2) transform of both vehicles.

        The Air6D value function is invariant under rotating both vehicles
        about the origin by the same angle (both headings shift by that angle)
        and under translating both positions by the same offset. Returns
        (x_net_sym, valid_mask) where valid_mask marks points whose
        transformed positions remain inside the position bounds.
        """
        x_phys = self._unscale_x(x_net)
        n = x_phys.shape[0]
        phi = (torch.rand(n, device=x_phys.device, dtype=x_phys.dtype) * 2.0 - 1.0) * math.pi
        c = torch.cos(phi)
        s = torch.sin(phi)

        out = x_phys.clone()
        out[:, 0] = c * x_phys[:, 0] - s * x_phys[:, 1]
        out[:, 1] = s * x_phys[:, 0] + c * x_phys[:, 1]
        out[:, 3] = c * x_phys[:, 3] - s * x_phys[:, 4]
        out[:, 4] = s * x_phys[:, 3] + c * x_phys[:, 4]

        if max_translation > 0.0:
            t = (torch.rand(n, 2, device=x_phys.device, dtype=x_phys.dtype) * 2.0 - 1.0) * float(max_translation)
            out[:, 0] = out[:, 0] + t[:, 0]
            out[:, 1] = out[:, 1] + t[:, 1]
            out[:, 3] = out[:, 3] + t[:, 0]
            out[:, 4] = out[:, 4] + t[:, 1]

        low, high = self.theta_bounds
        width = float(high - low)
        out[:, 2] = ((x_phys[:, 2] + phi - low) % width) + low
        out[:, 5] = ((x_phys[:, 5] + phi - low) % width) + low

        valid = (
            (out[:, 0] >= self.x_bounds[0]) & (out[:, 0] <= self.x_bounds[1])
            & (out[:, 1] >= self.y_bounds[0]) & (out[:, 1] <= self.y_bounds[1])
            & (out[:, 3] >= self.x_bounds[0]) & (out[:, 3] <= self.x_bounds[1])
            & (out[:, 4] >= self.y_bounds[0]) & (out[:, 4] <= self.y_bounds[1])
        )
        return self._scale_x(out), valid

    def compute_hamiltonian(self, x_phys, spatial_grad):
        thp = x_phys[..., 2]
        the = x_phys[..., 5]

        p_xp = spatial_grad[..., 0]
        p_yp = spatial_grad[..., 1]
        p_thp = spatial_grad[..., 2]

        p_xe = spatial_grad[..., 3]
        p_ye = spatial_grad[..., 4]
        p_the = spatial_grad[..., 5]

        base = (
            p_xp * self.vp * torch.cos(thp)
            + p_yp * self.vp * torch.sin(thp)
            + p_xe * self.ve * torch.cos(the)
            + p_ye * self.ve * torch.sin(the)
        )

        # Evader control: omega_e = u, maximizing.
        control_term = self.control_bound * torch.abs(p_the)

        # Pursuer disturbance: omega_p = d, minimizing.
        disturbance_term = -self.disturbance_bound * torch.abs(p_thp)

        return base + control_term + disturbance_term


class MultiVehicle9DResidual(HJVIResidualBase):
    """9D three-vehicle collision avoidance (DeepReach's high-dim benchmark).

    State layout matches the official DeepReach MultiVehicleCollision:
    s = [x1, y1, x2, y2, x3, y3, th1, th2, th3]. Three Dubins cars, each
    controlling its own turn rate to AVOID pairwise collision (no disturbance).
    Target g(x) = min over pairwise distances - R (unsafe when any pair is
    within R). v = 0.6, omega_max = 1.1, R = 0.25.
    """

    def __init__(
        self,
        velocity=0.6,
        omega_max=1.1,
        radius=0.25,
        T=1.0,
        x_bounds=(-1.0, 1.0),
        y_bounds=(-1.0, 1.0),
        theta_bounds=(-math.pi, math.pi),
        scale_to_minus1_1=True,
        scale_time_to_01=True,
        angle_alpha_factor=1.2,
        pde_only=False,
    ):
        super().__init__(
            coordinate_dim=9,
            radius=radius,
            T=T,
            scale_to_minus1_1=scale_to_minus1_1,
            scale_time_to_01=scale_time_to_01,
            pde_only=pde_only,
        )
        self.velocity = float(velocity)
        self.omega_max = float(omega_max)
        self.x_bounds = tuple(x_bounds)
        self.y_bounds = tuple(y_bounds)
        self.theta_bounds = tuple(theta_bounds)
        self.angle_alpha_factor = float(angle_alpha_factor)
        self.angle_scale = self.angle_alpha_factor * math.pi
        # position dims 0-5, angle dims 6-8
        self._pos_idx = [0, 1, 2, 3, 4, 5]
        self._ang_idx = [6, 7, 8]

    def target_function(self, x_phys):
        p1 = x_phys[..., 0:2]
        p2 = x_phys[..., 2:4]
        p3 = x_phys[..., 4:6]
        d12 = torch.sqrt(((p1 - p2) ** 2).sum(-1) + 1e-12)
        d13 = torch.sqrt(((p1 - p3) ** 2).sum(-1) + 1e-12)
        d23 = torch.sqrt(((p2 - p3) ** 2).sum(-1) + 1e-12)
        return torch.minimum(torch.minimum(d12, d13), d23) - self.radius

    def _unscale_x(self, x_net):
        if not self.scale_to_minus1_1:
            return x_net
        x_phys = x_net.clone()
        for i in self._pos_idx:
            bnds = self.x_bounds if (i % 2 == 0) else self.y_bounds
            x_phys[..., i] = 0.5 * (x_net[..., i] + 1.0) * (bnds[1] - bnds[0]) + bnds[0]
        for i in self._ang_idx:
            x_phys[..., i] = x_net[..., i] * self.angle_scale
        return x_phys

    def _scale_x(self, x_phys):
        if not self.scale_to_minus1_1:
            return x_phys
        x_net = x_phys.clone()
        for i in self._pos_idx:
            bnds = self.x_bounds if (i % 2 == 0) else self.y_bounds
            x_net[..., i] = 2.0 * (x_phys[..., i] - bnds[0]) / (bnds[1] - bnds[0]) - 1.0
        for i in self._ang_idx:
            x_net[..., i] = x_phys[..., i] / self.angle_scale
        return x_net

    def _unscale_spatial_gradient(self, spatial_grad):
        if not self.scale_to_minus1_1:
            return spatial_grad
        grad = spatial_grad.clone()
        for i in self._pos_idx:
            bnds = self.x_bounds if (i % 2 == 0) else self.y_bounds
            grad[..., i] = spatial_grad[..., i] * (2.0 / (bnds[1] - bnds[0]))
        for i in self._ang_idx:
            grad[..., i] = spatial_grad[..., i] / self.angle_scale
        return grad

    def compute_hamiltonian(self, x_phys, spatial_grad):
        v = self.velocity
        w = self.omega_max
        th1 = x_phys[..., 6]
        th2 = x_phys[..., 7]
        th3 = x_phys[..., 8]
        ham = v * (torch.cos(th1) * spatial_grad[..., 0] + torch.sin(th1) * spatial_grad[..., 1]) + w * torch.abs(spatial_grad[..., 6])
        ham = ham + v * (torch.cos(th2) * spatial_grad[..., 2] + torch.sin(th2) * spatial_grad[..., 3]) + w * torch.abs(spatial_grad[..., 7])
        ham = ham + v * (torch.cos(th3) * spatial_grad[..., 4] + torch.sin(th3) * spatial_grad[..., 5]) + w * torch.abs(spatial_grad[..., 8])
        return ham

    def random_joint_se2(self, x_net, max_translation=0.0):
        """Joint SE(2): rotate all three vehicles about the origin by the same
        angle (headings shift by that angle) + optional joint translation.
        Returns (x_net_sym, valid_mask)."""
        x_phys = self._unscale_x(x_net)
        n = x_phys.shape[0]
        phi = (torch.rand(n, device=x_phys.device, dtype=x_phys.dtype) * 2.0 - 1.0) * math.pi
        c = torch.cos(phi)
        s = torch.sin(phi)
        out = x_phys.clone()
        for k in range(3):
            xi, yi = 2 * k, 2 * k + 1
            out[:, xi] = c * x_phys[:, xi] - s * x_phys[:, yi]
            out[:, yi] = s * x_phys[:, xi] + c * x_phys[:, yi]
        if max_translation > 0.0:
            t = (torch.rand(n, 2, device=x_phys.device, dtype=x_phys.dtype) * 2.0 - 1.0) * float(max_translation)
            for k in range(3):
                out[:, 2 * k] = out[:, 2 * k] + t[:, 0]
                out[:, 2 * k + 1] = out[:, 2 * k + 1] + t[:, 1]
        low, high = self.theta_bounds
        width = float(high - low)
        for i in self._ang_idx:
            out[:, i] = ((x_phys[:, i] + phi - low) % width) + low
        valid = torch.ones(n, dtype=torch.bool, device=x_phys.device)
        for k in range(3):
            valid &= (out[:, 2 * k] >= self.x_bounds[0]) & (out[:, 2 * k] <= self.x_bounds[1])
            valid &= (out[:, 2 * k + 1] >= self.y_bounds[0]) & (out[:, 2 * k + 1] <= self.y_bounds[1])
        return self._scale_x(out), valid


class Quadrotor13DResidual(HJVIResidualBase):
    """13D quadrotor obstacle-avoidance (DeepReach's highest-dim benchmark).

    State s = [x,y,z, qw,qx,qy,qz, vx,vy,vz, wx,wy,wz]: position, unit quaternion,
    linear velocity, body angular velocity. 4 thrust controls, avoid mode, no
    disturbance. Target g(x)=||pos||-R (unsafe inside a ball of radius R at the
    origin). Dynamics/Hamiltonian ported verbatim from the official DeepReach
    Quadrotor so the ROM is compared on the SAME Hamiltonian as cons-DeepReach.

    All 13 state ranges are symmetric about 0, so the [-1,1] scaling is a simple
    per-dim division by `state_scale` (no affine offset, no periodic angle).
    """

    # official state_test_range half-widths: pos +-1.5, quat +-1, vel +-10, omega +-10
    STATE_SCALE = [1.5, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]

    def __init__(
        self,
        collision_radius=0.5,
        thrust_max=1.0,
        T=1.0,
        scale_to_minus1_1=True,
        scale_time_to_01=True,
        pde_only=False,
    ):
        super().__init__(
            coordinate_dim=13,
            radius=collision_radius,
            T=T,
            scale_to_minus1_1=scale_to_minus1_1,
            scale_time_to_01=scale_time_to_01,
            pde_only=pde_only,
        )
        self.thrust_max = float(thrust_max)
        # physical constants (official Quadrotor)
        self.m = 1.0
        self.arm_l = 0.17
        self.CT = 1.0
        self.CM = 0.016
        self.Gz = -9.8
        self.register_buffer(
            "_scale", torch.tensor(self.STATE_SCALE, dtype=torch.float32)
        )

    def target_function(self, x_phys):
        return torch.sqrt(
            x_phys[..., 0] ** 2 + x_phys[..., 1] ** 2 + x_phys[..., 2] ** 2 + 1e-12
        ) - self.radius

    def _unscale_x(self, x_net):
        if not self.scale_to_minus1_1:
            return x_net
        return x_net * self._scale.to(x_net.device, x_net.dtype)

    def _scale_x(self, x_phys):
        if not self.scale_to_minus1_1:
            return x_phys
        return x_phys / self._scale.to(x_phys.device, x_phys.dtype)

    def _unscale_spatial_gradient(self, spatial_grad):
        if not self.scale_to_minus1_1:
            return spatial_grad
        # dV/dx_phys = dV/dx_net * dx_net/dx_phys = grad_net / scale
        return spatial_grad / self._scale.to(spatial_grad.device, spatial_grad.dtype)

    def compute_hamiltonian(self, x_phys, spatial_grad):
        qw, qx, qy, qz = x_phys[..., 3], x_phys[..., 4], x_phys[..., 5], x_phys[..., 6]
        vx, vy, vz = x_phys[..., 7], x_phys[..., 8], x_phys[..., 9]
        wx, wy, wz = x_phys[..., 10], x_phys[..., 11], x_phys[..., 12]
        g = spatial_grad  # dV/dstate in physical coordinates

        CT, m, arm_l, CM = self.CT, self.m, self.arm_l, self.CM
        C1 = 2 * (qw * qy + qx * qz) * CT / m
        C2 = 2 * (-qw * qx + qy * qz) * CT / m
        C3 = (1 - 2 * qx ** 2 - 2 * qy ** 2) * CT / m
        C4 = 4 * math.sqrt(2) * CT / (3 * arm_l * m)
        C5 = 4 * math.sqrt(2) * CT / (3 * arm_l * m)
        C6 = 12 * CT * CM / (7 * arm_l ** 2 * m)

        # drift (control-independent) terms
        ham = g[..., 0] * vx + g[..., 1] * vy + g[..., 2] * vz
        ham = ham - g[..., 3] * (wx * qx + wy * qy + wz * qz) / 2.0
        ham = ham + g[..., 4] * (wx * qw + wz * qy - wy * qz) / 2.0
        ham = ham + g[..., 5] * (wy * qw - wz * qx + wx * qz) / 2.0
        ham = ham + g[..., 6] * (wz * qw + wy * qx - wx * qy) / 2.0
        ham = ham + g[..., 9] * self.Gz
        ham = ham - g[..., 10] * 5 * wy * wz / 9.0 + g[..., 11] * 5 * wx * wz / 9.0

        # avoid (max over the 4 thrusts in [-thrust_max, thrust_max]): each |.| is
        # the total coefficient of one thrust u_i, so sum_i |coeff_i| * thrust_max.
        base = g[..., 7] * C1 + g[..., 8] * C2 + g[..., 9] * C3
        ham = ham + torch.abs(base + g[..., 10] * C4 - g[..., 11] * C5 + g[..., 12] * C6) * self.thrust_max
        ham = ham + torch.abs(base - g[..., 10] * C4 - g[..., 11] * C5 - g[..., 12] * C6) * self.thrust_max
        ham = ham + torch.abs(base - g[..., 10] * C4 + g[..., 11] * C5 + g[..., 12] * C6) * self.thrust_max
        ham = ham + torch.abs(base + g[..., 10] * C4 + g[..., 11] * C5 - g[..., 12] * C6) * self.thrust_max
        return ham




# ── Quaternion helpers (ported from MPC-DeepReach utils/quaternion.py) ──────────
def _quat_invert(q):
    scaling = torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device, dtype=q.dtype)
    return q * scaling


def _quat_raw_multiply(a, b):
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def _quat_apply(q, point):
    real = point.new_zeros(point.shape[:-1] + (1,))
    p = torch.cat((real, point), -1)
    out = _quat_raw_multiply(_quat_raw_multiply(q, p), _quat_invert(q))
    return out[..., 1:]


class SimpleQuadrotor13DResidual(HJVIResidualBase):
    """13D 'simpler quadrotor' as used by MPC-DeepReach (RSS 2025), ported verbatim.

    DISTINCT from Quadrotor13DResidual (the main-branch 4-rotor quad). Control here is
    u = [collective_thrust f, dwx, dwy, dwz] (1 thrust + 3 direct angular accels):
        dv = (thrust axis) * f ;  dw_i = u_{i+1} - Coriolis.
    Avoid mode, no disturbance. Obstacle = a vertical cylinder; g(x) is the exact
    full-body distance-to-cylinder (accounts for the arm extent under the current
    attitude). Constants/ranges match their dynamics.py Quadrotor exactly."""

    # symmetric state box half-widths: pos +-3, quat +-1, vel +-5, omega +-5
    STATE_SCALE = [3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]

    def __init__(self, collision_radius=0.5, thrust_max=20.0, dwx_max=8.0, dwy_max=8.0,
                 dwz_max=4.0, T=1.0, scale_to_minus1_1=True, scale_time_to_01=True, pde_only=False):
        super().__init__(coordinate_dim=13, radius=collision_radius, T=T,
                         scale_to_minus1_1=scale_to_minus1_1, scale_time_to_01=scale_time_to_01,
                         pde_only=pde_only)
        self.thrust_max = float(thrust_max)
        self.dwx_max, self.dwy_max, self.dwz_max = float(dwx_max), float(dwy_max), float(dwz_max)
        self.m, self.arm_l, self.CT, self.Gz = 1.0, 0.17, 1.0, -9.8
        self.register_buffer("_scale", torch.tensor(self.STATE_SCALE, dtype=torch.float32))

    def _dist_to_cylinder(self, x_phys, a=0.0, b=0.0):
        px = x_phys[..., 0] - a
        py = x_phys[..., 1] - b
        q = x_phys[..., 3:7]
        vworld = _quat_apply(q, torch.zeros_like(x_phys[..., :3]) + torch.tensor(
            [0.0, 0.0, 1.0], device=x_phys.device, dtype=x_phys.dtype))
        vx, vy, vz = vworld[..., 0], vworld[..., 1], vworld[..., 2]
        dist = torch.sqrt(px ** 2 + py ** 2 + 1e-12)
        denom = (px ** 2 * vx ** 2 + px ** 2 * vz ** 2 + 2 * px * py * vx * vy
                 + py ** 2 * vy ** 2 + py ** 2 * vz ** 2) + 1e-6
        # arm-extent correction: physically bounded by arm_l; clamp for numerical
        # stability (the raw term blows up as denom -> 0 at near-axis states, which
        # was causing NaN gradients in full-state training).
        corr = torch.sqrt(
            (self.arm_l ** 2 * px ** 2 * vz ** 2) / denom
            + (self.arm_l ** 2 * py ** 2 * vz ** 2) / denom + 1e-12)
        corr = torch.clamp(corr, max=self.arm_l)
        return torch.maximum(dist - corr, torch.zeros_like(dist)) - self.radius

    def target_function(self, x_phys):
        return self._dist_to_cylinder(x_phys, 0.0, 0.0)

    def _unscale_x(self, x_net):
        return x_net * self._scale.to(x_net.device, x_net.dtype) if self.scale_to_minus1_1 else x_net

    def _scale_x(self, x_phys):
        return x_phys / self._scale.to(x_phys.device, x_phys.dtype) if self.scale_to_minus1_1 else x_phys

    def _unscale_spatial_gradient(self, spatial_grad):
        return spatial_grad / self._scale.to(spatial_grad.device, spatial_grad.dtype) if self.scale_to_minus1_1 else spatial_grad

    def compute_hamiltonian(self, x_phys, spatial_grad):
        g = spatial_grad
        qw, qx, qy, qz = x_phys[..., 3], x_phys[..., 4], x_phys[..., 5], x_phys[..., 6]
        vx, vy, vz = x_phys[..., 7], x_phys[..., 8], x_phys[..., 9]
        wx, wy, wz = x_phys[..., 10], x_phys[..., 11], x_phys[..., 12]
        c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
        c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
        c3 = (1 - 2 * qx ** 2 - 2 * qy ** 2) * self.CT / self.m
        ham = g[..., 0] * vx + g[..., 1] * vy + g[..., 2] * vz
        ham = ham - g[..., 3] * (wx * qx + wy * qy + wz * qz) / 2.0
        ham = ham + g[..., 4] * (wx * qw + wz * qy - wy * qz) / 2.0
        ham = ham + g[..., 5] * (wy * qw - wz * qx + wx * qz) / 2.0
        ham = ham + g[..., 6] * (wz * qw + wy * qx - wx * qy) / 2.0
        ham = ham + g[..., 9] * self.Gz
        ham = ham - g[..., 10] * 5 * wy * wz / 9.0 + g[..., 11] * 5 * wx * wz / 9.0
        # avoid: control MAXIMISES the Hamiltonian (worst case), so +|.|*bound
        ham = ham + torch.abs(g[..., 7] * c1 + g[..., 8] * c2 + g[..., 9] * c3) * self.thrust_max
        ham = ham + torch.abs(g[..., 10]) * self.dwx_max + torch.abs(g[..., 11]) * self.dwy_max + torch.abs(g[..., 12]) * self.dwz_max
        return ham


class Dubins3DResidual(HJVIResidualBase):
    """3D Dubins car AVOID BRT (single vehicle avoiding a radius-R circle at origin),
    ported verbatim from MPC-DeepReach's Dubins3D. State [x, y, theta], control =
    turn rate u in [-omega_max, omega_max]. Dynamics: xdot=v cos th, ydot=v sin th,
    thetadot=u. Avoid Hamiltonian: v(cos th * p_x + sin th * p_y) + omega_max |p_th|.
    x,y in [-1,1] (scale 1); theta in [-pi,pi] (scaled by angle_scale)."""

    def __init__(self, velocity=1.0, omega_max=1.2, goalR=0.5, angle_alpha_factor=1.0,
                 T=1.0, scale_to_minus1_1=True, scale_time_to_01=True, pde_only=False):
        super().__init__(coordinate_dim=3, radius=goalR, T=T,
                         scale_to_minus1_1=scale_to_minus1_1, scale_time_to_01=scale_time_to_01,
                         pde_only=pde_only)
        self.velocity = float(velocity)
        self.omega_max = float(omega_max)
        self.goalR = float(goalR)
        self.angle_alpha_factor = float(angle_alpha_factor)
        self.angle_scale = self.angle_alpha_factor * math.pi

    def target_function(self, x_phys):
        return torch.sqrt(x_phys[..., 0] ** 2 + x_phys[..., 1] ** 2 + 1e-12) - self.goalR

    def _unscale_x(self, x_net):
        if not self.scale_to_minus1_1:
            return x_net
        x = x_net.clone()
        x[..., 2] = x_net[..., 2] * self.angle_scale     # x,y identity ([-1,1]); theta scaled
        return x

    def _scale_x(self, x_phys):
        if not self.scale_to_minus1_1:
            return x_phys
        x = x_phys.clone()
        x[..., 2] = x_phys[..., 2] / self.angle_scale
        return x

    def _unscale_spatial_gradient(self, spatial_grad):
        if not self.scale_to_minus1_1:
            return spatial_grad
        g = spatial_grad.clone()
        g[..., 2] = spatial_grad[..., 2] / self.angle_scale
        return g

    def compute_hamiltonian(self, x_phys, spatial_grad):
        th = x_phys[..., 2]
        g = spatial_grad
        # avoid: control maximizes (worst-case) -> + omega_max |p_th|
        return self.velocity * (torch.cos(th) * g[..., 0] + torch.sin(th) * g[..., 1]) \
            + self.omega_max * torch.abs(g[..., 2])
