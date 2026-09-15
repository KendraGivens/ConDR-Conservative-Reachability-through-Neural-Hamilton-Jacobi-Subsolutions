import torch
from lvfm.training import causal_binned_loss, causal_temporal_loss
from lvfm.helpers import squeeze_last

class LossManager:
    def __init__(
        self,
        residual,
        loss_weights=None,
        brt_weight_sigma=None,
        tau_weight_power=None,
        one_sided_loss=False,
        one_sided_boundary_weight=1.0,
        vol_reg_weight=0.0,
        vol_reg_eps=0.1,
        vol_reg_guard_threshold=0.0,
        vol_reg_guard_temperature=0.001,
        grad_weight=0.0,
        use_boundary_weighting=False,
        lambda_band=5.0,
        sigma_band=0.05,
        max_band_weight=20.0,
        use_causal_weighting=False,
        num_time_bins=10,
        causal_epsilon=5.0,
        min_causal_weight=0.05,
        use_eikonal=False,
        lambda_eikonal=1e-3,
        eikonal_band_width=0.1,
        hjvi_loss_mode="standard",
        hjvi_ineq_weight=1.0,
        hjvi_active_weight=0.25,
        hjvi_under_weight=0.0,
        hjvi_smooth_temp=0.02,
        pde_upper_weight=0.0,
        value_upper_weight=0.0,
        control_margin_weight=0.0,
        control_margin=0.05,
        control_margin_band=0.08,
        symmetry_weight=0.0,
        symmetry_transform=None,
        symmetry_translation=0.0,
    ):
        self.residual = residual
        self.loss_weights = {
            "terminal": 0.0,
            "pinn": 1.0,
            "raw_reg": 0.0,
            "latent_reg": 0.0,
        } if loss_weights is None else loss_weights
        self.brt_weight_sigma = brt_weight_sigma
        self.tau_weight_power = tau_weight_power
        # One-sided HJIVI: pde_residual² everywhere + relu(V-g)² (no dead zone).
        # Standard HJIVI dead zone: when both pde_res<0 and V-g<0, max(·)²≈0.
        # One-sided breaks this by always computing PDE gradient.
        self.one_sided_loss = one_sided_loss
        self.one_sided_boundary_weight = one_sided_boundary_weight
        # BRT volume regularisation: penalise sigmoid(-V/eps) to discourage
        # over-large BRTs. Unsupervised; weight must stay small to avoid under-approx.
        self.vol_reg_weight = vol_reg_weight
        self.vol_reg_eps = vol_reg_eps
        self.vol_reg_guard_threshold = float(vol_reg_guard_threshold)
        self.vol_reg_guard_temperature = float(vol_reg_guard_temperature)
        # Gradient-enhanced PDE loss (gPINNs-style): adds λ|∂r/∂x|² to the PINN
        # loss so that the residual has small spatial gradients everywhere.
        # Forces PDE satisfaction not just at sampled points but also nearby —
        # improves accuracy near sharp boundary features (e.g. Air3D D-shape).
        self.grad_weight = float(grad_weight)
        self.use_boundary_weighting = bool(use_boundary_weighting)
        self.lambda_band = float(lambda_band)
        self.sigma_band = float(sigma_band)
        self.max_band_weight = float(max_band_weight)
        self.use_causal_weighting = bool(use_causal_weighting)
        self.num_time_bins = int(num_time_bins)
        self.causal_epsilon = float(causal_epsilon)
        self.min_causal_weight = float(min_causal_weight)
        self.use_eikonal = bool(use_eikonal)
        self.lambda_eikonal = float(lambda_eikonal)
        self.eikonal_band_width = float(eikonal_band_width)
        self.hjvi_loss_mode = str(hjvi_loss_mode).lower()
        self.hjvi_ineq_weight = float(hjvi_ineq_weight)
        self.hjvi_active_weight = float(hjvi_active_weight)
        self.hjvi_under_weight = float(hjvi_under_weight)
        self.hjvi_smooth_temp = float(hjvi_smooth_temp)
        self.pde_upper_weight = float(pde_upper_weight)
        self.value_upper_weight = float(value_upper_weight)
        self.control_margin_weight = float(control_margin_weight)
        self.control_margin = float(control_margin)
        self.control_margin_band = float(control_margin_band)
        self.symmetry_weight = float(symmetry_weight)
        self.symmetry_transform = None if symmetry_transform is None else str(symmetry_transform).lower()
        self.symmetry_translation = float(symmetry_translation)

    def compute_losses(
        self,
        model,
        xt_interior,
        V=None,
        deepreach=True,
        causal_loss=False,
        causal_chunks=16,
        causal_eps=1.0,
    ):
        causal_weights = None
        causal_chunk_losses = None

        residual = None
        unweighted_point_loss = None
        pde_component = None
        vminusg_component = None
        loss_hjvi_ineq = torch.zeros((), device=xt_interior.device)
        loss_hjvi_active = torch.zeros((), device=xt_interior.device)
        pde_positive_frac = torch.zeros((), device=xt_interior.device)
        value_positive_frac = torch.zeros((), device=xt_interior.device)

        if self.one_sided_loss:
            if V is None:
                V = model(xt_interior)
            pde_res, V_minus_g = self.residual.compute_deepreach_residual(
                model=model, xt=xt_interior, return_components=True,
            )
            pde_component = pde_res
            vminusg_component = V_minus_g
            pde_point_loss = pde_res.pow(2)
            boundary_point_loss = torch.relu(V_minus_g).pow(2)
            loss_hjvi_ineq = torch.relu(pde_res).pow(2).mean() + boundary_point_loss.mean()
            pde_positive_frac = (pde_res > 0.0).float().mean()
            value_positive_frac = (V_minus_g > 0.0).float().mean()
            residual = pde_res
            unweighted_point_loss = pde_point_loss

            if self.tau_weight_power is not None and self.tau_weight_power > 0:
                tau_phys = self.residual._tau_phys_from_xt(xt_interior)
                tw = (tau_phys / self.residual.T).clamp(min=1e-6) ** self.tau_weight_power
                tw = tw / (tw.mean() + 1e-8)
                pde_point_loss = pde_point_loss * tw
                boundary_point_loss = boundary_point_loss * tw

            pde_point_loss = self._apply_point_weights(pde_point_loss, V)
            boundary_point_loss = self._apply_point_weights(boundary_point_loss, V)

            loss_pinn = pde_point_loss.mean() + self.one_sided_boundary_weight * boundary_point_loss.mean()

        else:
            if V is None:
                V = model(xt_interior)
            loss_mode = self.hjvi_loss_mode
            # Always split the VI into its two branches. Previously the split was
            # computed only for the loss modes that consume it, so the violation
            # diagnostics (pde_positive_frac, value_positive_frac, loss_hjvi_ineq,
            # loss_hjvi_active) kept their zero initialisation for `standard` runs
            # and were still written to train_metrics.csv -- a hard 0.0 that reads
            # as "no violations" and inverts the conservative-vs-vanilla
            # comparison (audit P0-4). The combined residual is reconstructed
            # below, so every loss mode is numerically unchanged.
            pde_res, V_minus_g = self.residual.compute_deepreach_residual(
                model=model, xt=xt_interior, return_components=True,
            )

            pde_component = pde_res
            vminusg_component = V_minus_g
            loss_hjvi_ineq = (
                torch.relu(pde_res).pow(2) + torch.relu(V_minus_g).pow(2)
            ).mean()
            pde_positive_frac = (pde_res > 0.0).float().mean()
            value_positive_frac = (V_minus_g > 0.0).float().mean()

            # Mirror HJVIResidualBase._hjvi(): the raw PDE residual when pde_only
            # is set, max(pde_res, V - g) otherwise.
            combined_residual = (
                pde_res
                if getattr(self.residual, "pde_only", False)
                else torch.maximum(pde_res, V_minus_g)
            )

            if loss_mode in {"complementarity", "relaxed_complementarity"}:
                residual = combined_residual
                ineq_loss = torch.relu(pde_res).pow(2) + torch.relu(V_minus_g).pow(2)
                active_loss = torch.minimum(pde_res.pow(2), V_minus_g.pow(2))
                loss_hjvi_ineq = ineq_loss.mean()
                loss_hjvi_active = active_loss.mean()
                pde_positive_frac = (pde_res > 0.0).float().mean()
                value_positive_frac = (V_minus_g > 0.0).float().mean()
                point_loss = self.hjvi_ineq_weight * ineq_loss + self.hjvi_active_weight * active_loss
                # Asymmetric PDE penalty. relu(pde_res)^2 above only penalises
                # VIOLATIONS (pde_res > 0); a tube that is too SMALL has
                # pde_res < 0 and is completely unpenalised, so nothing pulls it
                # back to the correct size (the observed high-dimensional
                # under-growth). Penalising relu(-pde_res)^2 with a SMALL weight
                # pulls the tube toward PDE equality while keeping the
                # conservative bias (violations still cost hjvi_ineq_weight >>
                # hjvi_under_weight), so V <= g and the subsolution direction are
                # preserved.
                if self.hjvi_under_weight > 0.0:
                    under_loss = torch.relu(-pde_res).pow(2)
                    point_loss = point_loss + self.hjvi_under_weight * under_loss
            elif loss_mode == "smoothmax":
                temp = max(self.hjvi_smooth_temp, 1e-6)
                residual = temp * torch.logsumexp(
                    torch.stack([pde_res / temp, V_minus_g / temp], dim=0),
                    dim=0,
                )
                point_loss = residual.pow(2)
            elif loss_mode in {"subsolution", "one_sided_subsolution"}:
                residual = combined_residual
                point_loss = torch.relu(pde_res).pow(2) + torch.relu(V_minus_g).pow(2)
            elif loss_mode == "subsolution_max":
                residual = combined_residual
                point_loss = torch.relu(residual).pow(2)
            else:
                residual = combined_residual
                point_loss = residual.pow(2)

            unweighted_point_loss = point_loss
            point_loss = self._apply_point_weights(point_loss, V)

            if self.tau_weight_power is not None and self.tau_weight_power > 0:
                tau_phys = self.residual._tau_phys_from_xt(xt_interior)
                tw = (tau_phys / self.residual.T).clamp(min=1e-6) ** self.tau_weight_power
                point_loss = point_loss * (tw / (tw.mean() + 1e-8))

            loss_pinn, causal_weights, causal_chunk_losses = self._reduce_point_loss(
                point_loss=point_loss,
                residual=residual,
                xt_interior=xt_interior,
                causal_loss=causal_loss,
                causal_chunks=causal_chunks,
                causal_eps=causal_eps,
            )

        # gPINN gradient-enhanced term. This used to live inside the
        # non-one-sided branch only, so `loss.grad_weight` was silently dropped
        # whenever `loss.one_sided_loss` was set (audit P2-4). It belongs to both
        # paths; `residual` is defined by either.
        if self.grad_weight > 0.0 and residual is not None:
            drdxt = torch.autograd.grad(
                residual.sum(), xt_interior,
                create_graph=True, retain_graph=True,
            )[0]
            drdx = drdxt[:, :self.residual.coordinate_dim]
            loss_pinn = loss_pinn + self.grad_weight * drdx.pow(2).sum(-1).mean()

        if residual is not None and unweighted_point_loss is not None:
            loss_hji_unweighted = unweighted_point_loss.mean()
        else:
            loss_hji_unweighted = torch.zeros((), device=xt_interior.device)

        # BRT volume regularisation: sigmoid(-V/eps) is a soft indicator for
        # x being inside the predicted BRT. Penalising its mean at large tau
        # pushes V positive, directly fighting over-approximation. Applied only
        # where tau > 0.3*T to leave small-tau learning unaffected.
        if self.vol_reg_weight > 0.0 and V is not None:
            V_s = V.squeeze(-1)
            tau_phys_vr = self.residual._tau_phys_from_xt(xt_interior)
            large_tau = (tau_phys_vr > 0.3 * self.residual.T).float().detach()
            n_large = large_tau.sum().clamp_min(1.0)
            loss_vol_reg_raw = (torch.sigmoid(-V_s / self.vol_reg_eps) * large_tau).sum() / n_large
            if self.vol_reg_guard_threshold > 0.0:
                guard_temp = max(self.vol_reg_guard_temperature, 1e-8)
                loss_vol_reg_guard = torch.sigmoid(
                    (self.vol_reg_guard_threshold - loss_hji_unweighted.detach()) / guard_temp
                )
                loss_vol_reg = loss_vol_reg_raw * loss_vol_reg_guard
            else:
                loss_vol_reg_guard = torch.ones((), device=xt_interior.device)
                loss_vol_reg = loss_vol_reg_raw
        else:
            loss_vol_reg_raw = torch.zeros((), device=xt_interior.device)
            loss_vol_reg_guard = torch.ones((), device=xt_interior.device)
            loss_vol_reg = torch.zeros((), device=xt_interior.device)

        loss_eikonal = self._compute_eikonal_loss(V, xt_interior)
        loss_symmetry = self._compute_symmetry_loss(model, V, xt_interior)

        if pde_component is not None and self.pde_upper_weight > 0.0:
            loss_pde_upper = torch.relu(pde_component).pow(2).mean()
        else:
            loss_pde_upper = torch.zeros((), device=xt_interior.device)

        if vminusg_component is not None and self.value_upper_weight > 0.0:
            loss_value_upper = torch.relu(vminusg_component).pow(2).mean()
        else:
            loss_value_upper = torch.zeros((), device=xt_interior.device)

        if (
            self.control_margin_weight > 0.0
            and V is not None
            and hasattr(self.residual, "compute_control_coeff")
        ):
            V_scalar = squeeze_last(V)
            grad_xt = torch.autograd.grad(
                V_scalar.sum(),
                xt_interior,
                create_graph=True,
                retain_graph=True,
            )[0]
            x_phys = self.residual._unscale_x(xt_interior[:, : self.residual.coordinate_dim])
            grad_phys = self.residual._unscale_spatial_gradient(
                grad_xt[:, : self.residual.coordinate_dim]
            )
            control_coeff = self.residual.compute_control_coeff(x_phys, grad_phys)
            band_weight = torch.exp(
                -V_scalar.detach().abs() / max(self.control_margin_band, 1e-8)
            )
            loss_control_margin = (
                torch.relu(self.control_margin - control_coeff.abs()).pow(2) * band_weight
            ).mean()
        else:
            loss_control_margin = torch.zeros((), device=xt_interior.device)

        loss_train = (
            self.loss_weights.get("pinn", 1.0) * loss_pinn
            + self.vol_reg_weight * loss_vol_reg
            + self.lambda_eikonal * loss_eikonal
            + self.symmetry_weight * loss_symmetry
            + self.pde_upper_weight * loss_pde_upper
            + self.value_upper_weight * loss_value_upper
            + self.control_margin_weight * loss_control_margin
        )

        out = {
            "loss_pinn": loss_pinn,
            "loss_hji_unweighted": loss_hji_unweighted,
            "loss_hjvi_ineq": loss_hjvi_ineq,
            "loss_hjvi_active": loss_hjvi_active,
            "loss_vol_reg": loss_vol_reg,
            "loss_vol_reg_raw": loss_vol_reg_raw,
            "loss_vol_reg_guard": loss_vol_reg_guard,
            "loss_eikonal": loss_eikonal,
            "loss_symmetry": loss_symmetry,
            "loss_pde_upper": loss_pde_upper,
            "loss_value_upper": loss_value_upper,
            "loss_control_margin": loss_control_margin,
            "pde_positive_frac": pde_positive_frac,
            "value_positive_frac": value_positive_frac,
            "loss_train": loss_train,
        }

        if causal_weights is not None:
            out["causal_weights"] = causal_weights.detach()
            out["causal_chunk_losses"] = causal_chunk_losses.detach()

        return out

    def _apply_point_weights(self, point_loss, V):
        if self.use_boundary_weighting and V is not None:
            V_detached = squeeze_last(V).detach().abs()
            band_weight = 1.0 + self.lambda_band * torch.exp(
                -V_detached / max(self.sigma_band, 1e-8)
            )
            band_weight = band_weight.clamp(max=self.max_band_weight)
            return point_loss * band_weight

        # Backward-compatible old knob.
        if self.brt_weight_sigma is not None and V is not None:
            bw = torch.exp(-squeeze_last(V).detach().abs() / self.brt_weight_sigma)
            return point_loss * (bw / (bw.mean() + 1e-8))

        return point_loss

    def _reduce_point_loss(
        self,
        point_loss,
        residual,
        xt_interior,
        causal_loss=False,
        causal_chunks=16,
        causal_eps=1.0,
    ):
        use_new_causal = self.use_causal_weighting
        use_old_causal = causal_loss and not use_new_causal
        tau_phys = self.residual._tau_phys_from_xt(xt_interior)

        if use_new_causal:
            return causal_binned_loss(
                point_losses=point_loss,
                tau_phys=tau_phys,
                T=self.residual.T,
                num_bins=self.num_time_bins,
                causal_epsilon=self.causal_epsilon,
                min_weight=self.min_causal_weight,
            )

        if use_old_causal:
            return causal_temporal_loss(
                residual=residual,
                tau_phys=tau_phys,
                T=self.residual.T,
                num_chunks=causal_chunks,
                eps=causal_eps,
                normalize_losses=True,
            )

        return point_loss.mean(), None, None

    def _compute_eikonal_loss(self, V, xt_interior):
        if not self.use_eikonal or V is None:
            return torch.zeros((), device=xt_interior.device)

        V_scalar = squeeze_last(V)
        V_detached = V_scalar.detach().abs()
        mask = V_detached < self.eikonal_band_width
        if not mask.any():
            return torch.zeros((), device=xt_interior.device)

        grad_xt = torch.autograd.grad(
            V_scalar.sum(),
            xt_interior,
            create_graph=True,
            retain_graph=True,
        )[0]
        grad_x = self.residual._unscale_spatial_gradient(
            grad_xt[:, : self.residual.coordinate_dim]
        )
        grad_norm = grad_x.norm(dim=-1)
        return (grad_norm[mask] - 1.0).pow(2).mean()

    def _compute_symmetry_loss(self, model, V, xt_interior):
        if self.symmetry_weight <= 0.0 or self.symmetry_transform in {None, "", "none"}:
            return torch.zeros((), device=xt_interior.device)

        transform = self.symmetry_transform
        mask = None
        if transform in {"air6d_se2", "air6d_joint_se2"}:
            # Continuous joint SE(2) invariance: rotate both vehicles by the
            # same random angle (and optionally translate both positions),
            # masking points whose transformed positions leave the domain.
            if self.residual.coordinate_dim != 6 or not hasattr(self.residual, "random_joint_se2"):
                return torch.zeros((), device=xt_interior.device)
            x_net_sym, valid = self.residual.random_joint_se2(
                xt_interior[:, : self.residual.coordinate_dim].detach(),
                max_translation=self.symmetry_translation,
            )
            if not valid.any():
                return torch.zeros((), device=xt_interior.device)
            xt_sym = xt_interior.detach().clone()
            xt_sym[:, : self.residual.coordinate_dim] = x_net_sym
            mask = valid
        elif transform in {"air3d_reflection", "air3d_y_psi_reflection"}:
            if self.residual.coordinate_dim != 3:
                return torch.zeros((), device=xt_interior.device)
            xt_sym = xt_interior.detach().clone()
            xt_sym[:, 1] = -xt_sym[:, 1]
            xt_sym[:, 2] = -xt_sym[:, 2]
        elif transform in {"multi9d_se2", "multi_9d_se2"}:
            if self.residual.coordinate_dim != 9 or not hasattr(self.residual, "random_joint_se2"):
                return torch.zeros((), device=xt_interior.device)
            x_net_sym, valid = self.residual.random_joint_se2(
                xt_interior[:, : self.residual.coordinate_dim].detach(),
                max_translation=self.symmetry_translation,
            )
            if not valid.any():
                return torch.zeros((), device=xt_interior.device)
            xt_sym = xt_interior.detach().clone()
            xt_sym[:, : self.residual.coordinate_dim] = x_net_sym
            mask = valid
        elif transform in {"air6d_rot90", "air6d_joint_rot90"}:
            if self.residual.coordinate_dim != 6:
                return torch.zeros((), device=xt_interior.device)
            xt_sym = xt_interior.detach().clone()
            x_phys = self.residual._unscale_x(
                xt_interior[:, : self.residual.coordinate_dim].detach()
            )

            # Rotate both vehicles by the same quarter-turn about the origin.
            # This preserves the square position domain while enforcing a
            # physical SE(2)-style invariance without reducing to relative
            # coordinates inside the model.
            k = torch.randint(1, 4, (x_phys.shape[0],), device=x_phys.device)
            phi = k.to(x_phys.dtype) * (0.5 * torch.pi)
            c = torch.cos(phi)
            s = torch.sin(phi)

            xp, yp = x_phys[:, 0], x_phys[:, 1]
            xe, ye = x_phys[:, 3], x_phys[:, 4]
            x_phys_sym = x_phys.clone()
            x_phys_sym[:, 0] = c * xp - s * yp
            x_phys_sym[:, 1] = s * xp + c * yp
            x_phys_sym[:, 3] = c * xe - s * ye
            x_phys_sym[:, 4] = s * xe + c * ye

            low, high = getattr(self.residual, "theta_bounds", (-torch.pi, torch.pi))
            width = float(high - low)
            x_phys_sym[:, 2] = ((x_phys_sym[:, 2] + phi - low) % width) + low
            x_phys_sym[:, 5] = ((x_phys_sym[:, 5] + phi - low) % width) + low

            if self.residual.scale_to_minus1_1:
                x_net_sym = x_phys_sym.clone()
                x_net_sym[:, 0] = 2.0 * (x_phys_sym[:, 0] - self.residual.x_bounds[0]) / (
                    self.residual.x_bounds[1] - self.residual.x_bounds[0]
                ) - 1.0
                x_net_sym[:, 1] = 2.0 * (x_phys_sym[:, 1] - self.residual.y_bounds[0]) / (
                    self.residual.y_bounds[1] - self.residual.y_bounds[0]
                ) - 1.0
                x_net_sym[:, 2] = x_phys_sym[:, 2] / self.residual.angle_scale
                x_net_sym[:, 3] = 2.0 * (x_phys_sym[:, 3] - self.residual.x_bounds[0]) / (
                    self.residual.x_bounds[1] - self.residual.x_bounds[0]
                ) - 1.0
                x_net_sym[:, 4] = 2.0 * (x_phys_sym[:, 4] - self.residual.y_bounds[0]) / (
                    self.residual.y_bounds[1] - self.residual.y_bounds[0]
                ) - 1.0
                x_net_sym[:, 5] = x_phys_sym[:, 5] / self.residual.angle_scale
            else:
                x_net_sym = x_phys_sym

            xt_sym[:, : self.residual.coordinate_dim] = x_net_sym
        else:
            raise ValueError(f"Unknown symmetry_transform: {self.symmetry_transform}")

        V_sym = model(xt_sym)

        diff_sq = (squeeze_last(V) - squeeze_last(V_sym)).pow(2)
        if mask is not None:
            return diff_sq[mask].mean()
        return diff_sq.mean()
