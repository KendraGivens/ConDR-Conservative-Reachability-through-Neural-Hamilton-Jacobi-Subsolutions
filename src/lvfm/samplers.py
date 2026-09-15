import math
from dataclasses import dataclass

import torch


@dataclass
class SamplerConfig:
    type: str = "uniform"
    uniform_frac: float = 1.0
    pred_boundary_frac: float = 0.0
    target_boundary_frac: float = 0.0
    residual_frac: float = 0.0
    residual_score: str = "abs"
    hard_psi_frac: float = 0.0
    hard_psi_centers: tuple = ()
    hard_psi_std: float = 0.20
    candidate_multiplier: float = 4.0
    topk_frac: float = 0.2
    boundary_temperature: float = 0.05
    detach_boundary_selection: bool = True
    chunk_size: int = 4096

    @classmethod
    def from_config(cls, cfg):
        if cfg is None:
            return cls()
        data = vars(cfg) if hasattr(cfg, "__dict__") else dict(cfg)
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


def _empty_points(n, coordinate_dim):
    return {
        "x_phys": torch.empty(n, coordinate_dim),
        "tau_phys": torch.empty(n, 1),
        "xt": torch.empty(n, coordinate_dim + 1),
    }


class MixedCollocationSampler:
    """Unsupervised collocation sampler for HJI training.

    Ground-truth value grids are intentionally not accepted here. Selection uses
    the current model prediction, the analytic target function g(x), and the HJI
    residual only.
    """

    def __init__(self, dataset, residual, config=None, device="cpu", deepreach=False):
        self.dataset = dataset
        self.residual = residual
        self.config = SamplerConfig.from_config(config)
        self.device = device
        self.deepreach = deepreach
        self.coordinate_dim = int(dataset.coordinate_dim)
        self._warned_topk = False

    def is_uniform(self):
        return self.config.type.lower() == "uniform"

    def sample_batch(self, model, batch=None, num_interior=None):
        if num_interior is None:
            num_interior = int(self.dataset.num_interior)

        if self.is_uniform():
            return batch if batch is not None else self.dataset[0]

        counts = self._component_counts(num_interior)
        parts = []

        if counts["uniform"] > 0:
            parts.append(("uniform", self._sample_uniform_points(counts["uniform"])))
        if counts["pred_boundary"] > 0:
            parts.append(
                ("pred_boundary", self._sample_predicted_boundary(model, counts["pred_boundary"]))
            )
        if counts["target_boundary"] > 0:
            parts.append(("target_boundary", self._sample_target_boundary(counts["target_boundary"])))
        if counts["residual"] > 0:
            parts.append(("residual", self._sample_residual_adaptive(model, counts["residual"])))
        if counts["hard_psi"] > 0:
            parts.append(("hard_psi", self._sample_hard_psi_points(counts["hard_psi"])))

        xt = torch.cat([p["xt"] for _, p in parts], dim=0)
        x_phys = torch.cat([p["x_phys"] for _, p in parts], dim=0)
        tau_phys = torch.cat([p["tau_phys"] for _, p in parts], dim=0).squeeze(-1)

        perm = torch.randperm(xt.shape[0])
        out = dict(batch) if batch is not None else self.dataset[0]
        out["xt_interior"] = xt[perm].float()
        out["x_interior_phys"] = x_phys[perm].float()
        out["tau_interior_phys"] = tau_phys[perm].float()
        out["sampler_counts"] = {name: pts["xt"].shape[0] for name, pts in parts}
        return out

    def _component_counts(self, n):
        sampler_type = self.config.type.lower()
        fracs = {
            "uniform": float(self.config.uniform_frac),
            "pred_boundary": float(self.config.pred_boundary_frac),
            "target_boundary": float(self.config.target_boundary_frac),
            "residual": float(self.config.residual_frac),
            "hard_psi": float(self.config.hard_psi_frac),
        }

        if sampler_type == "boundary":
            fracs["residual"] = 0.0
            fracs["hard_psi"] = 0.0
        elif sampler_type == "residual":
            fracs["pred_boundary"] = 0.0
            fracs["target_boundary"] = 0.0
            fracs["hard_psi"] = 0.0
        elif sampler_type != "mixed":
            fracs = {
                "uniform": 1.0,
                "pred_boundary": 0.0,
                "target_boundary": 0.0,
                "residual": 0.0,
                "hard_psi": 0.0,
            }

        total = sum(max(v, 0.0) for v in fracs.values())
        if total <= 0.0:
            fracs["uniform"] = 1.0
            total = 1.0

        names = list(fracs)
        raw = [n * max(fracs[name], 0.0) / total for name in names]
        counts = {name: int(math.floor(v)) for name, v in zip(names, raw)}
        remainder = n - sum(counts.values())
        order = sorted(range(len(names)), key=lambda i: raw[i] - math.floor(raw[i]), reverse=True)
        for i in order[:remainder]:
            counts[names[i]] += 1
        return counts

    def _sample_uniform_states(self, n):
        if n <= 0:
            return torch.empty(0, self.coordinate_dim)
        if hasattr(self.dataset, "_sample_uniform_states"):
            return self.dataset._sample_uniform_states(n)
        return self.dataset._sample_states_phys(n)

    def _sample_tau(self, n):
        if n <= 0:
            return torch.empty(0, 1)
        return self.dataset._sample_tau_phys(n)

    def _pack(self, x_phys, tau_phys):
        x_net = self.dataset._scale_states(x_phys)
        tau_net = self.dataset._scale_tau(tau_phys)
        xt = torch.cat([x_net, tau_net], dim=-1)
        return {"x_phys": x_phys, "tau_phys": tau_phys, "xt": xt}

    def _sample_uniform_points(self, n):
        if n <= 0:
            return _empty_points(0, self.coordinate_dim)
        return self._pack(self._sample_uniform_states(n), self._sample_tau(n))

    def _num_candidates(self, n):
        return max(n, int(math.ceil(n * float(self.config.candidate_multiplier))))

    def _draw_from_scores(self, scores, n, largest):
        if n <= 0:
            return torch.empty(0, dtype=torch.long)
        num_candidates = scores.numel()
        k = max(n, int(math.ceil(num_candidates * float(self.config.topk_frac))))
        k = min(k, num_candidates)
        # When candidate_multiplier * topk_frac <= 1 the guard binds, k == n, and
        # the softmax/temperature draw below is unreachable -- the sampler silently
        # degenerates to a deterministic top-n and `boundary_temperature` /
        # `topk_frac` have no effect (audit P2-7). Say so once, loudly.
        if k <= n and not self._warned_topk:
            self._warned_topk = True
            print(
                f"[sampler] WARNING: candidate_multiplier="
                f"{self.config.candidate_multiplier} x topk_frac={self.config.topk_frac} "
                f"gives only {num_candidates} candidates for {n} draws, so selection is "
                f"deterministic top-{n} and boundary_temperature="
                f"{self.config.boundary_temperature} is inert. Raise "
                f"candidate_multiplier (e.g. 4.0) to make the temperature matter.",
                flush=True,
            )
        _, top_idx = torch.topk(scores, k=k, largest=largest)
        top_scores = scores[top_idx]

        temp = float(self.config.boundary_temperature)
        if temp <= 0.0 or top_scores.numel() == n:
            choice = torch.randperm(top_scores.numel())[:n]
        else:
            logits = top_scores / temp if largest else -top_scores / temp
            probs = torch.softmax(logits - logits.max(), dim=0)
            choice = torch.multinomial(probs, num_samples=n, replacement=n > probs.numel())
        return top_idx[choice]

    def _sample_target_boundary(self, n):
        candidates = self._num_candidates(n)
        x_phys = self._sample_uniform_states(candidates)
        with torch.no_grad():
            scores = self.residual.target_function(x_phys).abs().detach().cpu()
        idx = self._draw_from_scores(scores, n, largest=False)
        return self._pack(x_phys[idx], self._sample_tau(n))

    def _sample_predicted_boundary(self, model, n):
        candidates = self._num_candidates(n)
        pts = self._sample_uniform_points(candidates)
        scores = self._score_value_abs(model, pts["xt"], pts["tau_phys"].squeeze(-1))
        idx = self._draw_from_scores(scores, n, largest=False)
        return self._pack(pts["x_phys"][idx], pts["tau_phys"][idx])

    def _sample_residual_adaptive(self, model, n):
        candidates = self._num_candidates(n)
        pts = self._sample_uniform_points(candidates)
        scores = self._score_residual(model, pts["xt"], pts["tau_phys"].squeeze(-1))
        idx = self._draw_from_scores(scores, n, largest=True)
        return self._pack(pts["x_phys"][idx], pts["tau_phys"][idx])

    def _sample_hard_psi_points(self, n):
        x_phys = self._sample_uniform_states(n)
        centers = tuple(float(c) for c in self.config.hard_psi_centers)
        if not centers:
            low, high = getattr(self.dataset, "psi_bounds", (-math.pi, math.pi))
            centers = (float(low), float(high))

        center_idx = torch.randint(0, len(centers), (n,))
        center_values = torch.tensor(centers, dtype=x_phys.dtype)[center_idx].unsqueeze(-1)
        psi = center_values + float(self.config.hard_psi_std) * torch.randn(n, 1)

        if hasattr(self.dataset, "psi_bounds"):
            low, high = self.dataset.psi_bounds
            width = float(high - low)
            psi = ((psi - low) % width) + low

        x_phys[:, 2:3] = psi
        return self._pack(x_phys, self._sample_tau(n))

    def _score_value_abs(self, model, xt_cpu, tau_phys_cpu):
        was_training = model.training
        model.eval()
        vals = []
        with torch.no_grad():
            for start in range(0, xt_cpu.shape[0], int(self.config.chunk_size)):
                end = min(start + int(self.config.chunk_size), xt_cpu.shape[0])
                xt = xt_cpu[start:end].to(self.device).float()
                tau = tau_phys_cpu[start:end].to(self.device).float()
                V = model(xt)
                vals.append(V.reshape(-1).abs().detach().cpu())
        if was_training:
            model.train()
        return torch.cat(vals, dim=0)

    def _score_residual(self, model, xt_cpu, tau_phys_cpu):
        was_training = model.training
        model.eval()
        vals = []
        score_mode = str(self.config.residual_score).lower()
        for start in range(0, xt_cpu.shape[0], int(self.config.chunk_size)):
            end = min(start + int(self.config.chunk_size), xt_cpu.shape[0])
            xt = xt_cpu[start:end].to(self.device).float().requires_grad_(True)
            tau = tau_phys_cpu[start:end].to(self.device).float()
            residual = self.residual.compute_deepreach_residual(model=model, xt=xt)
            residual = torch.relu(residual) if score_mode in {"violation", "positive"} else residual.abs()
            vals.append(residual.reshape(-1).detach().cpu())
            del xt, tau, residual
        if was_training:
            model.train()
        return torch.cat(vals, dim=0)
