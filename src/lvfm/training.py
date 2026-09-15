import torch 

def causal_temporal_loss(
    residual,
    tau_phys,
    T,
    num_chunks=16,
    eps=1.0,
    normalize_losses=True,
):

    residual = residual.reshape(-1)
    tau_phys = tau_phys.reshape(-1).detach()

    point_losses = residual.pow(2)

    edges = torch.linspace(
        0.0,
        float(T),
        steps=num_chunks + 1,
        device=residual.device,
        dtype=tau_phys.dtype,
    )

    chunk_losses = []

    for i in range(num_chunks):
        if i == num_chunks - 1:
            mask = (tau_phys >= edges[i]) & (tau_phys <= edges[i + 1])
        else:
            mask = (tau_phys >= edges[i]) & (tau_phys < edges[i + 1])

        if mask.any():
            chunk_losses.append(point_losses[mask].mean())
        else:
            # Empty chunk contributes no direct loss.
            chunk_losses.append(torch.zeros((), device=residual.device))

    chunk_losses = torch.stack(chunk_losses)

    # Build weights without backpropagating through the weights.
    with torch.no_grad():
        detached = chunk_losses.detach()

        if normalize_losses:
            detached = detached / detached.mean().clamp_min(1e-8)

        cumulative_previous = torch.cat(
            [
                torch.zeros(1, device=residual.device),
                torch.cumsum(detached, dim=0)[:-1],
            ],
            dim=0,
        )

        weights = torch.exp(-float(eps) * cumulative_previous)

    loss = (weights * chunk_losses).sum() / weights.sum().clamp_min(1e-8)

    return loss, weights, chunk_losses


def causal_binned_loss(
    point_losses,
    tau_phys,
    T,
    num_bins=10,
    causal_epsilon=5.0,
    min_weight=0.05,
):
    point_losses = point_losses.reshape(-1)
    tau_phys = tau_phys.reshape(-1).detach()

    edges = torch.linspace(
        0.0,
        float(T),
        steps=int(num_bins) + 1,
        device=point_losses.device,
        dtype=tau_phys.dtype,
    )

    bin_losses = []
    for i in range(int(num_bins)):
        if i == int(num_bins) - 1:
            mask = (tau_phys >= edges[i]) & (tau_phys <= edges[i + 1])
        else:
            mask = (tau_phys >= edges[i]) & (tau_phys < edges[i + 1])

        if mask.any():
            bin_losses.append(point_losses[mask].mean())
        else:
            bin_losses.append(torch.zeros((), device=point_losses.device))

    bin_losses = torch.stack(bin_losses)

    with torch.no_grad():
        cumulative_prev = torch.cat(
            [
                torch.zeros(1, device=point_losses.device),
                torch.cumsum(bin_losses.detach(), dim=0)[:-1],
            ],
            dim=0,
        )
        weights = torch.exp(-float(causal_epsilon) * cumulative_prev)
        weights = weights.clamp_min(float(min_weight))

    loss = (weights * bin_losses).sum() / weights.sum().clamp_min(1e-8)
    return loss, weights, bin_losses
    








