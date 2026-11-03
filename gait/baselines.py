"""DCT features and the SR / exact LLC baselines in equations (13)–(14)."""

import math

import torch
import torch.nn.functional as F


def dct_descriptors(images, cfg):
    """8×8 orthonormal DCT-II, low-frequency 5×5, reflected boundary."""
    size = cfg["dct_patch_size"]
    frequencies = torch.arange(cfg["dct_frequencies"], device=images.device)
    locations = torch.arange(size, device=images.device)
    basis = math.sqrt(2 / size) * torch.cos(math.pi / size * frequencies[:, None]
                                          * (locations[None] + 0.5))
    basis[0] /= math.sqrt(2)
    kernels = (basis[:, None, :, None] * basis[None, :, None, :]).flatten(0, 1)[:, None]
    before, after = (size - 1) // 2, size // 2
    height, width = images.shape[-2:]
    responses = []
    for batch in images.split(cfg["image_batch_size"]):
        padded = F.pad(batch[:, None], (before, after, before, after), mode="reflect")
        response = F.conv2d(padded, kernels)
        response = F.interpolate(response, size=(height // 2, width // 2), mode="bilinear",
                                 align_corners=False, antialias=True)
        responses.append(response.flatten(2).transpose(1, 2))
    descriptors = torch.cat(responses)
    y, x = torch.meshgrid(torch.linspace(0, 1, height // 2, device=images.device),
                         torch.linspace(0, 1, width // 2, device=images.device), indexing="ij")
    xy = torch.stack((x.flatten(), y.flatten()), dim=1)
    return torch.cat((descriptors, xy.expand(len(images), -1, -1)), dim=2)


def sparse_representation(gram, cross, probe_norms, regularization, cfg):
    """FISTA for equation (13), with a maximum-coordinate KKT stopping rule."""
    lipschitz = float(torch.linalg.eigvalsh(gram)[-1])
    coefficients = torch.zeros_like(cross)
    extrapolated = coefficients.clone()
    momentum = 1.0
    trace = []
    stop_reason = "iteration_limit"
    for iteration in range(1, cfg["sr_iterations"] + 1):
        step = extrapolated - (gram @ extrapolated - cross) / lipschitz
        updated = step.sign() * (step.abs() - regularization / lipschitz).clamp_min(0)
        next_momentum = (1 + math.sqrt(1 + 4 * momentum**2)) / 2
        extrapolated = updated + (momentum - 1) / next_momentum * (updated - coefficients)
        coefficients = updated
        momentum = next_momentum
        if iteration % cfg["sr_check_every"] == 0 or iteration == cfg["sr_iterations"]:
            gradient = gram @ coefficients - cross
            kkt = torch.where(coefficients != 0,
                              (gradient + regularization * coefficients.sign()).abs(),
                              (gradient.abs() - regularization).clamp_min(0)).amax(dim=0)
            objective = (0.5 * probe_norms.sum()
                         + 0.5 * (coefficients * (gradient - cross)).sum()
                         + regularization * coefficients.abs().sum())
            trace.append({"iteration": iteration, "objective": float(objective),
                          "max_kkt_residual": float(kkt.max())})
            if float(kkt.max()) < cfg["sr_kkt_tolerance"]:
                stop_reason = "kkt_tolerance"
                break
    return coefficients, {"iterations": iteration, "stop_reason": stop_reason,
                          "lipschitz": lipschitz, "max_kkt_residual": float(kkt.max()),
                          "per_gei_kkt_residual": kkt.tolist(), "trace": trace}


def exact_llc(gram, cross, probe_norms, regularization, sigma, batch_size=16):
    """Solve the equality-constrained quadratic in equation (14), footnote 1."""
    coefficients, diagnostics = [], []
    for start in range(0, cross.shape[1], batch_size):
        xy = cross[:, start:start + batch_size].T
        yy = probe_norms[start:start + batch_size]
        distance = (gram.diag()[None] + yy[:, None] - 2 * xy).clamp_min(0).sqrt()
        # exp(distance / sigma) divided by its maximum, as specified in III-C.
        weights = ((distance - distance.max(1, keepdim=True).values) / sigma).exp()
        local_gram = gram[None] - xy[:, :, None] - xy[:, None, :] + yy[:, None, None]
        local_gram.diagonal(dim1=1, dim2=2).add_(2 * regularization * weights.square())
        ones = torch.ones_like(xy)
        raw = torch.linalg.solve(local_gram, ones)
        solution = raw / raw.sum(1, keepdim=True)
        gradient = solution @ gram - xy + 2 * regularization * weights.square() * solution
        stationarity = gradient - gradient.mean(1, keepdim=True)
        diagnostics.extend({"max_kkt_residual": float(value), "sum_constraint_residual": float(constraint)}
                           for value, constraint in zip(stationarity.abs().amax(1),
                                                        (solution.sum(1) - 1).abs()))
        coefficients.append(solution.T)
    return torch.cat(coefficients, dim=1), diagnostics
