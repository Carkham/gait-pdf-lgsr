"""Equations (1)–(4): Gabor descriptors, diagonal UBM, and MAP encoding."""

import math

import torch
import torch.nn.functional as F


def gabor_descriptors(images, cfg):
    """images: N x H x W in [0,1]. Returns N x (H/2 W/2) x 42.

    Explicit conventions: complex magnitude, zero padding, finite 3-sigma
    kernels, antialiased bilinear downsampling, XY in [0,1].
    """
    height, width = images.shape[-2:]
    delta = 2 * math.pi
    radius = math.ceil(cfg["gabor_sigma_extent"] * delta /
                       (math.pi * 2 ** (-(max(cfg["gabor_scales"]) + 2) / 2)))
    axis = torch.arange(-radius, radius + 1, device=images.device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    kernels = []
    for scale in cfg["gabor_scales"]:
        frequency = math.pi * 2 ** (-(scale + 2) / 2)
        envelope = frequency**2 / delta**2 * torch.exp(
            -frequency**2 * (xx**2 + yy**2) / (2 * delta**2))
        support = (xx.abs() <= math.ceil(cfg["gabor_sigma_extent"] * delta / frequency)) & (
            yy.abs() <= math.ceil(cfg["gabor_sigma_extent"] * delta / frequency))
        for orientation in cfg["gabor_orientations"]:
            angle = math.pi * orientation / 8
            phase = frequency * (xx * math.cos(angle) + yy * math.sin(angle))
            kernels.append(envelope * support * (torch.exp(1j * phase) - math.exp(-delta**2 / 2)))
    size = [2 ** math.ceil(math.log2(n + 2 * radius)) for n in (height, width)]
    bank = torch.fft.fft2(torch.stack(kernels), s=size)
    descriptors = []
    for batch in images.split(cfg["image_batch_size"]):
        spectrum = torch.fft.fft2(batch, s=size)
        response = torch.fft.ifft2(spectrum[:, None] * bank[None])
        response = response[..., radius:radius + height, radius:radius + width].abs()
        response = F.interpolate(response, size=(height // 2, width // 2),
                                 mode="bilinear", align_corners=False, antialias=True)
        descriptors.append(response.flatten(2).transpose(1, 2))
    patches = torch.cat(descriptors)
    if cfg["use_xy"]:
        y, x = torch.meshgrid(torch.linspace(0, 1, height // 2, device=images.device),
                              torch.linspace(0, 1, width // 2, device=images.device), indexing="ij")
        xy = torch.stack((x.flatten(), y.flatten()), dim=1)
        patches = torch.cat((patches, xy.expand(len(images), -1, -1)), dim=2)
    return patches


def log_prob(x, model):
    inverse = model["variances"].reciprocal()
    means = model["means"]
    squared = x.square() @ inverse.T - 2 * x @ (means * inverse).T
    squared += (means.square() * inverse).sum(1)
    normalizer = x.shape[1] * math.log(2 * math.pi) + model["variances"].log().sum(1)
    return model["weights"].log()[None] - 0.5 * (squared + normalizer)


def fit_ubm(patches, cfg, log):
    """Full gallery EM on GPU; no patch subsampling. Float64 sufficient statistics."""
    x = patches.flatten(0, 1).double()
    count, dimension = x.shape
    k = cfg["components"]
    batch_size = cfg["patch_batch_size"]
    means = x[torch.randperm(count, device=x.device)[:k]].clone()
    history = {"kmeans": [], "em": []}
    # Lloyd updates. Empty clusters are a known case with repeated background
    # descriptors; reseed them from the largest current quantization errors.
    for iteration in range(cfg["kmeans_iterations"]):
        counts = x.new_zeros(k)
        sums = x.new_zeros((k, dimension))
        squares = x.new_zeros((k, dimension))
        distances = []
        for batch in x.split(batch_size):
            distance = (batch.square().sum(1, keepdim=True) + means.square().sum(1)
                        - 2 * batch @ means.T).clamp_min(0)
            error, labels = distance.min(1)
            distances.append(error)
            counts += torch.bincount(labels, minlength=k)
            sums.index_add_(0, labels, batch)
            squares.index_add_(0, labels, batch.square())
        means = sums / counts.clamp_min(1)[:, None]
        variances = (squares / counts.clamp_min(1)[:, None] - means.square()).clamp_min(cfg["variance_floor"])
        empty = counts == 0
        empty_count = int(empty.sum())
        if empty_count:
            means[empty] = x[torch.cat(distances).topk(empty_count).indices]
            variances[empty] = x.var(0, correction=0).clamp_min(cfg["variance_floor"])
        loss = float(torch.cat(distances).mean())
        history["kmeans"].append({"iteration": iteration, "distortion": loss, "empty_clusters": empty_count})
        log({"ubm/kmeans_distortion": loss, "ubm/kmeans_empty_clusters": empty_count})
    model = {"means": means, "variances": variances, "weights": x.new_full((k,), 1 / k)}
    previous = None
    converged = False
    for iteration in range(cfg["em_iterations"]):
        counts = x.new_zeros(k)
        sums = x.new_zeros((k, dimension))
        squares = x.new_zeros((k, dimension))
        likelihood = x.new_zeros(())
        for batch in x.split(batch_size):
            probability = log_prob(batch, model)
            log_density = probability.logsumexp(1)
            responsibility = (probability - log_density[:, None]).exp()
            counts += responsibility.sum(0)
            sums += responsibility.T @ batch
            squares += responsibility.T @ batch.square()
            likelihood += log_density.sum()
        value = float(likelihood / count)
        history["em"].append({"iteration": iteration, "mean_log_likelihood": value})
        log({"ubm/mean_log_likelihood": value})
        if previous is not None and abs(value - previous) < cfg["em_tolerance"]:
            converged = True
            break
        means = sums / counts[:, None]
        model = {"means": means,
                 "variances": (squares / counts[:, None] - means.square()).clamp_min(cfg["variance_floor"]),
                 "weights": counts / count}
        previous = value
    history["em_reached_tolerance"] = converged
    return model, history


def map_encode(patches, model, relevance, batch_size):
    result = []
    for batch in patches.split(batch_size):
        batch = batch.double()
        n, h, d = batch.shape
        posterior = log_prob(batch.reshape(-1, d), model).softmax(1).reshape(n, h, -1)
        occupancy = posterior.sum(1)
        first = posterior.transpose(1, 2) @ batch
        adapted = (first + relevance * model["means"]) / (occupancy[..., None] + relevance)
        vector = (occupancy[..., None] / (2 * h)).sqrt() * (adapted - model["means"]) / model["variances"].sqrt()
        vector = vector.flatten(1)
        vector -= vector.mean(1, keepdim=True)
        vector /= vector.norm(dim=1, keepdim=True)
        result.append(vector.float())
    return torch.cat(result)
