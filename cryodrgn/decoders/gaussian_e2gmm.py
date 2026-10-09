"""
E2GMM-inspired Gaussian primitive decoder for cryoDRGN.

Key difference from the previous implementation
-----------------------------------------------
The previous decoder used a separate shared learnable Gaussian cloud plus
z-dependent residual offsets.  This version makes the parameter generator
predict the *absolute* Gaussian parameters for each conformation:

    z -> {mu_j(z), a_j(z), sigma_j(z)}_{j=1}^G

This is closer to the E2GMM-style decoder parameterization while retaining
cryoDRGN's encoder, pose/CTF handling, Fourier/Hartley training path, and
Decoder interface.

cryoDRGN lattice coordinates are frequencies in cycles / pixel, while Gaussian
centers and widths are represented in real-space pixel units.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cryodrgn import fft
from cryodrgn.decoders.base import Decoder

Norm = Sequence[Any]


def _inverse_softplus(x: Tensor) -> Tensor:
    """Numerically stable inverse of softplus for positive x."""
    return x + torch.log(-torch.expm1(-x))


class _GaussianParamGenerator(nn.Module):
    """
    MLP mapping latent z directly to per-Gaussian absolute parameters.

    For each Gaussian j the network outputs five unconstrained values:
        (center_x, center_y, center_z, amplitude_logit, sigma_logit)

    Physical constraints are applied by GaussianDecoder.gaussian_parameters().
    """

    def __init__(
        self,
        zdim: int,
        n_gaussians: int,
        hidden_dim: int = 256,
        nlayers: int = 3,
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        if zdim <= 0:
            raise ValueError("zdim must be > 0 for _GaussianParamGenerator")
        if n_gaussians <= 0:
            raise ValueError("n_gaussians must be > 0")
        if nlayers < 1:
            raise ValueError("nlayers must be >= 1")

        layers: list[nn.Module] = [
            nn.Linear(zdim, hidden_dim),
            activation(),
        ]
        for _ in range(nlayers - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    activation(),
                ]
            )

        self.output = nn.Linear(hidden_dim, n_gaussians * 5)
        layers.append(self.output)
        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class GaussianDecoder(Decoder):
    """
    E2GMM-inspired heterogeneous Gaussian decoder compatible with cryoDRGN.

    Input
    -----
    lattice:
        Training path: shape (B, N, 3 + zdim)
        eval_volume path: shape (N, 3 + zdim)

        The first 3 values are rotated Fourier coordinates in cycles/pixel.
        The remaining values are the latent conformation z, repeated at every
        Fourier coordinate by HetOnlyVAE.cat_z().

    Output
    ------
    Hartley-space values with shape (B, N), or (N,) for unbatched input.

    Parameterization
    ----------------
    For each conformation z, the generator predicts absolute parameters

        mu_j(z)     : center in real-space pixels
        a_j(z)      : positive amplitude
        sigma_j(z)  : positive isotropic width in real-space pixels

    There is no separate trainable shared base Gaussian cloud.  The final-layer
    bias is initialized to a spread-out consensus-like GMM only to provide a
    stable starting point; after initialization, all Gaussian parameters are
    produced directly by the z-conditioned generator.
    """

    def __init__(
        self,
        zdim: int,
        D: int,
        n_gaussians: int = 512,
        hidden_dim: int = 256,
        nlayers: int = 3,
        sigma: float = 1.5,
        activation: Type[nn.Module] = nn.ReLU,
        chunk_size: int = 256,
        init_radius_fraction: float = 0.60,
        min_sigma: float = 0.20,
    ) -> None:
        super().__init__()

        if D < 3:
            raise ValueError(f"D must be >= 3, got {D}")
        if n_gaussians <= 0:
            raise ValueError("n_gaussians must be > 0")
        if sigma <= min_sigma:
            raise ValueError("sigma must be > min_sigma")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if not (0.0 < init_radius_fraction < 1.0):
            raise ValueError("init_radius_fraction must lie in (0, 1)")

        self.zdim = int(zdim)
        self.D = int(D)
        self.n_gaussians = int(n_gaussians)
        self.init_sigma = float(sigma)
        self.min_sigma = float(min_sigma)
        self.chunk_size = int(chunk_size)

        # cryoDRGN evaluates an odd Fourier lattice and removes the duplicated
        # positive-frequency endpoint before inverse transforming.
        self.box_size = D - 1
        self.half_box = self.box_size / 2.0

        if self.zdim <= 0:
            raise ValueError(
                "This E2GMM-inspired parameterization requires zdim > 0. "
                "Use the original fixed/consensus Gaussian decoder for zdim=0."
            )

        self.generator = _GaussianParamGenerator(
            zdim=self.zdim,
            n_gaussians=self.n_gaussians,
            hidden_dim=hidden_dim,
            nlayers=nlayers,
            activation=activation,
        )

        # A single global amplitude scale keeps the initial total density O(1).
        self.log_global_amp_scale = nn.Parameter(
            torch.tensor(-math.log(float(n_gaussians)), dtype=torch.float32)
        )

        self._initialize_generator_output(init_radius_fraction)

    def _initialize_generator_output(self, init_radius_fraction: float) -> None:
        """
        Initialize the direct generator to output a spread-out consensus-like GMM
        for every z at step 0, without introducing separate base-center params.
        """
        directions = torch.randn(self.n_gaussians, 3)
        directions = directions / directions.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        radii = torch.rand(self.n_gaussians, 1).pow(1.0 / 3.0)
        radii = radii * init_radius_fraction * self.half_box
        init_centers = directions * radii

        normalized = (init_centers / self.half_box).clamp(-0.98, 0.98)
        center_logits = torch.atanh(normalized)

        amp_logits = torch.zeros(self.n_gaussians)
        sigma_target = torch.full(
            (self.n_gaussians,),
            self.init_sigma - self.min_sigma,
        )
        sigma_logits = _inverse_softplus(sigma_target)

        bias = torch.cat(
            [
                center_logits,
                amp_logits[:, None],
                sigma_logits[:, None],
            ],
            dim=-1,
        ).reshape(-1)

        # Zero weights mean all z initially map to the same reasonable GMM.
        # Unlike the previous implementation, this consensus lives only in the
        # generator bias; there is no separate shared base cloud being added.
        nn.init.zeros_(self.generator.output.weight)
        with torch.no_grad():
            self.generator.output.bias.copy_(bias)

    # ------------------------------------------------------------------
    # Gaussian parameter generation
    # ------------------------------------------------------------------

    def gaussian_parameters(
        self, z: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Generate absolute Gaussian parameters for latent codes z.

        Parameters
        ----------
        z: (B, zdim)

        Returns
        -------
        centers:    (B, G, 3), real-space pixels
        amplitudes: (B, G), positive
        sigmas:     (B, G), positive isotropic widths in pixels
        """
        if z.ndim != 2:
            raise ValueError(f"Expected z with shape (B, zdim), got {tuple(z.shape)}")
        if z.shape[-1] != self.zdim:
            raise ValueError(
                f"Expected zdim={self.zdim}, got last dimension {z.shape[-1]}"
            )

        raw = self.generator(z).view(z.shape[0], self.n_gaussians, 5)

        center_logits = raw[..., :3]
        amp_logits = raw[..., 3]
        sigma_logits = raw[..., 4]

        centers = self.half_box * torch.tanh(center_logits)
        amplitudes = (
            torch.exp(self.log_global_amp_scale) * F.softplus(amp_logits)
        )
        sigmas = self.min_sigma + F.softplus(sigma_logits)

        return centers, amplitudes, sigmas

    # ------------------------------------------------------------------
    # Fourier / Hartley rendering
    # ------------------------------------------------------------------

    def _fourier_chunk(
        self,
        coords: Tensor,
        centers: Tensor,
        amplitudes: Tensor,
        sigmas: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Evaluate complex Fourier values for one coordinate chunk.

        coords:      (B, M, 3), cycles / pixel
        centers:     (B, G, 3), pixels
        amplitudes:  (B, G)
        sigmas:      (B, G), pixels

        Returns
        -------
        real, imag: each (B, M)
        """
        phase = 2.0 * math.pi * torch.einsum(
            "bmc,bgc->bmg", coords, centers
        )

        # Per-Gaussian Fourier envelope:
        # exp(-2*pi^2*sigma_j(z)^2*||k||^2)
        k2 = torch.sum(coords * coords, dim=-1, keepdim=True)  # (B,M,1)
        sigma2 = sigmas.unsqueeze(1).pow(2)                   # (B,1,G)
        envelope = torch.exp(-2.0 * (math.pi**2) * k2 * sigma2)

        weights = amplitudes.unsqueeze(1) * envelope
        real = torch.sum(weights * torch.cos(phase), dim=-1)
        imag = -torch.sum(weights * torch.sin(phase), dim=-1)
        return real, imag

    def fourier_values(
        self,
        coords: Tensor,
        z: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Evaluate F_z(k) at arbitrary Fourier coordinates."""
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError(
                f"Expected coords with shape (B, N, 3), got {tuple(coords.shape)}"
            )

        centers, amplitudes, sigmas = self.gaussian_parameters(z)

        real_chunks = []
        imag_chunks = []
        for start in range(0, coords.shape[1], self.chunk_size):
            end = min(start + self.chunk_size, coords.shape[1])
            real, imag = self._fourier_chunk(
                coords[:, start:end], centers, amplitudes, sigmas
            )
            real_chunks.append(real)
            imag_chunks.append(imag)

        return torch.cat(real_chunks, dim=1), torch.cat(imag_chunks, dim=1)

    def forward(self, lattice: Tensor) -> Tensor:
        """Evaluate the decoder in cryoDRGN Hartley space."""
        squeeze_batch = False
        if lattice.ndim == 2:
            lattice = lattice.unsqueeze(0)
            squeeze_batch = True

        if lattice.ndim != 3:
            raise ValueError(
                "GaussianDecoder expects lattice of shape "
                "(B, N, 3+zdim) or (N, 3+zdim); "
                f"got {tuple(lattice.shape)}"
            )

        expected_dim = 3 + self.zdim
        if lattice.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected last dimension {expected_dim} "
                f"(3 coords + zdim={self.zdim}), got {lattice.shape[-1]}"
            )

        coords = lattice[..., :3]
        z = lattice[:, 0, 3:]

        real, imag = self.fourier_values(coords, z)
        hartley = real - imag

        if squeeze_batch:
            hartley = hartley.squeeze(0)
        return hartley

    # ------------------------------------------------------------------
    # Direct real-space renderer for debugging / sanity checks
    # ------------------------------------------------------------------

    @torch.no_grad()
    def direct_real_space_volume(
        self,
        z: Tensor,
        box_size: Optional[int] = None,
        voxel_chunk_size: int = 8192,
    ) -> Tensor:
        """
        Render rho_z(x) directly in real space, without Fourier/Hartley inversion.

        This is intended as a sanity check for eval_volume().  The returned
        volume has shape (box_size, box_size, box_size), defaulting to D-1.
        """
        if z.ndim == 1:
            z = z.unsqueeze(0)
        if z.shape != (1, self.zdim):
            raise ValueError(
                f"Expected one latent code with shape ({self.zdim},) or "
                f"(1,{self.zdim}), got {tuple(z.shape)}"
            )

        centers, amplitudes, sigmas = self.gaussian_parameters(z)
        centers = centers[0]
        amplitudes = amplitudes[0]
        sigmas = sigmas[0]

        B = int(box_size or self.box_size)
        half = B / 2.0
        axis = torch.arange(B, device=z.device, dtype=z.dtype) - (B - 1) / 2.0
        zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
        points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

        out = []
        for start in range(0, points.shape[0], voxel_chunk_size):
            p = points[start : start + voxel_chunk_size]
            diff = p[:, None, :] - centers[None, :, :]
            dist2 = torch.sum(diff * diff, dim=-1)
            density = torch.sum(
                amplitudes[None, :]
                * torch.exp(-dist2 / (2.0 * sigmas[None, :].pow(2))),
                dim=-1,
            )
            out.append(density)

        return torch.cat(out, dim=0).view(B, B, B)

    # ------------------------------------------------------------------
    # Volume evaluation for cryodrgn eval_vol / analyze
    # ------------------------------------------------------------------

    def eval_volume(
        self,
        coords: Tensor,
        D: int,
        extent: float,
        norm: Norm,
        zval: Optional[np.ndarray] = None,
    ) -> Tensor:
        """
        Evaluate a D x D x D Hartley lattice and inverse-transform it.

        This deliberately keeps the existing cryoDRGN-compatible evaluation
        path.  Use direct_real_space_volume() separately to test whether this
        Fourier/Hartley reconstruction is consistent with the explicit GMM.
        """
        assert not self.training, "Call model.eval() before eval_volume()"
        assert extent <= 0.5

        device = coords.device
        dtype = coords.dtype

        if zval is None:
            z = torch.zeros(self.zdim, dtype=dtype, device=device)
        else:
            if len(zval) != self.zdim:
                raise ValueError(
                    f"Expected zval of length {self.zdim}, got {len(zval)}"
                )
            z = torch.as_tensor(zval, dtype=dtype, device=device)

        vol_f = torch.zeros((D, D, D), dtype=dtype, device=device)
        z_slices = np.linspace(
            -extent, extent, D, endpoint=True, dtype=np.float32
        )

        for i, dz in enumerate(z_slices):
            x = coords + torch.tensor(
                [0.0, 0.0, float(dz)], dtype=dtype, device=device
            )
            keep = x.pow(2).sum(dim=1) <= extent**2
            x_keep = x[keep]
            x_keep = torch.cat(
                [x_keep, z.expand(x_keep.shape[0], self.zdim)], dim=-1
            )

            y = self.forward(x_keep)
            slice_ = torch.zeros(D * D, dtype=dtype, device=device)
            slice_[keep] = y
            vol_f[i] = slice_.view(D, D)

        vol_f = vol_f * norm[1] + norm[0]
        vol = fft.ihtn_center(vol_f[:-1, :-1, :-1])
        return vol
