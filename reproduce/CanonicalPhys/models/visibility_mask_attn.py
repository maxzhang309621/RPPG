"""VisibilityMaskAttn: deterministic visibility-aware spatial-mask attention.

This replaces the v1 LandmarkAttn module. The new module:
  - Has exactly 4 trainable scalars (3 region weights + 1 floor).
  - Has no learnable softmax over landmarks (degenerate-mouth-blob
    collapse is structurally impossible).
  - Computes per-landmark visibility from MediaPipe's facial transformation
    matrix instead of from a flatlining gating MLP.
  - Suppresses non-skin features toward a learnable floor instead of
    amplifying a fixed prior.

See docs/superpowers/specs/2026-04-08-visibility-mask-attn-design.md.
"""

import torch
import torch.nn as nn


def compute_visibility(R, n_canon, sharpness=8.0, bias=0.55):
    """Per-landmark visibility from rotation matrix and canonical normals.

    Args:
        R:        (B, T, 3, 3) per-frame rotation matrix (the upper-left
                  3x3 of MediaPipe's facial transformation matrix).
        n_canon:  (K, 3) canonical outward unit normals per landmark.
        sharpness: scalar controlling the sigmoid steepness (default 8.0).
        bias:     constant added to (R @ n)_z before the sigmoid. The
                  centroid-based canonical normals used in this project
                  have a slight back-bias for landmarks that don't sit on
                  the front of the face (e.g. jaw vertices used in cheek
                  polygons), so a bias of ~0.55 brings frontal-pose
                  visibility for those landmarks to ~0.5 while keeping
                  forehead landmarks at ~1.0 and giving a clear yaw
                  asymmetry. Set bias=0 to use the raw formula.

    Returns:
        (B, T, K) visibility values in (0, 1).
    """
    # rotated[b, t, k, i] = sum_j R[b, t, i, j] * n_canon[k, j]
    rotated = torch.einsum('btij,kj->btki', R, n_canon)
    z_component = rotated[..., 2]                          # (B, T, K)
    return torch.sigmoid(sharpness * (z_component + bias))


def signed_distance_to_convex_polygon(grid_xy, poly_verts):
    """Signed half-plane distance from grid cells to a convex polygon.

    Auto-detects the polygon's winding direction (CW or CCW in image-y-down
    coordinates) by computing the signed shoelace area, and flips the
    outward normals if needed. Returns negative distances for points
    inside the polygon, positive outside. The polygon must be convex
    (non-convex shapes will give wrong sign at concave corners).

    Args:
        grid_xy:    (H, W, 2) cell coordinates in [0, 1].
        poly_verts: (B, T, P, 2) per-frame polygon vertices.

    Returns:
        (B, T, H, W) signed distance.
    """
    B, T, P, _ = poly_verts.shape
    H, W, _ = grid_xy.shape

    # Edge vectors
    next_verts = torch.roll(poly_verts, shifts=-1, dims=2)         # (B, T, P, 2)
    edges = next_verts - poly_verts                                 # (B, T, P, 2)

    # Outward normals assuming CW in image-y-down: rotate edge by -90deg
    # in math (which is +90deg in image-y-down), giving (dy, -dx).
    normals = torch.stack([edges[..., 1], -edges[..., 0]], dim=-1)  # (B, T, P, 2)
    normals = normals / (torch.linalg.norm(normals, dim=-1,
                                           keepdim=True) + 1e-8)

    # Auto-detect winding via signed shoelace area. For a CW polygon in
    # image-y-down (= CCW in math-y-up), the signed area is positive.
    # If signed_area is negative, flip the normals so they point outward.
    # signed_area = 0.5 * sum_i (x_i * y_{i+1} - x_{i+1} * y_i), shape (B, T)
    x = poly_verts[..., 0]
    y = poly_verts[..., 1]
    next_x = next_verts[..., 0]
    next_y = next_verts[..., 1]
    signed_area = 0.5 * (x * next_y - next_x * y).sum(dim=-1)      # (B, T)
    # If signed_area < 0, the polygon is CCW (in image-y-down) -> flip normals.
    sign = torch.where(signed_area >= 0,
                       torch.ones_like(signed_area),
                       -torch.ones_like(signed_area))               # (B, T)
    normals = normals * sign.view(B, T, 1, 1)                       # broadcast over P, 2

    # Broadcast: grid -> (1, 1, 1, H, W, 2), verts/normals -> (B, T, P, 1, 1, 2)
    g = grid_xy.view(1, 1, 1, H, W, 2)
    v = poly_verts.view(B, T, P, 1, 1, 2)
    n = normals.view(B, T, P, 1, 1, 2)

    diff = g - v                                                   # (B, T, P, H, W, 2)
    signed_per_edge = (diff * n).sum(dim=-1)                       # (B, T, P, H, W)
    return signed_per_edge.max(dim=2).values                       # (B, T, H, W)


def soft_rasterize_polygon(poly_verts, grid_xy, raster_temp=50.0):
    """Soft membership map for a convex polygon (higher inside).

    Args:
        poly_verts: (B, T, P, 2) per-frame polygon vertices in [0, 1].
        grid_xy:    (H, W, 2) grid cell centers in [0, 1].
        raster_temp: sigmoid temperature (default 50.0). Higher values give
                     sharper polygon boundaries. At raster_temp=50, a point
                     0.2 units inside the polygon reads ~1.0 and a point
                     0.3 units outside reads ~0.0.

    Returns:
        (B, T, H, W) soft mask in (0, 1).
    """
    sdf = signed_distance_to_convex_polygon(grid_xy, poly_verts)
    return torch.sigmoid(-raster_temp * sdf)


# Anatomical landmark index lists for cheek polygons.
# Lists are in CW order in image coordinates (image-y-down), so the polygon
# interior is on the right of each edge — which is the convention required
# by signed_distance_to_convex_polygon.
#
# Subject-right cheek (image left of a frontal face):
RIGHT_CHEEK_IDX = [1, 2, 3, 4, 31, 41, 36]
# Subject-left cheek (image right of a frontal face):
LEFT_CHEEK_IDX = [15, 14, 13, 12, 35, 46, 45]
# Brow indices: subject-right brow 17..21, subject-left brow 22..26.
BROW_IDX = list(range(17, 27))
JAW_IDX = list(range(0, 17))


def build_forehead_polygon(landmarks):
    """Build a convex 4-vertex forehead rectangle from brow + jaw landmarks.

    The polygon is bounded below by brow_y_min (the topmost brow point)
    and extended upward by half the brow-to-jaw distance, clamped to
    y >= 0. Walking the four vertices top-left -> top-right -> bottom-right
    -> bottom-left is clockwise in image-y-down coordinates, so the
    polygon interior is on the right of each edge (matches the SDF
    helper's convention).

    Note: the previous version of this function tried to walk the curved
    brow line as the polygon's bottom edge, which produced a non-convex
    12-vertex polygon. The half-plane SDF only works for strictly convex
    polygons, so the rasterized mask was essentially zero. The simpler
    rectangle is convex by construction.

    Args:
        landmarks: (B, T, 68, 2) in normalized [0, 1].

    Returns:
        (B, T, 4, 2) polygon vertices.
    """
    brow = landmarks[:, :, BROW_IDX, :]    # (B, T, 10, 2)
    jaw  = landmarks[:, :, JAW_IDX,  :]    # (B, T, 17, 2)
    brow_y_min = brow[..., 1].min(dim=-1, keepdim=True).values   # (B, T, 1)
    jaw_y_max  = jaw[..., 1].max(dim=-1, keepdim=True).values    # (B, T, 1)
    face_h = jaw_y_max - brow_y_min                              # (B, T, 1)
    forehead_top_y = (brow_y_min - 0.5 * face_h).clamp(min=0.0)  # (B, T, 1)

    left_x  = brow[..., 0].min(dim=-1, keepdim=True).values      # (B, T, 1)
    right_x = brow[..., 0].max(dim=-1, keepdim=True).values      # (B, T, 1)

    # Walk CW in image-y-down: top-left, top-right, bottom-right, bottom-left.
    top_left     = torch.cat([left_x,  forehead_top_y], dim=-1).unsqueeze(-2)
    top_right    = torch.cat([right_x, forehead_top_y], dim=-1).unsqueeze(-2)
    bottom_right = torch.cat([right_x, brow_y_min     ], dim=-1).unsqueeze(-2)
    bottom_left  = torch.cat([left_x,  brow_y_min     ], dim=-1).unsqueeze(-2)
    polygon = torch.cat([top_left, top_right, bottom_right, bottom_left], dim=2)
    return polygon


def build_cheek_polygon(landmarks, idx_list):
    """Slice cheek polygon vertices from a fixed dlib index list.

    Args:
        landmarks: (B, T, 68, 2)
        idx_list: list of K landmark indices in CW order (image coords).

    Returns:
        (B, T, K, 2)
    """
    return landmarks[:, :, idx_list, :]


import math
import os

import numpy as np


def _logit(p):
    return math.log(p / (1.0 - p))


class VisibilityMaskAttn(nn.Module):
    """Visibility-aware spatial-mask attention.

    The whole module has exactly 4 trainable scalars (region_logits[3]
    + alpha_logit[1]). Pose dependence enters via per-landmark visibility
    computed from MediaPipe's facial transformation matrix.

    See docs/superpowers/specs/2026-04-08-visibility-mask-attn-design.md
    for the full motivation.

    Args:
        canonical_normals_path: path to a (68, 3) float32 .npy file
            produced by scripts/precompute_canonical_normals.py.
        spatial_size: side length of the output mask (default 13, must
            match the backbone voxel embedding spatial dim).
        alpha_init: initial floor value for the fusion (default 0.3).
        visibility_sharpness: sigmoid sharpness for the visibility
            computation (default 8.0).
        visibility_bias: constant added to (R @ n)_z before the sigmoid
            in the visibility computation (default 0.55, calibrates
            cheek polygons to ~0.5 visibility at frontal pose).
        raster_temp: sigmoid temperature for the polygon rasterization
            (default 50.0).
        mask_dropout: probability that each spatial cell of the skin mask
            is zeroed during training (default 0.0, disabled). At eval
            time the mask is always used unmodified.
    """

    def __init__(self,
                 canonical_normals_path,
                 spatial_size=13,
                 alpha_init=0.3,
                 visibility_sharpness=8.0,
                 visibility_bias=0.55,
                 raster_temp=50.0,
                 mask_dropout=0.0):
        super().__init__()
        self.spatial_size = spatial_size
        self.visibility_sharpness = float(visibility_sharpness)
        self.visibility_bias = float(visibility_bias)
        self.raster_temp = float(raster_temp)
        self.mask_dropout = float(mask_dropout)

        if not os.path.isabs(canonical_normals_path):
            here = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(here)
            canonical_normals_path = os.path.join(project_root,
                                                  canonical_normals_path)
        n_canon_np = np.load(canonical_normals_path).astype(np.float32)
        assert n_canon_np.shape == (68, 3), \
            f"expected (68, 3), got {n_canon_np.shape}"
        self.register_buffer("n_canon", torch.from_numpy(n_canon_np))

        # Precomputed grid for soft rasterization (cell centers in [0,1]).
        cells = torch.linspace(0, 1, spatial_size)
        grid_y, grid_x = torch.meshgrid(cells, cells, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)   # (H, W, 2)
        self.register_buffer("grid_xy", grid)

        # The only trainable parameters: 3 region weights + 1 floor.
        self.region_logits = nn.Parameter(torch.zeros(3))                 # init -> sigmoid 0.5
        self.alpha_logit  = nn.Parameter(torch.tensor(_logit(alpha_init)))

    def compute_skin_mask(self, landmarks, transforms):
        """Compute the per-frame skin mask without applying it.

        Args:
            landmarks:  (B, T, 68, 2) normalized landmark coordinates.
            transforms: (B, T, 4, 4) MediaPipe facial transformation
                        matrices (we use only the upper-left 3x3 rotation
                        block).

        Returns:
            (B, T, H, W) skin mask in (0, 1).
        """
        B, T = landmarks.shape[:2]
        R = transforms[..., :3, :3]                       # (B, T, 3, 3)

        # Per-landmark visibility (B, T, 68)
        v = compute_visibility(R, self.n_canon,
                               sharpness=self.visibility_sharpness,
                               bias=self.visibility_bias)

        # Forehead polygon: (B, T, 12, 2)
        fh_poly = build_forehead_polygon(landmarks)
        v_fh = v[..., BROW_IDX].mean(dim=-1)              # (B, T)

        # Subject-right cheek polygon
        rc_poly = build_cheek_polygon(landmarks, RIGHT_CHEEK_IDX)
        v_rc = v[..., RIGHT_CHEEK_IDX].mean(dim=-1)       # (B, T)

        # Subject-left cheek polygon
        lc_poly = build_cheek_polygon(landmarks, LEFT_CHEEK_IDX)
        v_lc = v[..., LEFT_CHEEK_IDX].mean(dim=-1)        # (B, T)

        # Soft-rasterize each polygon to (B, T, H, W).
        m_fh = soft_rasterize_polygon(fh_poly, self.grid_xy, self.raster_temp)
        m_rc = soft_rasterize_polygon(rc_poly, self.grid_xy, self.raster_temp)
        m_lc = soft_rasterize_polygon(lc_poly, self.grid_xy, self.raster_temp)

        # Per-region weights in [0, 1].
        w = torch.sigmoid(self.region_logits)             # (3,)
        w_fh, w_rc, w_lc = w[0], w[1], w[2]

        # Visibility-weighted region masks. Each (B, T, H, W).
        weighted = torch.stack([
            (w_fh * v_fh).unsqueeze(-1).unsqueeze(-1) * m_fh,
            (w_rc * v_rc).unsqueeze(-1).unsqueeze(-1) * m_rc,
            (w_lc * v_lc).unsqueeze(-1).unsqueeze(-1) * m_lc,
        ], dim=0)                                         # (3, B, T, H, W)

        skin_mask = weighted.max(dim=0).values            # (B, T, H, W)
        return skin_mask.clamp(0.0, 1.0)

    def forward(self, voxel_embeddings, landmarks, transforms,
                valid_mask=None):
        """Apply visibility-aware spatial-mask attention.

        Args:
            voxel_embeddings: (B, C, T, H, W) backbone features.
            landmarks:        (B, T, 68, 2) normalized landmark coords.
            transforms:       (B, T, 4, 4) MediaPipe facial transformation
                              matrices.
            valid_mask:       (B, T) float in {0, 1}. When provided,
                              invalid frames see an all-ones skin mask, so
                              the modulation collapses to identity.

        Returns:
            enhanced: (B, C, T, H, W).
        """
        skin_mask = self.compute_skin_mask(landmarks, transforms)   # (B, T, H, W)

        if valid_mask is not None:
            vm = valid_mask.to(skin_mask.dtype).unsqueeze(-1).unsqueeze(-1)
            skin_mask = skin_mask * vm + (1.0 - vm)

        # Spatial dropout: randomly zero out mask cells during training.
        if self.mask_dropout > 0.0 and self.training:
            drop_mask = torch.bernoulli(
                torch.full_like(skin_mask, 1.0 - self.mask_dropout))
            skin_mask = skin_mask * drop_mask

        alpha = torch.sigmoid(self.alpha_logit)
        # Reshape skin_mask to (B, 1, T, H, W) for broadcasting against
        # (B, C, T, H, W) voxel_embeddings.
        skin_mask_5d = skin_mask.unsqueeze(1)
        return voxel_embeddings * (alpha + (1.0 - alpha) * skin_mask_5d)
