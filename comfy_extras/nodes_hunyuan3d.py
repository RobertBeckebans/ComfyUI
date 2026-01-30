import torch
import os
import json
import struct
import numpy as np
from comfy.ldm.modules.diffusionmodules.mmdit import get_1d_sincos_pos_embed_from_grid_torch
import folder_paths
import comfy.model_management
from comfy.cli_args import args
from typing_extensions import override
from comfy_api.latest import ComfyExtension, IO, Types
from comfy_api.latest._util import MESH, VOXEL  # only for backward compatibility if someone import it from this file (will be removed later) # noqa


class EmptyLatentHunyuan3Dv2(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EmptyLatentHunyuan3Dv2",
            category="latent/3d",
            inputs=[
                IO.Int.Input("resolution", default=3072, min=1, max=8192),
                IO.Int.Input("batch_size", default=1, min=1, max=4096, tooltip="The number of latent images in the batch."),
            ],
            outputs=[
                IO.Latent.Output(),
            ]
        )

    @classmethod
    def execute(cls, resolution, batch_size) -> IO.NodeOutput:
        latent = torch.zeros([batch_size, 64, resolution], device=comfy.model_management.intermediate_device())
        return IO.NodeOutput({"samples": latent, "type": "hunyuan3dv2"})

    generate = execute  # TODO: remove


class Hunyuan3Dv2Conditioning(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Hunyuan3Dv2Conditioning",
            category="conditioning/video_models",
            inputs=[
                IO.ClipVisionOutput.Input("clip_vision_output"),
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
            ]
        )

    @classmethod
    def execute(cls, clip_vision_output) -> IO.NodeOutput:
        embeds = clip_vision_output.last_hidden_state
        positive = [[embeds, {}]]
        negative = [[torch.zeros_like(embeds), {}]]
        return IO.NodeOutput(positive, negative)

    encode = execute  # TODO: remove


class Hunyuan3Dv2ConditioningMultiView(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Hunyuan3Dv2ConditioningMultiView",
            category="conditioning/video_models",
            inputs=[
                IO.ClipVisionOutput.Input("front", optional=True),
                IO.ClipVisionOutput.Input("left", optional=True),
                IO.ClipVisionOutput.Input("back", optional=True),
                IO.ClipVisionOutput.Input("right", optional=True),
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
            ]
        )

    @classmethod
    def execute(cls, front=None, left=None, back=None, right=None) -> IO.NodeOutput:
        all_embeds = [front, left, back, right]
        out = []
        pos_embeds = None
        for i, e in enumerate(all_embeds):
            if e is not None:
                if pos_embeds is None:
                    pos_embeds = get_1d_sincos_pos_embed_from_grid_torch(e.last_hidden_state.shape[-1], torch.arange(4))
                out.append(e.last_hidden_state + pos_embeds[i].reshape(1, 1, -1))

        embeds = torch.cat(out, dim=1)
        positive = [[embeds, {}]]
        negative = [[torch.zeros_like(embeds), {}]]
        return IO.NodeOutput(positive, negative)

    encode = execute  # TODO: remove


class VAEDecodeHunyuan3D(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VAEDecodeHunyuan3D",
            category="latent/3d",
            inputs=[
                IO.Latent.Input("samples"),
                IO.Vae.Input("vae"),
                IO.Int.Input("num_chunks", default=8000, min=1000, max=500000),
                IO.Int.Input("octree_resolution", default=256, min=16, max=512),
            ],
            outputs=[
                IO.Voxel.Output(),
            ]
        )

    @classmethod
    def execute(cls, vae, samples, num_chunks, octree_resolution) -> IO.NodeOutput:
        voxels = Types.VOXEL(vae.decode(samples["samples"], vae_options={"num_chunks": num_chunks, "octree_resolution": octree_resolution}))
        return IO.NodeOutput(voxels)

    decode = execute  # TODO: remove


def voxel_to_mesh(voxels, threshold=0.5, device=None):
    if device is None:
        device = torch.device("cpu")
    voxels = voxels.to(device)

    binary = (voxels > threshold).float()
    padded = torch.nn.functional.pad(binary, (1, 1, 1, 1, 1, 1), 'constant', 0)

    D, H, W = binary.shape

    neighbors = torch.tensor([
        [0, 0, 1],
        [0, 0, -1],
        [0, 1, 0],
        [0, -1, 0],
        [1, 0, 0],
        [-1, 0, 0]
    ], device=device)

    z, y, x = torch.meshgrid(
        torch.arange(D, device=device),
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    voxel_indices = torch.stack([z.flatten(), y.flatten(), x.flatten()], dim=1)

    solid_mask = binary.flatten() > 0
    solid_indices = voxel_indices[solid_mask]

    corner_offsets = [
        torch.tensor([
            [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]
        ], device=device),
        torch.tensor([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
        ], device=device),
        torch.tensor([
            [0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]
        ], device=device),
        torch.tensor([
            [0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]
        ], device=device),
        torch.tensor([
            [1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0]
        ], device=device),
        torch.tensor([
            [0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 0]
        ], device=device)
    ]

    all_vertices = []
    all_indices = []

    vertex_count = 0

    for face_idx, offset in enumerate(neighbors):
        neighbor_indices = solid_indices + offset

        padded_indices = neighbor_indices + 1

        is_exposed = padded[
            padded_indices[:, 0],
            padded_indices[:, 1],
            padded_indices[:, 2]
        ] == 0

        if not is_exposed.any():
            continue

        exposed_indices = solid_indices[is_exposed]

        corners = corner_offsets[face_idx].unsqueeze(0)

        face_vertices = exposed_indices.unsqueeze(1) + corners

        all_vertices.append(face_vertices.reshape(-1, 3))

        num_faces = exposed_indices.shape[0]
        face_indices = torch.arange(
            vertex_count,
            vertex_count + 4 * num_faces,
            device=device
        ).reshape(-1, 4)

        all_indices.append(torch.stack([face_indices[:, 0], face_indices[:, 1], face_indices[:, 2]], dim=1))
        all_indices.append(torch.stack([face_indices[:, 0], face_indices[:, 2], face_indices[:, 3]], dim=1))

        vertex_count += 4 * num_faces

    if len(all_vertices) > 0:
        vertices = torch.cat(all_vertices, dim=0)
        faces = torch.cat(all_indices, dim=0)
    else:
        vertices = torch.zeros((1, 3))
        faces = torch.zeros((1, 3))

    v_min = 0
    v_max = max(voxels.shape)

    vertices = vertices - (v_min + v_max) / 2

    scale = (v_max - v_min) / 2
    if scale > 0:
        vertices = vertices / scale

    vertices = torch.fliplr(vertices)
    return vertices, faces

def voxel_to_mesh_surfnet(voxels, threshold=0.5, device=None):
    if device is None:
        device = torch.device("cpu")
    voxels = voxels.to(device)

    D, H, W = voxels.shape

    padded = torch.nn.functional.pad(voxels, (1, 1, 1, 1, 1, 1), 'constant', 0)
    z, y, x = torch.meshgrid(
        torch.arange(D, device=device),
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    cell_positions = torch.stack([z.flatten(), y.flatten(), x.flatten()], dim=1)

    corner_offsets = torch.tensor([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ], device=device)

    pos = cell_positions.unsqueeze(1) + corner_offsets.unsqueeze(0)
    z_idx, y_idx, x_idx = pos.unbind(-1)
    corner_values = padded[z_idx, y_idx, x_idx]

    corner_signs = corner_values > threshold
    has_inside = torch.any(corner_signs, dim=1)
    has_outside = torch.any(~corner_signs, dim=1)
    contains_surface = has_inside & has_outside

    active_cells = cell_positions[contains_surface]
    active_signs = corner_signs[contains_surface]
    active_values = corner_values[contains_surface]

    if active_cells.shape[0] == 0:
        return torch.zeros((0, 3), device=device), torch.zeros((0, 3), dtype=torch.long, device=device)

    edges = torch.tensor([
        [0, 1], [0, 2], [0, 4], [1, 3],
        [1, 5], [2, 3], [2, 6], [3, 7],
        [4, 5], [4, 6], [5, 7], [6, 7]
    ], device=device)

    cell_vertices = {}
    progress = comfy.utils.ProgressBar(100)

    for edge_idx, (e1, e2) in enumerate(edges):
        progress.update(1)
        crossing = active_signs[:, e1] != active_signs[:, e2]
        if not crossing.any():
            continue

        cell_indices = torch.nonzero(crossing, as_tuple=True)[0]

        v1 = active_values[cell_indices, e1]
        v2 = active_values[cell_indices, e2]

        t = torch.zeros_like(v1, device=device)
        denom = v2 - v1
        valid = denom != 0
        t[valid] = (threshold - v1[valid]) / denom[valid]
        t[~valid] = 0.5

        p1 = corner_offsets[e1].float()
        p2 = corner_offsets[e2].float()

        intersection = p1.unsqueeze(0) + t.unsqueeze(1) * (p2.unsqueeze(0) - p1.unsqueeze(0))

        for i, point in zip(cell_indices.tolist(), intersection):
            if i not in cell_vertices:
                cell_vertices[i] = []
            cell_vertices[i].append(point)

    # Calculate the final vertices as the average of intersection points for each cell
    vertices = []
    vertex_lookup = {}

    vert_progress_mod = round(len(cell_vertices)/50)

    for i, points in cell_vertices.items():
        if not i % vert_progress_mod:
            progress.update(1)

        if points:
            vertex = torch.stack(points).mean(dim=0)
            vertex = vertex + active_cells[i].float()
            vertex_lookup[tuple(active_cells[i].tolist())] = len(vertices)
            vertices.append(vertex)

    if not vertices:
        return torch.zeros((0, 3), device=device), torch.zeros((0, 3), dtype=torch.long, device=device)

    final_vertices = torch.stack(vertices)

    inside_corners_mask = active_signs
    outside_corners_mask = ~active_signs

    inside_counts = inside_corners_mask.sum(dim=1, keepdim=True).float()
    outside_counts = outside_corners_mask.sum(dim=1, keepdim=True).float()

    inside_pos = torch.zeros((active_cells.shape[0], 3), device=device)
    outside_pos = torch.zeros((active_cells.shape[0], 3), device=device)

    for i in range(8):
        mask_inside = inside_corners_mask[:, i].unsqueeze(1)
        mask_outside = outside_corners_mask[:, i].unsqueeze(1)
        inside_pos += corner_offsets[i].float().unsqueeze(0) * mask_inside
        outside_pos += corner_offsets[i].float().unsqueeze(0) * mask_outside

    inside_pos /= inside_counts
    outside_pos /= outside_counts
    gradients = inside_pos - outside_pos

    pos_dirs = torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], device=device)

    cross_products = [
        torch.linalg.cross(pos_dirs[i].float(), pos_dirs[j].float())
        for i in range(3) for j in range(i+1, 3)
    ]

    faces = []
    all_keys = set(vertex_lookup.keys())

    face_progress_mod = round(len(active_cells)/38*3)

    for pair_idx, (i, j) in enumerate([(0,1), (0,2), (1,2)]):
        dir_i = pos_dirs[i]
        dir_j = pos_dirs[j]
        cross_product = cross_products[pair_idx]

        ni_positions = active_cells + dir_i
        nj_positions = active_cells + dir_j
        diag_positions = active_cells + dir_i + dir_j

        alignments = torch.matmul(gradients, cross_product)

        valid_quads = []
        quad_indices = []

        for idx, active_cell in enumerate(active_cells):
            if not idx % face_progress_mod:
                progress.update(1)
            cell_key = tuple(active_cell.tolist())
            ni_key = tuple(ni_positions[idx].tolist())
            nj_key = tuple(nj_positions[idx].tolist())
            diag_key = tuple(diag_positions[idx].tolist())

            if cell_key in all_keys and ni_key in all_keys and nj_key in all_keys and diag_key in all_keys:
                v0 = vertex_lookup[cell_key]
                v1 = vertex_lookup[ni_key]
                v2 = vertex_lookup[nj_key]
                v3 = vertex_lookup[diag_key]

                valid_quads.append((v0, v1, v2, v3))
                quad_indices.append(idx)

        for q_idx, (v0, v1, v2, v3) in enumerate(valid_quads):
            cell_idx = quad_indices[q_idx]
            if alignments[cell_idx] > 0:
                faces.append(torch.tensor([v0, v1, v3], device=device, dtype=torch.long))
                faces.append(torch.tensor([v0, v3, v2], device=device, dtype=torch.long))
            else:
                faces.append(torch.tensor([v0, v3, v1], device=device, dtype=torch.long))
                faces.append(torch.tensor([v0, v2, v3], device=device, dtype=torch.long))

    if faces:
        faces = torch.stack(faces)
    else:
        faces = torch.zeros((0, 3), dtype=torch.long, device=device)

    v_min = 0
    v_max = max(D, H, W)

    final_vertices = final_vertices - (v_min + v_max) / 2

    scale = (v_max - v_min) / 2
    if scale > 0:
        final_vertices = final_vertices / scale

    final_vertices = torch.fliplr(final_vertices)

    return final_vertices, faces


class VoxelToMeshBasic(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VoxelToMeshBasic",
            category="3d",
            inputs=[
                IO.Voxel.Input("voxel"),
                IO.Float.Input("threshold", default=0.6, min=-1.0, max=1.0, step=0.01),
            ],
            outputs=[
                IO.Mesh.Output(),
            ]
        )

    @classmethod
    def execute(cls, voxel, threshold) -> IO.NodeOutput:
        vertices = []
        faces = []
        for x in voxel.data:
            v, f = voxel_to_mesh(x, threshold=threshold, device=None)
            vertices.append(v)
            faces.append(f)

        return IO.NodeOutput(Types.MESH(torch.stack(vertices), torch.stack(faces)))

    decode = execute  # TODO: remove


class VoxelToMesh(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VoxelToMesh",
            category="3d",
            inputs=[
                IO.Voxel.Input("voxel"),
                IO.Combo.Input("algorithm", options=["surface net", "basic"]),
                IO.Float.Input("threshold", default=0.6, min=-1.0, max=1.0, step=0.01),
                IO.Combo.Input("vertex_colors", options=["none", "height", "density", "normal", "image_projection"], default="none", tooltip="Generate vertex colors for the mesh"),
                IO.Image.Input("reference_image", optional=True, tooltip="Reference image for color projection (required when vertex_colors='image_projection')"),
            ],
            outputs=[
                IO.Mesh.Output(),
            ]
        )

    @classmethod
    def execute(cls, voxel, algorithm, threshold, vertex_colors, reference_image=None) -> IO.NodeOutput:
        vertices = []
        faces = []
        colors = []

        if algorithm == "basic":
            mesh_function = voxel_to_mesh
        elif algorithm == "surface net":
            mesh_function = voxel_to_mesh_surfnet

        for x in voxel.data:
            v, f = mesh_function(x, threshold=threshold, device=None)
            vertices.append(v)
            faces.append(f)

            # Generate vertex colors if requested
            if vertex_colors != "none":
                if vertex_colors == "image_projection" and reference_image is not None:
                    # Use image projection for colors
                    c = project_image_colors_to_vertices(v, reference_image[0] if len(reference_image.shape) == 4 else reference_image)
                else:
                    # Use procedural colors
                    c = generate_vertex_colors_from_voxels(v, x, threshold=threshold, color_mode=vertex_colors)
                colors.append(c)

        mesh = Types.MESH(torch.stack(vertices), torch.stack(faces))

        # Attach vertex colors to mesh if generated
        if vertex_colors != "none" and len(colors) > 0:
            mesh.vertex_colors = torch.stack(colors)

        return IO.NodeOutput(mesh)

    decode = execute  # TODO: remove


def project_image_colors_to_vertices(vertices, image, camera_distance=2.0, fov=40.0, num_views=4):
    """
    Project colors from a reference image onto mesh vertices using simple orthographic/perspective projection.

    Parameters:
    vertices: torch.Tensor of shape (N, 3) - Vertex coordinates (normalized -1 to 1)
    image: torch.Tensor of shape (H, W, 3) - Reference image in [0, 1] range (RGB)
    camera_distance: float - Distance of camera from origin
    fov: float - Field of view in degrees (for perspective projection)
    num_views: int - Number of views to project from (1=front only, 4=front+sides)

    Returns:
    torch.Tensor of shape (N, 4) - RGBA colors as uint8 [0-255]
    """
    device = vertices.device
    num_vertices = vertices.shape[0]

    if num_vertices == 0:
        return torch.zeros((0, 4), dtype=torch.uint8, device=device)

    # Convert image to device and ensure correct format
    if image.device != device:
        image = image.to(device)

    # Image is expected to be (H, W, 3) in range [0, 1]
    img_h, img_w = image.shape[0], image.shape[1]

    # Initialize color accumulator
    vertex_colors_sum = torch.zeros((num_vertices, 3), dtype=torch.float32, device=device)
    vertex_weights = torch.zeros((num_vertices, 1), dtype=torch.float32, device=device)

    # Define camera angles for multi-view projection
    if num_views == 1:
        camera_angles = [(0, 0)]  # Front view only
    elif num_views == 4:
        camera_angles = [(0, 0), (90, 0), (180, 0), (270, 0)]  # Front + 3 sides
    else:
        # Evenly distributed around
        camera_angles = [(i * 360 / num_views, 0) for i in range(num_views)]

    for azimuth, elevation in camera_angles:
        # Rotate vertices based on camera angle
        azimuth_rad = torch.tensor(azimuth * np.pi / 180.0, device=device)
        elevation_rad = torch.tensor(elevation * np.pi / 180.0, device=device)

        # Rotation matrix (simplified - rotate around Y axis for azimuth)
        cos_a = torch.cos(azimuth_rad)
        sin_a = torch.sin(azimuth_rad)

        # Apply rotation
        v_rotated = vertices.clone()
        x_rot = vertices[:, 0] * cos_a - vertices[:, 2] * sin_a
        z_rot = vertices[:, 0] * sin_a + vertices[:, 2] * cos_a
        v_rotated[:, 0] = x_rot
        v_rotated[:, 2] = z_rot

        # Orthographic projection (simple: just take x, y coordinates)
        # Vertices are in range [-1, 1], map to image coordinates [0, W-1] and [0, H-1]
        x_proj = ((v_rotated[:, 0] + 1.0) * 0.5 * (img_w - 1)).clamp(0, img_w - 1)
        y_proj = ((1.0 - (v_rotated[:, 1] + 1.0) * 0.5) * (img_h - 1)).clamp(0, img_h - 1)  # Flip Y

        # Check which vertices are visible (positive z after rotation = facing camera)
        visible = v_rotated[:, 2] > -0.5  # Allow some tolerance

        # Sample colors from image using bilinear interpolation
        x_floor = x_proj.floor().long()
        y_floor = y_proj.floor().long()
        x_ceil = (x_floor + 1).clamp(max=img_w - 1)
        y_ceil = (y_floor + 1).clamp(max=img_h - 1)

        # Bilinear interpolation weights
        x_frac = (x_proj - x_floor.float()).unsqueeze(1)
        y_frac = (y_proj - y_floor.float()).unsqueeze(1)

        # Sample four corners
        c00 = image[y_floor, x_floor]  # (N, 3)
        c01 = image[y_floor, x_ceil]
        c10 = image[y_ceil, x_floor]
        c11 = image[y_ceil, x_ceil]

        # Bilinear interpolation
        c0 = c00 * (1 - x_frac) + c01 * x_frac
        c1 = c10 * (1 - x_frac) + c11 * x_frac
        sampled_colors = c0 * (1 - y_frac) + c1 * y_frac

        # Accumulate colors for visible vertices
        visible_mask = visible.unsqueeze(1).float()
        vertex_colors_sum += sampled_colors * visible_mask
        vertex_weights += visible_mask

    # Average colors across views
    vertex_weights = vertex_weights.clamp(min=1e-6)  # Avoid division by zero
    final_colors = vertex_colors_sum / vertex_weights

    # For vertices that weren't visible from any view, use a default color (gray)
    no_color_mask = (vertex_weights.squeeze() < 0.1)
    if no_color_mask.any():
        # Fallback: use position-based coloring for invisible vertices
        final_colors[no_color_mask] = ((vertices[no_color_mask] + 1.0) * 0.5).clamp(0, 1)

    # Convert to uint8 RGBA
    r = (final_colors[:, 0] * 255).clamp(0, 255).to(torch.uint8)
    g = (final_colors[:, 1] * 255).clamp(0, 255).to(torch.uint8)
    b = (final_colors[:, 2] * 255).clamp(0, 255).to(torch.uint8)
    a = torch.full((num_vertices,), 255, dtype=torch.uint8, device=device)

    colors = torch.stack([r, g, b, a], dim=1)
    return colors


def generate_vertex_colors_from_voxels(vertices, voxels, threshold=0.5, color_mode="height"):
    """
    Generate vertex colors for mesh vertices based on voxel data.

    Parameters:
    vertices: torch.Tensor of shape (N, 3) - The vertex coordinates (normalized -1 to 1)
    voxels: torch.Tensor of shape (D, H, W) - The voxel density data
    threshold: float - The threshold used for mesh extraction
    color_mode: str - "height" (height-based gradient), "density" (voxel density values), or "normal" (normal-based)

    Returns:
    torch.Tensor of shape (N, 4) - RGBA colors as uint8 [0-255]
    """
    device = vertices.device
    num_vertices = vertices.shape[0]

    if num_vertices == 0:
        return torch.zeros((0, 4), dtype=torch.uint8, device=device)

    if color_mode == "height":
        # Height-based gradient coloring (blue to red)
        z_coords = vertices[:, 2]  # Assuming Z is up
        z_min = z_coords.min()
        z_max = z_coords.max()

        if z_max > z_min:
            normalized_height = (z_coords - z_min) / (z_max - z_min)
        else:
            normalized_height = torch.ones_like(z_coords) * 0.5

        # Create a gradient from blue (low) to red (high)
        r = (normalized_height * 255).clamp(0, 255).to(torch.uint8)
        g = ((1.0 - torch.abs(normalized_height - 0.5) * 2.0) * 255).clamp(0, 255).to(torch.uint8)
        b = ((1.0 - normalized_height) * 255).clamp(0, 255).to(torch.uint8)
        a = torch.full((num_vertices,), 255, dtype=torch.uint8, device=device)

        colors = torch.stack([r, g, b, a], dim=1)

    elif color_mode == "density":
        # Sample voxel density values at vertex positions
        D, H, W = voxels.shape

        # Convert normalized vertices (-1 to 1) back to voxel coordinates
        v_max = max(D, H, W)
        voxel_coords = vertices * (v_max / 2) + (v_max / 2)
        voxel_coords = torch.fliplr(voxel_coords)  # Undo the flip from mesh generation

        # Clamp to valid voxel range
        voxel_coords[:, 0] = voxel_coords[:, 0].clamp(0, D - 1)
        voxel_coords[:, 1] = voxel_coords[:, 1].clamp(0, H - 1)
        voxel_coords[:, 2] = voxel_coords[:, 2].clamp(0, W - 1)

        # Sample voxel values (using nearest neighbor for simplicity)
        indices = voxel_coords.long()
        sampled_densities = voxels[indices[:, 0], indices[:, 1], indices[:, 2]]

        # Normalize densities to 0-1 range
        density_min = sampled_densities.min()
        density_max = sampled_densities.max()

        if density_max > density_min:
            normalized_density = (sampled_densities - density_min) / (density_max - density_min)
        else:
            normalized_density = torch.ones_like(sampled_densities) * 0.5

        # Color based on density (grayscale to colored gradient)
        r = (normalized_density * 255).clamp(0, 255).to(torch.uint8)
        g = (normalized_density * 200).clamp(0, 255).to(torch.uint8)
        b = (normalized_density * 150).clamp(0, 255).to(torch.uint8)
        a = torch.full((num_vertices,), 255, dtype=torch.uint8, device=device)

        colors = torch.stack([r, g, b, a], dim=1)

    elif color_mode == "normal":
        # Simple pseudo-normal based coloring
        # This is a simplified version - proper normals would require face information
        x_norm = (vertices[:, 0] + 1.0) * 0.5
        y_norm = (vertices[:, 1] + 1.0) * 0.5
        z_norm = (vertices[:, 2] + 1.0) * 0.5

        r = (x_norm * 255).clamp(0, 255).to(torch.uint8)
        g = (y_norm * 255).clamp(0, 255).to(torch.uint8)
        b = (z_norm * 255).clamp(0, 255).to(torch.uint8)
        a = torch.full((num_vertices,), 255, dtype=torch.uint8, device=device)

        colors = torch.stack([r, g, b, a], dim=1)

    else:
        # Default: white color
        colors = torch.full((num_vertices, 4), 255, dtype=torch.uint8, device=device)

    return colors


def save_glb(vertices, faces, filepath, metadata=None, colors=None):
    """
    Save PyTorch tensor vertices and faces as a GLB file without external dependencies.

    Parameters:
    vertices: torch.Tensor of shape (N, 3) - The vertex coordinates
    faces: torch.Tensor of shape (M, 3) - The face indices (triangle faces)
    filepath: str - Output filepath (should end with .glb)
    """

    # Convert tensors to numpy arrays
    vertices_np = vertices.cpu().numpy().astype(np.float32)
    faces_np = faces.cpu().numpy().astype(np.uint32)

    colors_np = None
    if colors is not None:
        c = colors.detach().cpu()
        if c.dtype != torch.uint8:
            # float -> uint8 (robust)
            c = c.to(torch.float32)
            if float(c.max()) <= 1.0:
                c = (c * 255.0).clamp(0, 255).to(torch.uint8)
            else:
                c = c.clamp(0, 255).to(torch.uint8)

        c_np = c.numpy()
        if c_np.ndim == 2 and c_np.shape[0] == vertices_np.shape[0] and c_np.shape[1] in (3, 4):
            if c_np.shape[1] == 3:
                alpha = np.full((c_np.shape[0], 1), 255, dtype=np.uint8)
                c_np = np.concatenate([c_np, alpha], axis=1)
            colors_np = c_np  # (N,4) uint8 RGBA

    vertices_buffer = vertices_np.tobytes()
    indices_buffer = faces_np.tobytes()
    colors_buffer   = colors_np.tobytes() if colors_np is not None else b""

    def pad_to_4_bytes(buffer):
        padding_length = (4 - (len(buffer) % 4)) % 4
        return buffer + b'\x00' * padding_length

    vertices_buffer_padded = pad_to_4_bytes(vertices_buffer)
    indices_buffer_padded = pad_to_4_bytes(indices_buffer)
    colors_buffer_padded = pad_to_4_bytes(colors_buffer) if colors_buffer else b""

    buffer_data = vertices_buffer_padded + indices_buffer_padded + colors_buffer_padded

    vertices_byte_length = len(vertices_buffer)
    vertices_byte_offset = 0
    indices_byte_length = len(indices_buffer)
    indices_byte_offset = len(vertices_buffer_padded)
    colors_byte_length = len(colors_buffer)
    colors_byte_offset = indices_byte_offset + len(indices_buffer_padded)

    # Build bufferViews
    buffer_views = [
        {
            "buffer": 0,
            "byteOffset": vertices_byte_offset,
            "byteLength": vertices_byte_length,
            "target": 34962  # ARRAY_BUFFER
        },
        {
            "buffer": 0,
            "byteOffset": indices_byte_offset,
            "byteLength": indices_byte_length,
            "target": 34963  # ELEMENT_ARRAY_BUFFER
        }
    ]

    # Build accessors
    accessors = [
        {
            "bufferView": 0,
            "byteOffset": 0,
            "componentType": 5126,  # FLOAT
            "count": len(vertices_np),
            "type": "VEC3",
            "max": vertices_np.max(axis=0).tolist(),
            "min": vertices_np.min(axis=0).tolist()
        },
        {
            "bufferView": 1,
            "byteOffset": 0,
            "componentType": 5125,  # UNSIGNED_INT
            "count": faces_np.size,
            "type": "SCALAR"
        }
    ]

    # Build primitive attributes
    primitive_attributes = {"POSITION": 0}

    # Add color buffer view and accessor if colors are provided
    if colors_np is not None:
        buffer_views.append({
            "buffer": 0,
            "byteOffset": colors_byte_offset,
            "byteLength": colors_byte_length,
            "target": 34962  # ARRAY_BUFFER
        })

        accessors.append({
            "bufferView": 2,
            "byteOffset": 0,
            "componentType": 5121,  # UNSIGNED_BYTE
            "count": len(colors_np),
            "type": "VEC4",
            "normalized": True  # Important: tells glTF to normalize [0,255] to [0,1]
        })

        primitive_attributes["COLOR_0"] = 2

    gltf = {
        "asset": {"version": "2.0", "generator": "ComfyUI"},
        "buffers": [
            {
                "byteLength": len(buffer_data)
            }
        ],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": primitive_attributes,
                        "indices": 1,
                        "mode": 4  # TRIANGLES
                    }
                ]
            }
        ],
        "nodes": [
            {
                "mesh": 0
            }
        ],
        "scenes": [
            {
                "nodes": [0]
            }
        ],
        "scene": 0
    }

    if metadata is not None:
        gltf["asset"]["extras"] = metadata

    # Convert the JSON to bytes
    gltf_json = json.dumps(gltf).encode('utf8')

    def pad_json_to_4_bytes(buffer):
        padding_length = (4 - (len(buffer) % 4)) % 4
        return buffer + b' ' * padding_length

    gltf_json_padded = pad_json_to_4_bytes(gltf_json)

    # Create the GLB header
    # Magic glTF
    glb_header = struct.pack('<4sII', b'glTF', 2, 12 + 8 + len(gltf_json_padded) + 8 + len(buffer_data))

    # Create JSON chunk header (chunk type 0)
    json_chunk_header = struct.pack('<II', len(gltf_json_padded), 0x4E4F534A)  # "JSON" in little endian

    # Create BIN chunk header (chunk type 1)
    bin_chunk_header = struct.pack('<II', len(buffer_data), 0x004E4942)  # "BIN\0" in little endian

    # Write the GLB file
    with open(filepath, 'wb') as f:
        f.write(glb_header)
        f.write(json_chunk_header)
        f.write(gltf_json_padded)
        f.write(bin_chunk_header)
        f.write(buffer_data)

    return filepath


class SaveGLB(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SaveGLB",
            category="3d",
            is_output_node=True,
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.String.Input("filename_prefix", default="mesh/ComfyUI"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo]
        )

    @classmethod
    def execute(cls, mesh, filename_prefix) -> IO.NodeOutput:
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        results = []

        metadata = {}
        if not args.disable_metadata:
            if cls.hidden.prompt is not None:
                metadata["prompt"] = json.dumps(cls.hidden.prompt)
            if cls.hidden.extra_pnginfo is not None:
                for x in cls.hidden.extra_pnginfo:
                    metadata[x] = json.dumps(cls.hidden.extra_pnginfo[x])

        for i in range(mesh.vertices.shape[0]):
            f = f"{filename}_{counter:05}_.glb"
            c = None
            if hasattr(mesh, "vertex_colors"):
                c = mesh.vertex_colors[i]  # (V,4) uint8 or float
            save_glb(mesh.vertices[i], mesh.faces[i], os.path.join(full_output_folder, f), metadata, c)
            results.append({
                "filename": f,
                "subfolder": subfolder,
                "type": "output"
            })
            counter += 1
        return IO.NodeOutput(ui={"3d": results})


class ProjectImageColorsToMesh(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ProjectImageColorsToMesh",
            category="3d",
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.Image.Input("image", tooltip="Reference image to project colors from"),
                IO.Int.Input("num_views", default=4, min=1, max=8, tooltip="Number of camera angles to project from (1=front, 4=front+sides, 8=all around)"),
                IO.Float.Input("camera_distance", default=2.0, min=0.1, max=10.0, step=0.1, tooltip="Distance of camera from object"),
                IO.Combo.Input("fallback_mode", options=["gray", "position", "height"], default="position", tooltip="Color mode for vertices not visible from any camera angle"),
            ],
            outputs=[
                IO.Mesh.Output(),
            ]
        )

    @classmethod
    def execute(cls, mesh, image, num_views, camera_distance, fallback_mode) -> IO.NodeOutput:
        # Process each mesh in the batch
        all_vertex_colors = []

        for batch_idx in range(mesh.vertices.shape[0]):
            vertices = mesh.vertices[batch_idx]  # (V, 3)

            # Use first image in batch if image is batched
            img = image[0] if len(image.shape) == 4 else image

            # Project colors with enhanced fallback
            colors = project_image_colors_to_vertices_enhanced(
                vertices,
                img,
                camera_distance=camera_distance,
                num_views=num_views,
                fallback_mode=fallback_mode
            )

            all_vertex_colors.append(colors)

        # Create output mesh with colors
        output_mesh = Types.MESH(mesh.vertices, mesh.faces)
        output_mesh.vertex_colors = torch.stack(all_vertex_colors)

        return IO.NodeOutput(output_mesh)


def project_image_colors_to_vertices_enhanced(vertices, image, camera_distance=2.0, num_views=4, fallback_mode="position"):
    """
    Enhanced version with better fallback handling for non-visible vertices.
    """
    device = vertices.device
    num_vertices = vertices.shape[0]

    if num_vertices == 0:
        return torch.zeros((0, 4), dtype=torch.uint8, device=device)

    # Convert image to device and ensure correct format
    if image.device != device:
        image = image.to(device)

    img_h, img_w = image.shape[0], image.shape[1]

    # Initialize color accumulator
    vertex_colors_sum = torch.zeros((num_vertices, 3), dtype=torch.float32, device=device)
    vertex_weights = torch.zeros((num_vertices, 1), dtype=torch.float32, device=device)

    # Define camera angles for multi-view projection
    if num_views == 1:
        camera_angles = [(0, 0)]
    elif num_views == 4:
        camera_angles = [(0, 0), (90, 0), (180, 0), (270, 0)]
    elif num_views == 6:
        camera_angles = [(0, 0), (90, 0), (180, 0), (270, 0), (0, 90), (0, -90)]
    else:
        # Evenly distributed around
        camera_angles = [(i * 360 / num_views, 0) for i in range(num_views)]

    for azimuth, elevation in camera_angles:
        # Rotate vertices based on camera angle
        azimuth_rad = torch.tensor(azimuth * np.pi / 180.0, device=device)
        elevation_rad = torch.tensor(elevation * np.pi / 180.0, device=device)

        # Rotation matrix (around Y axis for azimuth)
        cos_a = torch.cos(azimuth_rad)
        sin_a = torch.sin(azimuth_rad)

        # Apply rotation
        v_rotated = vertices.clone()
        x_rot = vertices[:, 0] * cos_a - vertices[:, 2] * sin_a
        z_rot = vertices[:, 0] * sin_a + vertices[:, 2] * cos_a
        v_rotated[:, 0] = x_rot
        v_rotated[:, 2] = z_rot

        # Orthographic projection
        x_proj = ((v_rotated[:, 0] + 1.0) * 0.5 * (img_w - 1)).clamp(0, img_w - 1)
        y_proj = ((1.0 - (v_rotated[:, 1] + 1.0) * 0.5) * (img_h - 1)).clamp(0, img_h - 1)

        # Check visibility
        visible = v_rotated[:, 2] > -0.5

        # Bilinear interpolation
        x_floor = x_proj.floor().long()
        y_floor = y_proj.floor().long()
        x_ceil = (x_floor + 1).clamp(max=img_w - 1)
        y_ceil = (y_floor + 1).clamp(max=img_h - 1)

        x_frac = (x_proj - x_floor.float()).unsqueeze(1)
        y_frac = (y_proj - y_floor.float()).unsqueeze(1)

        c00 = image[y_floor, x_floor]
        c01 = image[y_floor, x_ceil]
        c10 = image[y_ceil, x_floor]
        c11 = image[y_ceil, x_ceil]

        c0 = c00 * (1 - x_frac) + c01 * x_frac
        c1 = c10 * (1 - x_frac) + c11 * x_frac
        sampled_colors = c0 * (1 - y_frac) + c1 * y_frac

        # Accumulate colors for visible vertices
        visible_mask = visible.unsqueeze(1).float()
        vertex_colors_sum += sampled_colors * visible_mask
        vertex_weights += visible_mask

    # Average colors across views
    vertex_weights = vertex_weights.clamp(min=1e-6)
    final_colors = vertex_colors_sum / vertex_weights

    # Handle vertices not visible from any view
    no_color_mask = (vertex_weights.squeeze() < 0.1)
    if no_color_mask.any():
        if fallback_mode == "gray":
            final_colors[no_color_mask] = 0.5
        elif fallback_mode == "height":
            z_coords = vertices[no_color_mask, 2]
            z_min, z_max = z_coords.min(), z_coords.max()
            if z_max > z_min:
                normalized_height = (z_coords - z_min) / (z_max - z_min)
            else:
                normalized_height = torch.ones_like(z_coords) * 0.5
            final_colors[no_color_mask, 0] = normalized_height
            final_colors[no_color_mask, 1] = 1.0 - torch.abs(normalized_height - 0.5) * 2.0
            final_colors[no_color_mask, 2] = 1.0 - normalized_height
        else:  # position
            final_colors[no_color_mask] = ((vertices[no_color_mask] + 1.0) * 0.5).clamp(0, 1)

    # Convert to uint8 RGBA
    r = (final_colors[:, 0] * 255).clamp(0, 255).to(torch.uint8)
    g = (final_colors[:, 1] * 255).clamp(0, 255).to(torch.uint8)
    b = (final_colors[:, 2] * 255).clamp(0, 255).to(torch.uint8)
    a = torch.full((num_vertices,), 255, dtype=torch.uint8, device=device)

    colors = torch.stack([r, g, b, a], dim=1)
    return colors


class Hunyuan3dExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            EmptyLatentHunyuan3Dv2,
            Hunyuan3Dv2Conditioning,
            Hunyuan3Dv2ConditioningMultiView,
            VAEDecodeHunyuan3D,
            VoxelToMeshBasic,
            VoxelToMesh,
            SaveGLB,
            ProjectImageColorsToMesh,
        ]


async def comfy_entrypoint() -> Hunyuan3dExtension:
    return Hunyuan3dExtension()
