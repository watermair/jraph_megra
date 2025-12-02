import functools
from typing import Tuple, Callable

from absl import app
from frozendict import frozendict

import jax
import pickle
jax.config.update("jax_enable_x64",True)
import jax.numpy as jnp
import jraph
import matplotlib.pyplot as plt
import numpy as np
from jraph._src import graph as gn_graph
from jraph._src import models as gn_models


import trimesh
import jax.numpy as jnp
from frozendict import frozendict



import time


import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import numpy as np


def plot_meshgraph_3d(mesh):
    """
    Plot a MeshGraph in 3D:
    - primal vertices (colored by flip_id)
    - primal edges (colored by flip_site)
    - primal faces as semi-transparent triangles
    - dual mesh (face centers + dual edges)
    """

    # Convert JAX → numpy
    pos = np.asarray(mesh.nodes["position"])          # [n_nodes, 3]
    flip_id = np.asarray(mesh.nodes["flip_id"])       # [n_nodes]
    senders = np.asarray(mesh.senders)
    receivers = np.asarray(mesh.receivers)

    edge_flip = np.asarray(mesh.edges["lij"])

    face_senders = np.asarray(mesh.face_senders)       # [n_edges] face index
    face_receivers = np.asarray(mesh.face_receivers)   # [n_edges] face index
    face_vertices = np.asarray(mesh.faces["vertices"])     # [n_faces, 3]

    # ---------------------------------------------------------
    # 1) FACE CENTERS (dual vertices)
    # ---------------------------------------------------------
    face_centers = pos[face_vertices].mean(axis=1)  # [n_faces, 3]

    dual_s = face_senders
    dual_r = face_receivers

    dual_mask = (dual_s >= 0) & (dual_r >= 0)
    dual_s = dual_s[dual_mask]
    dual_r = dual_r[dual_mask]

    # ---------------------------------------------------------
    # 2) Plot setup
    # ---------------------------------------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Primal vertices
    ax.scatter(
        pos[:, 0], pos[:, 1], pos[:, 2],
        c=flip_id,
        s=40,
        cmap="viridis",
        alpha=1.0
    )



    # ---------------------------------------------------------
    # 3) Primal semi-transparent faces
    # ---------------------------------------------------------
    face_tris = pos[face_vertices]     # shape [n_faces, 3, 3]

    mesh_color = (0.4, 0.6, 1.0, 0.25)  # RGBA with alpha=0.25

    tri_collection = Poly3DCollection(
        face_tris,
        facecolors=mesh_color,
        edgecolors=(0.2, 0.3, 0.5, 0.5),
        linewidths=0.5
    )
    ax.add_collection3d(tri_collection)

    # ---------------------------------------------------------
    # 4) Primal edges (colored by flip_site)
    # ---------------------------------------------------------
    norm = plt.Normalize(vmin=edge_flip.min(), vmax=edge_flip.max())
    cmap = plt.cm.viridis

    for s, r, val in zip(senders, receivers, edge_flip):
        p1 = pos[s]
        p2 = pos[r]
        c = cmap(norm(val))
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color=c,
            linewidth=2,
            alpha=0.9
        )

    # ---------------------------------------------------------
    # 5) Dual mesh
    # ---------------------------------------------------------
    # Dual vertices (face centers)
    ax.scatter(
        face_centers[:, 0], face_centers[:, 1], face_centers[:, 2],
        c="red",
        s=20,
        alpha=0.8
    )
    fail= [32,35,37,39,42]
    ax.scatter(
        face_centers[fail, 0], face_centers[fail, 1], face_centers[fail, 2],
        c="yellow",
        s=60,
        alpha=0.8
    )

    # Dual edges
    for fs, fr in zip(dual_s, dual_r):
        c1 = face_centers[fs]
        c2 = face_centers[fr]
        ax.plot(
            [c1[0], c2[0]],
            [c1[1], c2[1]],
            [c1[2], c2[2]],
            color="red",
            linewidth=1,
            alpha=0.6
        )

    # ---------------------------------------------------------
    # Final layout
    # ---------------------------------------------------------
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    plt.show()

def plot_mesh_fast(mesh, face_color="lightblue", edge_color='r', cmap=None):
    """
    Ultra-fast 3D mesh plotter for large triangle meshes.
    - No per-vertex scatter
    - No loops over faces or edges
    - Single Poly3DCollection for maximum speed

    Params:
        mesh: MeshGraphsTuple
        face_color: constant color OR None
        edge_color: None or color string
        cmap: optional colormap applied to a scalar per-face field (e.g. curvature)
    """

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np

    def set_axes_equal(ax):
        """Force 3D axes to have equal aspect ratio."""
        import numpy as np

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])

        plot_radius = 0.5 * max([x_range, y_range, z_range])

        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    pos = np.asarray(mesh.nodes["position"])  # (N, 3)
    face_vertices = np.asarray(mesh.faces["vertices"])  # (F, 3)

    # Extract triangles (F, 3, 3)
    tris = pos[face_vertices]

    # If cmap is specified, use scalar field e.g. mean curvature or area or whatever you want
    if cmap is not None:
        # Example: mean curvature per face (F,)
        # You can pass your own scalar field instead
        scalar = np.linalg.norm(tris.mean(axis=1), axis=1)  # dummy
        face_colors = plt.cm.get_cmap(cmap)((scalar - scalar.min()) / (scalar.max() - scalar.min() + 1e-12))
    else:
        # constant color, transparent backside
        face_colors = face_color

    # Build single collection
    coll = Poly3DCollection(
        tris,
        facecolors=face_colors,
        edgecolors=edge_color,
        linewidths=0.1 if edge_color else 0.0
    )
    coll.set_alpha(0.6)

    # Plot
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(coll)

    # Autoscale: VERY IMPORTANT for fast rendering
    xyz = pos
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2
    span = (maxs - mins).max() / 2
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    set_axes_equal(ax)
    ax.set_axis_off()

    plt.tight_layout()
    plt.show()


def check_mesh_consistency(mesh, eps=1e-12, verbose=True):
    """
    Robust consistency checker for the MeshGraph layout you're using.

    Assumes the MeshGraph provides:
      - mesh.nodes["position"]        (nV,3)
      - mesh.senders, mesh.receivers  (nE,)
      - mesh.faces["vertices"]        (nF,3)
      - mesh.face_senders, face_receivers  (nE,)   (dual adjacency)
      - mesh.edges_topo[...]             (prev/next/orient/compute flags)

    This function is resilient against:
      - faces that are duplicates (same vertex set, different order)
      - faces whose inferred `face_edges` from dual adjacency are incomplete:
          in that case it falls back to scanning all edges to find those
          that connect two vertices of the face.
    """
    import numpy as np

    # --- Read arrays (your usage is already correct) ---
    pos = np.asarray(mesh.nodes["position"])   # (nV,3)
    send = np.asarray(mesh.senders)            # (nE,)
    recv = np.asarray(mesh.receivers)          # (nE,)
    FV = np.asarray(mesh.faces["vertices"])    # (nF,3)

    face_senders = np.asarray(mesh.face_senders)    # (nE,)
    face_receivers = np.asarray(mesh.face_receivers)  # (nE,)

    topo = mesh.edges_topo
    fs_prev = np.asarray(topo["face_senders_prev"])
    fs_next = np.asarray(topo["face_senders_next"])
    fr_prev = np.asarray(topo["face_receivers_prev"])
    fr_next = np.asarray(topo["face_receivers_next"])

    fs_prev_or = np.asarray(topo["face_senders_prev_orient"])
    fs_next_or = np.asarray(topo["face_senders_next_orient"])
    fr_prev_or = np.asarray(topo["face_receivers_prev_orient"])
    fr_next_or = np.asarray(topo["face_receivers_next_orient"])

    fs_or = np.asarray(topo["face_senders_orient"])
    fr_or = np.asarray(topo["face_receivers_orient"])

    compute_fs = np.asarray(topo["compute_face_senders"])
    compute_fr = np.asarray(topo["compute_face_receivers"])

    nV = pos.shape[0]
    nE = send.shape[0]
    nF = FV.shape[0]

    report = {}

    # -------------------------
    # 0) Basic face checks
    # -------------------------
    bad_faces = []
    areas = np.zeros(nF, dtype=float)
    for f in range(nF):
        a, b, c = FV[f]
        if a < 0 or b < 0 or c < 0:
            bad_faces.append((f, "negative index"))
        if len({a, b, c}) < 3:
            bad_faces.append((f, "duplicate vertices"))

        # compute face area (safe)
        v0 = pos[a]; v1 = pos[b]; v2 = pos[c]
        cross = np.cross(v1 - v0, v2 - v0)
        areas[f] = 0.5 * np.linalg.norm(cross)

    report["bad_faces"] = bad_faces
    report["zero_area_faces"] = np.where(areas <= eps)[0].tolist()
    report["tiny_area_faces"] = np.where((areas > eps) & (areas < 1e-8))[0].tolist()

    # -------------------------
    # 1) Duplicate / non-unique faces
    #    (faces that refer to the same triple of vertices in some order)
    # -------------------------
    sorted_face_keys = np.array([tuple(sorted(FV[f])) for f in range(nF)])
    # Map sorted key to list of face indices
    from collections import defaultdict
    key_to_faces = defaultdict(list)
    for f, k in enumerate(sorted_face_keys):
        key_to_faces[k].append(f)
    duplicated_face_groups = [v for v in key_to_faces.values() if len(v) > 1]
    report["duplicate_face_groups"] = duplicated_face_groups

    # -------------------------
    # 2) Build face_edges (robust)
    #    Primary method: collect edges from face_senders/face_receivers adjacency.
    #    Fallback: search edges whose endpoints are subset of the face vertices.
    # -------------------------
    face_edges = [[] for _ in range(nF)]
    # Primary pass: use per-edge dual mapping (face_senders, face_receivers)
    for e in range(nE):
        fs = int(face_senders[e])
        fr = int(face_receivers[e])
        if fs >= 0:
            face_edges[fs].append(e)
        if fr >= 0 and fr != fs:
            face_edges[fr].append(e)

    # Fallback search for any face that doesn't have exactly 3 edges yet:
    # find edges whose endpoint set is contained in the face vertex set.
    for f in range(nF):
        if len(face_edges[f]) == 3:
            continue
        fv_set = set(FV[f])
        # find candidate edges where both endpoints appear in face vertices
        candidates = [e for e in range(nE) if (send[e] in fv_set and recv[e] in fv_set)]
        # remove duplicates and keep order deterministic (sort)
        candidates = sorted(set(candidates))
        # prefer existing mapped edges first (if any)
        for e in candidates:
            if e not in face_edges[f]:
                face_edges[f].append(e)
            if len(face_edges[f]) == 3:
                break
        # if still not 3, report it later
    report["faces_with_incomplete_edge_list"] = [ (f, len(face_edges[f])) for f in range(nF) if len(face_edges[f]) != 3 ]

    # -------------------------
    # 3) Basic edge->face checks (edge endpoints must belong to the face's vertices)
    # -------------------------
    edge_face_mismatch = []
    for e in range(nE):
        u, v = int(send[e]), int(recv[e])
        for f in (int(face_senders[e]), int(face_receivers[e])):
            if f < 0:
                continue
            a, b, c = FV[f]
            if not ((u in (a,b,c)) and (v in (a,b,c))):
                edge_face_mismatch.append((e, f))
    report["edge_face_mismatch"] = edge_face_mismatch

    # -------------------------
    # 4) Check edge counts per face
    # -------------------------
    faces_wrong_edge_count = [(f, len(face_edges[f])) for f in range(nF) if len(face_edges[f]) != 3]
    report["faces_wrong_number_of_edges"] = faces_wrong_edge_count

    # -------------------------
    # 5) Check cycle membership / edge sets modulo rotation
    #    Accept any cyclic rotation of the edge ordering.
    # -------------------------
    incorrect_face_cycles = []
    for f in range(nF):
        fe = list(face_edges[f])
        if len(fe) != 3:
            incorrect_face_cycles.append((f, "bad_edge_count", fe))
            continue

        a, b, c = FV[f]
        required = [
            {a,b},
            {b,c},
            {c,a}
        ]
        given_sets = [{int(send[e]), int(recv[e])} for e in fe]

        ok = False
        for shift in range(3):
            rotated = given_sets[shift:] + given_sets[:shift]
            if all(rotated[i] == required[i] for i in range(3)):
                ok = True
                break
        if not ok:
            incorrect_face_cycles.append((f, fe))
    report["incorrect_face_cycles"] = incorrect_face_cycles

    # -------------------------
    # 6) Orientation check (directed) modulo rotation
    #    We verify that the directed edges of the face align CCW or CW in some rotation.
    # -------------------------
    incorrect_face_orientations = []
    for f in range(nF):
        fe = list(face_edges[f])
        if len(fe) != 3:
            incorrect_face_orientations.append((f, "bad_edge_count"))
            continue

        a, b, c = FV[f]
        expected_cw  = [(a,b), (b,c), (c,a)]
        expected_ccw = [(a,c), (c,b), (b,a)]

        dir_pairs = [(int(send[e]), int(recv[e])) for e in fe]

        # try rotations
        ok = False
        for shift in range(3):
            rotated = [dir_pairs[(i + shift) % 3] for i in range(3)]
            # compare directed endpoints to expected cw or ccw disregarding vertex ordering
            if all({rotated[i][0], rotated[i][1]} == {expected_cw[i][0], expected_cw[i][1]} for i in range(3)):
                ok = True
                break
            if all({rotated[i][0], rotated[i][1]} == {expected_ccw[i][0], expected_ccw[i][1]} for i in range(3)):
                ok = True
                break
        if not ok:
            incorrect_face_orientations.append((f, fe))
    report["incorrect_face_orientations"] = incorrect_face_orientations

    # -------------------------
    # 7) Prev/Next checks using robust local slot lookup
    #    Ensure edge's prev/next for the side (sender or receiver) match the face cycle.
    # -------------------------
    bad_prev_next = []

    # define helper to get prev/next arrays from topo
    fs_prev_arr = fs_prev
    fs_next_arr = fs_next
    fr_prev_arr = fr_prev
    fr_next_arr = fr_next

    for e in range(nE):
        for side in ("sender", "receiver"):
            fidx = int(face_senders[e]) if side == "sender" else int(face_receivers[e])
            if fidx < 0:
                continue
            fe = list(face_edges[fidx])
            if e not in fe:
                bad_prev_next.append((e, side, "edge not in face_edges"))
                continue
            k = int(np.where(np.array(fe) == e)[0][0])
            expected_prev = fe[(k - 1) % 3]
            expected_next = fe[(k + 1) % 3]

            got_prev = int(fs_prev_arr[e]) if side == "sender" else int(fr_prev_arr[e])
            got_next = int(fs_next_arr[e]) if side == "sender" else int(fr_next_arr[e])

            if not (got_prev == expected_prev and got_next == expected_next):
                bad_prev_next.append((e, side, f"expected prev={expected_prev}, next={expected_next}, got prev={got_prev}, next={got_next}"))

    report["bad_prev_next"] = bad_prev_next

    # -------------------------
    # 8) Self orientation correctness (face_senders_orient / face_receivers_orient)
    #    Verify the stored orientation +1/-1 matches the directed primal edge wrt the face's vertex cycle.
    # -------------------------
    orient_errors = []
    for e in range(nE):
        u, v = int(send[e]), int(recv[e])
        for side in ("sender", "receiver"):
            fidx = int(face_senders[e]) if side == "sender" else int(face_receivers[e])
            if fidx < 0:
                continue
            a, b, c = FV[fidx]
            if side == "sender":
                stored_or = int(fs_or[e])
            else:
                stored_or = int(fr_or[e])

            # if edge direction matches a->b or b->c or c->a then expected +1; if reversed then -1; else 0
            if (u == a and v == b) or (u == b and v == c) or (u == c and v == a):
                expected = +1
            elif (v == a and u == b) or (v == b and u == c) or (v == c and u == a):
                expected = -1
            else:
                expected = 0

            if stored_or != expected:
                orient_errors.append((e, side, fidx, stored_or, expected))
    report["orientation_errors"] = orient_errors

    # -------------------------
    # 9) Leader-selection: exactly one per face
    # -------------------------
    leaders_per_face = [[] for _ in range(nF)]
    for e in range(nE):
        if bool(compute_fs[e]):
            fs = int(face_senders[e])
            if fs >= 0:
                leaders_per_face[fs].append(("sender", e))
        if bool(compute_fr[e]):
            fr = int(face_receivers[e])
            if fr >= 0:
                leaders_per_face[fr].append(("receiver", e))

    faces_with_wrong_leader_count = []
    for f in range(nF):
        if len(face_edges[f]) != 3:
            continue
        if len(leaders_per_face[f]) != 1:
            faces_with_wrong_leader_count.append((f, leaders_per_face[f]))
    report["faces_with_wrong_leader_count"] = faces_with_wrong_leader_count

    # -------------------------
    # 10) Vertex degree (simple diagnostic)
    # -------------------------
    vertex_degree = np.zeros(nV, dtype=int)
    for (a,b,c) in FV:
        vertex_degree[int(a)] += 1
        vertex_degree[int(b)] += 1
        vertex_degree[int(c)] += 1
    report["vertex_degree"] = vertex_degree.tolist()
    report["vertices_degree_gt_6"] = np.where(vertex_degree > 6)[0].tolist()

    # -------------------------
    # 11) Summary / printing
    # -------------------------
    if verbose:
        print("==== Mesh Consistency Report ====")
        for key, val in report.items():
            if isinstance(val, list) and len(val) == 0:
                continue
            # keep concise single-line prints for large arrays
            if key == "vertex_degree":
                print(f"{key}: mean={np.mean(val):.3f}, max={np.max(val)}, n={len(val)}")
            else:
                print(f"{key}: {val}")

    return report




# Tell tree_util how to navigate frozendicts.
jax.tree_util.register_pytree_node(
    frozendict,
    flatten_func=lambda s: (tuple(s.values()), tuple(s.keys())),
    unflatten_func=lambda k, xs: frozendict(zip(k, xs)))


def build_meshgraph_from_trimesh_jax(mesh):
    """
    Fully JAX/JIT-compatible version of the mesh graph builder.
    Computes:
      - face_senders / face_receivers
      - prev/next edges and orientations
      - compute_face_senders / compute_face_receivers
    using only JAX ops (no python loops).

    Input:
      mesh.faces, mesh.vertices, mesh.edges_unique, mesh.edges_unique_inverse

    Returns:
      graph dict compatible with your GN model.
    """

    # -------------------------------------------
    # Convert mesh arrays (JAX-friendly)
    # -------------------------------------------
    faces = jnp.asarray(mesh.faces, dtype=jnp.int32)  # (F,3)
    vertices = jnp.asarray(mesh.vertices, dtype=jnp.float32)  # (V,3)
    edges_unique = jnp.asarray(mesh.edges_unique, dtype=jnp.int32)  # (E,2)
    edges_unique_inverse = jnp.asarray(mesh.edges_unique_inverse, dtype=jnp.int32)

    n_faces = faces.shape[0]
    n_edges = edges_unique.shape[0]
    n_nodes = vertices.shape[0]

    # -------------------------------------------
    # 1. Directed face edges
    # -------------------------------------------
    face_vertices = faces  # (F,3)

    # If faces = [i,j,k], we define CCW edges (i->j, j->k, k->i)
    face_edges_directed = jnp.stack([
        face_vertices[:, [0, 1]],
        face_vertices[:, [1, 2]],
        face_vertices[:, [2, 0]],
    ], axis=1)  # (F,3,2)

    # Face edges → unique-edge index
    face_edges = edges_unique_inverse.reshape(n_faces, 3)  # (F,3)

    # -------------------------------------------
    # 2. Orientation of face edges inside faces
    # -------------------------------------------
    ue0 = edges_unique[face_edges, 0]
    ue1 = edges_unique[face_edges, 1]

    fe0 = face_edges_directed[:, :, 0]
    fe1 = face_edges_directed[:, :, 1]

    face_edge_orientation = jnp.where((fe0 == ue0) & (fe1 == ue1), 1, -1).astype(jnp.int8)

    # -------------------------------------------
    # 3. edge → face adjacency (vectorized)
    # -------------------------------------------
    # We need, for each edge, the two faces referencing it.
    # Build flattened indices:
    flat_edges = face_edges.reshape(-1)  # (3F,)
    flat_faces = jnp.repeat(jnp.arange(n_faces), 3)  # (F each repeated 3)
    flat_pos = jnp.tile(jnp.arange(3), n_faces)  # (0,1,2 repeating)

    # For edges, gather all faces that include them
    # First: group by edge using segment_min/max on a mask trick

    # For each edge, we want first and second idx in flat list
    # Create an index array [0..3F-1]
    flat_idx = jnp.arange(flat_edges.shape[0])

    # Mask where edge idx matches
    # But segment operations require numeric values, so we'll map missing to large values.
    max_int = jnp.int32(2 ** 31 - 1)

    # Build segment-min over flat_idx where flat_edges == e, else large
    seg_min_idx = jax.ops.segment_min(
        jnp.where(flat_edges >= 0, flat_idx, max_int),
        flat_edges,
        n_edges
    )

    # For the second face, mask out the first occurrence
    # We create a masked version where the first index gets max_int
    masked_flat_idx = jnp.where(flat_idx == seg_min_idx[flat_edges], max_int, flat_idx)

    seg_second_idx = jax.ops.segment_min(masked_flat_idx, flat_edges, n_edges)

    # If no second, seg_second_idx[e] = max_int → treat as -1
    face0_idx = seg_min_idx
    face1_idx = jnp.where(seg_second_idx == max_int, -1, seg_second_idx)

    # Convert to face numbers and face positions
    face_senders = flat_faces[face0_idx]  # first face is sender by default
    pos_s = flat_pos[face0_idx]

    face_receivers = jnp.where(
        face1_idx >= 0,
        flat_faces[face1_idx],
        -1
    )
    pos_r = jnp.where(
        face1_idx >= 0,
        flat_pos[face1_idx],
        -1
    )

    # -------------------------------------------
    # 4. Correct sender/receiver assignment
    # Determine which adjacent face aligns with primal direction
    # -------------------------------------------
    senders_p = edges_unique[:, 0]
    receivers_p = edges_unique[:, 1]

    face0_edge = face_edges_directed[face_senders, pos_s]  # (E,2)
    face1_edge = jnp.where(
        face_receivers[:, None] >= 0,
        face_edges_directed[face_receivers, pos_r],
        jnp.zeros((n_edges, 2), dtype=jnp.int32)
    )

    match0 = (face0_edge[:, 0] == senders_p) & (face0_edge[:, 1] == receivers_p)
    match1 = (face1_edge[:, 0] == senders_p) & (face1_edge[:, 1] == receivers_p)

    # Fix sender / receiver faces using matches
    face_receivers_corrected = jnp.where(
        match0 & ~match1, face_senders,
        jnp.where(
            match1 & ~match0, face_receivers,
            face_receivers  # fallback deterministic
        ))

    face_senders_corrected = jnp.where(
        match0 & ~match1, face_receivers,
        jnp.where(
            match1 & ~match0, face_senders,
            face_senders
        ))

    face_senders = face_senders_corrected
    face_receivers = face_receivers_corrected

    # -------------------------------------------
    # 5. prev/next edges in each face (vectorized)
    # -------------------------------------------
    face_prev = jnp.roll(face_edges, 2, axis=1)
    face_next = jnp.roll(face_edges, -1, axis=1)

    face_prev_orient = jnp.roll(face_edge_orientation, 2, axis=1)
    face_next_orient = jnp.roll(face_edge_orientation, -1, axis=1)

    # Build dense (F,3) arrays mapping face->edge position for sender and receiver
    # Example: for edge e with sender face f_s and pos_s,
    # face_senders_prev[e] = face_prev[f_s,pos_s]

    # sender-side neighbors
    f_s = face_senders
    p_s = pos_s

    valid_s = f_s >= 0
    fs_prev = jnp.where(
        valid_s, face_prev[f_s, p_s], -1
    )
    fs_next = jnp.where(
        valid_s, face_next[f_s, p_s], -1
    )

    fs_prev_or = jnp.where(
        valid_s, face_prev_orient[f_s, p_s], 0
    )
    fs_next_or = jnp.where(
        valid_s, face_next_orient[f_s, p_s], 0
    )

    fs_orient = jnp.where(
        valid_s, face_edge_orientation[f_s, p_s], 0
    )

    # receiver-side neighbors
    f_r = face_receivers
    p_r = pos_r

    valid_r = f_r >= 0
    fr_prev = jnp.where(
        valid_r, face_prev[f_r, p_r], -1
    )
    fr_next = jnp.where(
        valid_r, face_next[f_r, p_r], -1
    )

    fr_prev_or = jnp.where(
        valid_r, face_prev_orient[f_r, p_r], 0
    )
    fr_next_or = jnp.where(
        valid_r, face_next_orient[f_r, p_r], 0
    )

    fr_orient = jnp.where(
        valid_r, face_edge_orientation[f_r, p_r], 0
    )

    # -------------------------------------------
    # 6. Vectorized leader selection (per face)
    # -------------------------------------------
    # one random per edge
    key = jax.random.PRNGKey(42)
    rand = jax.random.uniform(key, shape=(n_edges,), dtype=jnp.float64)

    # Pick minimal-rand edge per face
    face_rand_triplet = rand[face_edges]  # (F,3)
    leader_pos = jnp.argmin(face_rand_triplet, axis=1)  # (F,)

    # leader edge per face
    leader_edge_per_face = face_edges[jnp.arange(n_faces), leader_pos]

    # Flags per edge
    edge_idx = jnp.arange(n_edges, dtype=jnp.int32)

    compute_face_senders = (f_s >= 0) & (leader_edge_per_face[f_s] == edge_idx)
    compute_face_receivers = (f_r >= 0) & (leader_edge_per_face[f_r] == edge_idx)

    # -------------------------------------------
    # 7. Assemble final graph
    # -------------------------------------------
    nodes = frozendict({"position": vertices,"flip_id":jnp.zeros((vetices.shape[0],))})

    faces_dict = frozendict({
        "vertices": face_vertices,
        "edge_orientation": face_edge_orientation,
    })

    edges_topo = frozendict({
        "compute_face_senders": compute_face_senders,
        "compute_face_receivers": compute_face_receivers,

        "face_senders_prev": fs_prev,
        "face_senders_next": fs_next,
        "face_receivers_prev": fr_prev,
        "face_receivers_next": fr_next,

        "face_senders_prev_orient": fs_prev_or,
        "face_senders_next_orient": fs_next_or,
        "face_receivers_prev_orient": fr_prev_or,
        "face_receivers_next_orient": fr_next_or,

        "face_senders_orient": fs_orient,
        "face_receivers_orient": fr_orient,
    })

    edges_dict = frozendict({})

    globals_dict = frozendict({})

    return {
        "nodes": nodes,
        "edges": edges_dict,
        "edges_topo": edges_topo,
        "faces": faces_dict,

        "senders": edges_unique[:, 0],
        "receivers": edges_unique[:, 1],

        "face_senders": face_senders,
        "face_receivers": face_receivers,
        "face_vertices": face_vertices,

        "globals": globals_dict,

        "n_node": jnp.asarray([n_nodes]),
        "n_edge": jnp.asarray([n_edges]),
        "n_face": jnp.asarray([n_faces]),
        "n_node_max": n_nodes,
        "n_edge_max": n_edges,
        "rng_key": key,
    }



def build_meshgraph_from_trimesh(mesh):
    """
    Correct and robust mesh builder for MeshGraphNet-style processing.
    Ensures:
      - correct face → edge indexing
      - correct per-edge sender/receiver faces
      - correct prev/next edges inside faces
      - correct edge orientation inside each face
      - correct prev/next orientations
      - consistent JAX-ready output
    """

    import numpy as np
    import jax.numpy as jnp
    from frozendict import frozendict

    # ------------------------------------------------------------
    # 0. Load mesh data
    # ------------------------------------------------------------
    faces = np.asarray(mesh.faces, dtype=np.int32)        # (F,3)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)  # (V,3)

    edges_unique = np.asarray(mesh.edges_unique, dtype=np.int32)  # (E,2)
    edges_unique_inverse = np.asarray(mesh.edges_unique_inverse,
                                      dtype=np.int32)  # (F*3,)

    n_faces = faces.shape[0]
    n_nodes = vertices.shape[0]
    n_edges = edges_unique.shape[0]

    senders = edges_unique[:, 0].copy()
    receivers = edges_unique[:, 1].copy()

    # ------------------------------------------------------------
    # 1. Build face-local edges (CCW) and map to unique edges
    # ------------------------------------------------------------
    face_vertices = faces.copy()

    # Directed edges (v0→v1, v1→v2, v2→v0)
    face_edges_directed = np.stack([
        face_vertices[:, [0, 1]],
        face_vertices[:, [1, 2]],
        face_vertices[:, [2, 0]],
    ], axis=1)  # (F,3,2)

    # Unique edge indices per face slot
    face_edges = edges_unique_inverse.reshape(n_faces, 3)  # (F,3)

    # ------------------------------------------------------------
    # 2. Orientation of each face-edge in global unique edge direction
    # ------------------------------------------------------------
    ue0 = edges_unique[face_edges, 0]
    ue1 = edges_unique[face_edges, 1]
    fe0 = face_edges_directed[:, :, 0]
    fe1 = face_edges_directed[:, :, 1]

    face_edge_orientation = np.where(
        (fe0 == ue0) & (fe1 == ue1), 1, -1
    ).astype(np.int8)  # (F,3)

    # ------------------------------------------------------------
    # 3. Edge → faces adjacency
    # ------------------------------------------------------------
    edge_faces = [[] for _ in range(n_edges)]
    for f in range(n_faces):
        for slot in range(3):
            e = face_edges[f, slot]
            edge_faces[e].append((f, slot))

    # ------------------------------------------------------------
    # 4. Dual graph assignment: sender-face / receiver-face
    # ------------------------------------------------------------
    face_senders = np.full(n_edges, -1, dtype=np.int32)
    face_receivers = np.full(n_edges, -1, dtype=np.int32)

    for e in range(n_edges):
        adj = edge_faces[e]
        if len(adj) == 0:
            continue

        if len(adj) == 1:
            # boundary: only sender
            f0, _ = adj[0]
            face_senders[e] = f0
            continue

        # interior: 2 faces
        (f0, s0), (f1, s1) = adj
        u, v = senders[e], receivers[e]

        # check if face-edge matches primal direction
        match0 = (face_edges_directed[f0, s0, 0] == u) and \
                 (face_edges_directed[f0, s0, 1] == v)
        match1 = (face_edges_directed[f1, s1, 0] == u) and \
                 (face_edges_directed[f1, s1, 1] == v)

        if match0 and not match1:
            face_receivers[e] = f0
            face_senders[e] = f1
        elif match1 and not match0:
            face_receivers[e] = f1
            face_senders[e] = f0
        else:
            # fallback deterministic
            face_senders[e] = f0
            face_receivers[e] = f1

    # ------------------------------------------------------------
    # 5. Correct prev/next + orientations
    # ------------------------------------------------------------
    face_senders_prev = np.full(n_edges, -1, np.int32)
    face_senders_next = np.full(n_edges, -1, np.int32)
    face_receivers_prev = np.full(n_edges, -1, np.int32)
    face_receivers_next = np.full(n_edges, -1, np.int32)

    face_senders_prev_orient = np.zeros(n_edges, np.int8)
    face_senders_next_orient = np.zeros(n_edges, np.int8)
    face_receivers_prev_orient = np.zeros(n_edges, np.int8)
    face_receivers_next_orient = np.zeros(n_edges, np.int8)

    face_senders_orient = np.zeros(n_edges, np.int8)
    face_receivers_orient = np.zeros(n_edges, np.int8)

    for e in range(n_edges):

        # ----- SENDER SIDE -----
        fs = face_senders[e]
        if fs >= 0:
            # TRUE, ROBUST local slot location
            slot = np.where(face_edges[fs] == e)[0][0]

            prev_slot = (slot - 1) % 3
            next_slot = (slot + 1) % 3

            face_senders_prev[e] = face_edges[fs, prev_slot]
            face_senders_next[e] = face_edges[fs, next_slot]

            face_senders_prev_orient[e] = face_edge_orientation[fs, prev_slot]
            face_senders_next_orient[e] = face_edge_orientation[fs, next_slot]

            face_senders_orient[e] = face_edge_orientation[fs, slot]

        # ----- RECEIVER SIDE -----
        fr = face_receivers[e]
        if fr >= 0:
            slot = np.where(face_edges[fr] == e)[0][0]

            prev_slot = (slot - 1) % 3
            next_slot = (slot + 1) % 3

            face_receivers_prev[e] = face_edges[fr, prev_slot]
            face_receivers_next[e] = face_edges[fr, next_slot]

            face_receivers_prev_orient[e] = face_edge_orientation[fr, prev_slot]
            face_receivers_next_orient[e] = face_edge_orientation[fr, next_slot]

            face_receivers_orient[e] = face_edge_orientation[fr, slot]

    # ------------------------------------------------------------
    # 6. Per-face leader election — correct and independent
    # ------------------------------------------------------------
    rng = np.random.default_rng()
    rand_edge = rng.random(n_edges)

    face_rand = rand_edge[face_edges]                 # (F,3)
    leader_slot = np.argmin(face_rand, axis=1)        # (F,)
    leader_edge = face_edges[np.arange(n_faces), leader_slot]

    compute_face_senders = np.zeros(n_edges, bool)
    compute_face_receivers = np.zeros(n_edges, bool)

    for e in range(n_edges):
        fs = face_senders[e]
        fr = face_receivers[e]
        if fs >= 0 and leader_edge[fs] == e:
            compute_face_senders[e] = True
        if fr >= 0 and leader_edge[fr] == e:
            compute_face_receivers[e] = True

    # ------------------------------------------------------------
    # 7. Output in JAX-friendly form
    # ------------------------------------------------------------



    edges_topo = frozendict({

        "compute_face_senders": jnp.asarray(compute_face_senders),
        "compute_face_receivers": jnp.asarray(compute_face_receivers),

        "face_senders_prev": jnp.asarray(face_senders_prev),
        "face_senders_next": jnp.asarray(face_senders_next),
        "face_receivers_prev": jnp.asarray(face_receivers_prev),
        "face_receivers_next": jnp.asarray(face_receivers_next),

        "face_senders_prev_orient": jnp.asarray(face_senders_prev_orient),
        "face_senders_next_orient": jnp.asarray(face_senders_next_orient),
        "face_receivers_prev_orient": jnp.asarray(face_receivers_prev_orient),
        "face_receivers_next_orient": jnp.asarray(face_receivers_next_orient),

        "face_senders_orient": jnp.asarray(face_senders_orient),
        "face_receivers_orient": jnp.asarray(face_receivers_orient),
    })

    from typing import Dict, Any

    def pack_edges_topo(edges_topo: Dict[str, Any]):
        """
        Convert your original edges_topo frozendict into a single packed array `faces`
        with shape (E, 2, 6) containing per-edge information for the two faces:
          faces[e, f] = [ prev_edge_idx, next_edge_idx, prev_orient, next_orient, self_orient, compute_flag ]
        where f==0 is sender-side face, f==1 is receiver-side face.
        Returns a frozendict with key "faces" (dtype int32/float where appropriate) and
        keeps the original dict entries as well (optional).
        """
        # read inputs (may be Python lists or jnp arrays)
        fs_prev = jnp.asarray(edges_topo["face_senders_prev"], dtype=jnp.int32)
        fs_next = jnp.asarray(edges_topo["face_senders_next"], dtype=jnp.int32)
        fr_prev = jnp.asarray(edges_topo["face_receivers_prev"], dtype=jnp.int32)
        fr_next = jnp.asarray(edges_topo["face_receivers_next"], dtype=jnp.int32)

        # orientations: convert to int8 or int32 of +/-1
        fs_prev_or = jnp.asarray(edges_topo["face_senders_prev_orient"], dtype=jnp.int8)
        fs_next_or = jnp.asarray(edges_topo["face_senders_next_orient"], dtype=jnp.int8)
        fr_prev_or = jnp.asarray(edges_topo["face_receivers_prev_orient"], dtype=jnp.int8)
        fr_next_or = jnp.asarray(edges_topo["face_receivers_next_orient"], dtype=jnp.int8)

        fs_or = jnp.asarray(edges_topo["face_senders_orient"], dtype=jnp.int8)
        fr_or = jnp.asarray(edges_topo["face_receivers_orient"], dtype=jnp.int8)

        compute_fs = jnp.asarray(edges_topo["compute_face_senders"]).astype(jnp.int8)
        compute_fr = jnp.asarray(edges_topo["compute_face_receivers"]).astype(jnp.int8)

        # stack per-face arrays (E, 6)
        # For sender-face (f=0)
        face0 = jnp.stack([
            fs_prev,  # prev edge idx
            fs_next,
            fs_prev_or.astype(jnp.int32),
            fs_next_or.astype(jnp.int32),
            fs_or.astype(jnp.int32),
            compute_fs.astype(jnp.int32),
        ], axis=1)

        # For receiver-face (f=1)
        face1 = jnp.stack([
            fr_prev,
            fr_next,
            fr_prev_or.astype(jnp.int32),
            fr_next_or.astype(jnp.int32),
            fr_or.astype(jnp.int32),
            compute_fr.astype(jnp.int32),
        ], axis=1)

        # pack into shape (E, 2, 6)
        faces = jnp.stack([face0, face1], axis=1)

        # return packed dict; keep original for backwards compatibility if desired
        return frozendict({**edges_topo,"faces_packed": faces})

    edges_topo=pack_edges_topo(edges_topo)

    nodes = frozendict({"position": jnp.asarray(vertices),"flip_id":jnp.zeros((vertices.shape[0],))})
    edges = frozendict({})

    faces_dict = frozendict({
        "vertices": jnp.asarray(face_vertices),
    })

    globals_dict = frozendict({"l0":jnp.mean(jnp.linalg.norm(vertices[senders]-vertices[receivers],axis=1)),"V0":4*jnp.pi/3.0,"A0":4*jnp.pi})

    return {
        "nodes": nodes,
        "edges": edges,
        "edges_topo": edges_topo,
        "faces": faces_dict,

        "senders": jnp.asarray(senders),
        "receivers": jnp.asarray(receivers),

        "face_senders": jnp.asarray(face_senders),
        "face_receivers": jnp.asarray(face_receivers),
        "face_vertices":jnp.asarray(face_vertices),

        "globals": globals_dict,

        "n_node": jnp.asarray([n_nodes]),
        "n_edge": jnp.asarray([n_edges]),
        "n_face": jnp.asarray([n_faces]),
        "n_node_max": n_nodes,
        "n_edge_max": n_edges,

        "rng_key": jax.random.PRNGKey(42),
    }


def build_meshgraph_from_trimesh_old(mesh):
    """
    Build a fully consistent MeshGraph from a trimesh mesh.

    Enforces:
      - face_receiver is the face for which the primal edge sender→receiver
        follows the face’s CCW direction.

    Returns per-edge topology:
      - face_senders_prev, face_senders_next
      - face_receivers_prev, face_receivers_next
      - face_senders_prev_orient, face_senders_next_orient
      - face_receivers_prev_orient, face_receivers_next_orient
      - face_senders_orient, face_receivers_orient

    Also returns random-leader selection:
      - compute_face_senders, compute_face_receivers
    """



    # ------------------------------------------------------------
    # 0. Load arrays
    # ------------------------------------------------------------
    faces = np.asarray(mesh.faces, dtype=np.int32)         # (F,3)
    vertices = np.asarray(mesh.vertices, dtype=np.float32) # (V,3)

    edges_unique = np.asarray(mesh.edges_unique, dtype=np.int32)      # (E,2)
    edges_unique_inverse = np.asarray(mesh.edges_unique_inverse, dtype=np.int32)

    n_faces = faces.shape[0]
    n_edges = edges_unique.shape[0]
    n_nodes = vertices.shape[0]

    # ------------------------------------------------------------
    # 1. Directed per-face edges (cyclic CCW)
    # ------------------------------------------------------------
    face_vertices = faces.copy()
    face_edges_directed = np.stack([
        face_vertices[:, [0, 1]],
        face_vertices[:, [1, 2]],
        face_vertices[:, [2, 0]],
    ], axis=1)  # (F,3,2)

    # Unique edge index of each face-edge
    face_edges = edges_unique_inverse.reshape(n_faces, 3)

    # ------------------------------------------------------------
    # 2. Orientation of face-edges: does face-edge match the
    #    unique edge’s direction?
    # ------------------------------------------------------------
    ue0 = edges_unique[face_edges, 0]
    ue1 = edges_unique[face_edges, 1]

    fe0 = face_edges_directed[:, :, 0]
    fe1 = face_edges_directed[:, :, 1]

    face_edge_orientation = np.where(
        (fe0 == ue0) & (fe1 == ue1), 1, -1
    ).astype(np.int8)

    # ------------------------------------------------------------
    # 3. edge → face adjacency
    # ------------------------------------------------------------
    edge_faces = [[] for _ in range(n_edges)]
    for f in range(n_faces):
        for i in range(3):
            e = face_edges[f, i]
            edge_faces[e].append((f, i))

    face_senders = np.full(n_edges, -1, dtype=np.int32)
    face_receivers = np.full(n_edges, -1, dtype=np.int32)
    edge_face_positions = np.full((n_edges, 2), -1, dtype=np.int32)

    # ------------------------------------------------------------
    # 4. Primal edge directions (unmodified)
    # ------------------------------------------------------------
    senders = edges_unique[:, 0].copy()
    receivers = edges_unique[:, 1].copy()

    # ------------------------------------------------------------
    # 5. Determine dual sender/receiver faces
    # ------------------------------------------------------------
    for e in range(n_edges):

        flist = edge_faces[e]
        if len(flist) == 0:
            continue

        # Boundary: only sender face
        if len(flist) == 1:
            f0, _ = flist[0]
            face_senders[e] = f0
            continue

        (f0, p0), (f1, p1) = flist
        edge_face_positions[e] = [p0, p1]

        s = senders[e]
        r = receivers[e]

        fe0 = face_edges_directed[f0, p0]
        fe1 = face_edges_directed[f1, p1]

        m0 = (fe0[0] == s) and (fe0[1] == r)
        m1 = (fe1[0] == s) and (fe1[1] == r)

        if m0 and not m1:
            face_receivers[e] = f0
            face_senders[e]   = f1
        elif m1 and not m0:
            face_receivers[e] = f1
            face_senders[e]   = f0
        else:
            # fallback deterministic
            face_senders[e]   = f0
            face_receivers[e] = f1

    # ------------------------------------------------------------
    # 6. prev/next edges inside faces + their orientations
    # ------------------------------------------------------------
    face_prev  = np.roll(face_edges,            2, axis=1)
    face_next  = np.roll(face_edges,           -1, axis=1)
    face_prev_orient = np.roll(face_edge_orientation,  2, axis=1)
    face_next_orient = np.roll(face_edge_orientation, -1, axis=1)

    face_senders_prev = np.full(n_edges, -1, dtype=np.int32)
    face_senders_next = np.full(n_edges, -1, dtype=np.int32)
    face_receivers_prev = np.full(n_edges, -1, dtype=np.int32)
    face_receivers_next = np.full(n_edges, -1, dtype=np.int32)

    face_senders_prev_orient = np.zeros(n_edges, dtype=np.int8)
    face_senders_next_orient = np.zeros(n_edges, dtype=np.int8)
    face_receivers_prev_orient = np.zeros(n_edges, dtype=np.int8)
    face_receivers_next_orient = np.zeros(n_edges, dtype=np.int8)

    # New: Orientation of *this edge* inside each face
    face_senders_orient   = np.zeros(n_edges, dtype=np.int8)
    face_receivers_orient = np.zeros(n_edges, dtype=np.int8)

    # ------------------------------------------------------------
    # Populate arrays
    # ------------------------------------------------------------
    for e in range(n_edges):

        # ----- sender-side -----
        fs = face_senders[e]
        if fs >= 0:
            p = edge_face_positions[e, 0] if face_edges[fs, edge_face_positions[e,0]] == e else edge_face_positions[e,1]

            # neighbor edges
            face_senders_prev[e] = face_prev[fs, p]
            face_senders_next[e] = face_next[fs, p]

            # neighbor orientations
            face_senders_prev_orient[e] = face_prev_orient[fs, p]
            face_senders_next_orient[e] = face_next_orient[fs, p]

            # own orientation inside face
            face_senders_orient[e] = face_edge_orientation[fs, p]

        # ----- receiver-side -----
        fr = face_receivers[e]
        if fr >= 0:
            p = edge_face_positions[e, 0] if face_edges[fr, edge_face_positions[e,0]] == e else edge_face_positions[e,1]

            # neighbor edges
            face_receivers_prev[e] = face_prev[fr, p]
            face_receivers_next[e] = face_next[fr, p]

            # neighbor orientations
            face_receivers_prev_orient[e] = face_prev_orient[fr, p]
            face_receivers_next_orient[e] = face_next_orient[fr, p]

            # own orientation
            face_receivers_orient[e] = face_edge_orientation[fr, p]

    # --- BEFORE: you had per-edge neighbor-based selection (buggy) ---
    # rng = np.random.default_rng()
    # rand = rng.random(n_edges)
    # compute_face_senders = np.zeros(n_edges, dtype=bool)
    # compute_face_receivers = np.zeros(n_edges, dtype=bool)
    # ... per-edge loop that compared rand[e] with rand[p] and rand[n] ...
    # --- REPLACE WITH THE FOLLOWING ---

    # 1) one random value per edge
    rng = np.random.default_rng()
    rand = rng.random(n_edges)  # float array length E

    # 2) per-face leader selection: pick the edge (of the face's 3) with minimal rand
    # face_edges: (n_faces, 3)
    face_rand_triplet = rand[face_edges]  # (n_faces, 3)
    min_pos = np.argmin(face_rand_triplet, axis=1)  # (n_faces,) values in {0,1,2}
    leader_edge_per_face = face_edges[np.arange(n_faces), min_pos]  # (n_faces,) edge ids

    # 3) zero-initialize flags and set per-edge booleans using face->leader mapping
    compute_face_senders = np.zeros(n_edges, dtype=bool)
    compute_face_receivers = np.zeros(n_edges, dtype=bool)

    # For each edge e, it has at most two adjacent faces:
    #  - sender face = face_senders[e]  (may be -1 for boundary)
    #  - receiver face = face_receivers[e]  (may be -1)
    # Mark compute flag True if this edge is the leader for that face.
    for e in range(n_edges):
        fs = face_senders[e]
        fr = face_receivers[e]
        if fs >= 0 and leader_edge_per_face[fs] == e:
            compute_face_senders[e] = True
        if fr >= 0 and leader_edge_per_face[fr] == e:
            compute_face_receivers[e] = True

    # ------------------------------------------------------------
    # 8. Assemble final dictionary
    # ------------------------------------------------------------
    nodes = frozendict({"position": jnp.asarray(vertices)})

    faces_dict = frozendict({
        "vertices": jnp.asarray(face_vertices),
        "edges": jnp.asarray(face_edges),
        "edge_orientation": jnp.asarray(face_edge_orientation),

    })

    edges_topo_dict = {
        "compute_face_senders": jnp.asarray(compute_face_senders),
        "compute_face_receivers": jnp.asarray(compute_face_receivers),

        # prev/next neighbors
        "face_senders_prev": jnp.asarray(face_senders_prev),
        "face_senders_next": jnp.asarray(face_senders_next),
        "face_receivers_prev": jnp.asarray(face_receivers_prev),
        "face_receivers_next": jnp.asarray(face_receivers_next),

        # orientations of neighbors
        "face_senders_prev_orient": jnp.asarray(face_senders_prev_orient),
        "face_senders_next_orient": jnp.asarray(face_senders_next_orient),
        "face_receivers_prev_orient": jnp.asarray(face_receivers_prev_orient),
        "face_receivers_next_orient": jnp.asarray(face_receivers_next_orient),

        # orientation of *this edge* inside each face
        "face_senders_orient": jnp.asarray(face_senders_orient),
        "face_receivers_orient": jnp.asarray(face_receivers_orient),
    }


    from typing import Dict, Any

    def pack_edges_topo(edges_topo: Dict[str, Any]):
        """
        Convert your original edges_topo frozendict into a single packed array `faces`
        with shape (E, 2, 6) containing per-edge information for the two faces:
          faces[e, f] = [ prev_edge_idx, next_edge_idx, prev_orient, next_orient, self_orient, compute_flag ]
        where f==0 is sender-side face, f==1 is receiver-side face.
        Returns a frozendict with key "faces" (dtype int32/float where appropriate) and
        keeps the original dict entries as well (optional).
        """
        # read inputs (may be Python lists or jnp arrays)
        fs_prev = jnp.asarray(edges_topo["face_senders_prev"], dtype=jnp.int32)
        fs_next = jnp.asarray(edges_topo["face_senders_next"], dtype=jnp.int32)
        fr_prev = jnp.asarray(edges_topo["face_receivers_prev"], dtype=jnp.int32)
        fr_next = jnp.asarray(edges_topo["face_receivers_next"], dtype=jnp.int32)

        # orientations: convert to int8 or int32 of +/-1
        fs_prev_or = jnp.asarray(edges_topo["face_senders_prev_orient"], dtype=jnp.int8)
        fs_next_or = jnp.asarray(edges_topo["face_senders_next_orient"], dtype=jnp.int8)
        fr_prev_or = jnp.asarray(edges_topo["face_receivers_prev_orient"], dtype=jnp.int8)
        fr_next_or = jnp.asarray(edges_topo["face_receivers_next_orient"], dtype=jnp.int8)

        fs_or = jnp.asarray(edges_topo["face_senders_orient"], dtype=jnp.int8)
        fr_or = jnp.asarray(edges_topo["face_receivers_orient"], dtype=jnp.int8)

        compute_fs = jnp.asarray(edges_topo["compute_face_senders"]).astype(jnp.int8)
        compute_fr = jnp.asarray(edges_topo["compute_face_receivers"]).astype(jnp.int8)

        # stack per-face arrays (E, 6)
        # For sender-face (f=0)
        face0 = jnp.stack([
            fs_prev,  # prev edge idx
            fs_next,
            fs_prev_or.astype(jnp.int32),
            fs_next_or.astype(jnp.int32),
            fs_or.astype(jnp.int32),
            compute_fs.astype(jnp.int32),
        ], axis=1)

        # For receiver-face (f=1)
        face1 = jnp.stack([
            fr_prev,
            fr_next,
            fr_prev_or.astype(jnp.int32),
            fr_next_or.astype(jnp.int32),
            fr_or.astype(jnp.int32),
            compute_fr.astype(jnp.int32),
        ], axis=1)

        # pack into shape (E, 2, 6)
        faces = jnp.stack([face0, face1], axis=1)

        # return packed dict; keep original for backwards compatibility if desired
        return frozendict({"faces_packed": faces})

    edges_topo_dict_packed=pack_edges_topo(edges_topo_dict)

    print(edges_topo_dict_paced["faces_packed"])
    edges_dict = frozendict({
    })

    globals=frozendict({})

    return {
        "nodes": nodes,
        "edges": edges_dict,
        "edges_topo":edges_topo_dict_packed,
        "faces": faces_dict,
        "receivers": jnp.asarray(receivers),
        "senders": jnp.asarray(senders),


        "face_receivers": jnp.asarray(face_receivers),
        "face_senders": jnp.asarray(face_senders),
        "face_vertices": jnp.asarray(face_vertices),


        "face_edges": jnp.asarray(face_edges),
        "edge_face_index": jnp.asarray(edge_face_positions),
        "globals":globals,

        # counts
        "n_node": jnp.asarray([n_nodes]),
        "n_edge": jnp.asarray([n_edges]),
        "n_face": jnp.asarray([n_faces]),
        "rng_key": jax.random.PRNGKey(42),
        "n_node_max": n_nodes,
        "n_edge_max": n_edges,

    }







def mesh_geometry_fn(graph: gn_graph.MeshGraphsTuple) -> gn_graph.MeshGraphsTuple:
    def update_edge_fn_fused(edges, edges_topo, senders, receivers, globals_):
        """
        Vectorized, fused per-edge update for face geometry.
        Inputs:
          - edges: frozendict or dict with per-edge fields (kept and extended)
          - edges_topo: output of pack_edges_topo (must contain "faces_packed")
          - senders, receivers: dict-like with "position" arrays shape (E,3)
          - globals_: ignored here (kept for API compatibility)
        Returns:
          frozendict with added/updated edge fields:
            uij, lij, area, normal, phi, weight, area_face_local, cross_face_local
        """
        DTYPE = senders["position"].dtype
        EPS = jnp.array(1e-12, dtype=DTYPE)

        # positions and basic edge vector
        sender_pos = senders["position"]
        receiver_pos = receivers["position"]
        uij = receiver_pos - sender_pos  # (E,3)
        lij = jnp.linalg.norm(uij, axis=1)  # (E,)

        faces = edges_topo["faces_packed"]  # shape (E,2,6); ints
        # unpack
        prev_e = faces[..., 0].astype(jnp.int32)  # (E,2)
        next_e = faces[..., 1].astype(jnp.int32)  # (E,2)
        prev_or = faces[..., 2].astype(jnp.int32)  # (E,2) values ±1
        next_or = faces[..., 3].astype(jnp.int32)
        self_or = faces[..., 4].astype(jnp.int32)
        compute_f = faces[..., 5].astype(jnp.int32)  # (E,2) 0/1

        E = uij.shape[0]


        self_or_f = self_or.astype(DTYPE)
        u_self = uij[:, None, :] * self_or_f[..., None]

        # --- replace the u_next / u_prev gather part with this snippet ---

        # ensure orientations are floats (+1.0 or -1.0) for multiplication
        self_or_f = self_or.astype(DTYPE)  # shape (E,2)
        next_or_f = next_or.astype(DTYPE)
        prev_or_f = prev_or.astype(DTYPE)

        # safe gather indices (unchanged)
        next_idx_safe = jnp.where(next_e >= 0, next_e, 0)  # (E,2)
        prev_idx_safe = jnp.where(prev_e >= 0, prev_e, 0)

        # gather raw neighbor vectors (uij for prev/next edges)
        u_next_raw = uij[next_idx_safe]  # (E,2,3)
        u_prev_raw = uij[prev_idx_safe]  # (E,2,3)

        # APPLY orientation sign to make neighbor vectors point in face CCW direction
        # Expand orientation dims to multiply across the 3-vector
        u_next = u_next_raw * next_or_f[..., None]  # (E,2,3)
        u_prev = u_prev_raw * prev_or_f[..., None]  # (E,2,3)
        # (u_self already was multiplied by self_or earlier)

        # valid face if both neighbors exist (prev and next)
        valid_face = (prev_e >= 0) & (next_e >= 0)  # (E,2)
        valid_face_f = valid_face.astype(DTYPE)

        # compute cross product and face area
        cross_face = jnp.cross(u_self, u_next)  # (E,2,3)
        cross_face = cross_face * valid_face[..., None]  # mask invalid faces
        norm_cross = jnp.linalg.norm(cross_face, axis=2)  # (E,2)
        area_face_raw = 0.5 * norm_cross  # (E,2)

        # apply compute_flag: only edges responsible for face produce area/cross in area_face_local/cross_face_local
        compute_bool = compute_f.astype(bool)
        area_face_local_per_face = area_face_raw * compute_f.astype(DTYPE)  # (E,2)
        cross_face_local_per_face = cross_face * compute_f[..., None]  # (E,2,3)

        # Summed responsible-area and responsible-cross for this edge (either sender or receiver may be responsible)
        area_face_local = jnp.sum(area_face_local_per_face, axis=1)  # (E,)
        cross_face_local = jnp.sum(cross_face_local_per_face, axis=1)  # (E,3)

        # --- scatter responsible face contributions to neighbours (prev & next) ---
        # Each responsible face contributes its area and cross to its prev and next edges.
        # neighbors array shape (E, 2 faces, 2 neighbors)
        neigh = jnp.stack([prev_e, next_e], axis=-1)  # (E,2,2)
        neigh_safe = jnp.stack([prev_idx_safe, next_idx_safe], axis=-1)  # (E,2,2)

        # flatten for scatter
        neigh_flat = neigh_safe.reshape(-1)  # (E*4,)
        area_flat = area_face_local_per_face.reshape(
            -1)  # (E*2,) but careful: we need repeat for prev and next -> duplicate each face area twice
        # duplicate each face area to match (E*4,) ordering of neigh_flat:
        # area_face_local_per_face has layout (E,2) faces; repeating along neighbor axis:
        area_per_face_dup = jnp.repeat(area_face_local_per_face[..., None], 2, axis=2)  # (E,2,2)
        area_per_face_dup_flat = area_per_face_dup.reshape(-1)  # (E*4,)

        # cross similarly: (E,2,3) -> (E,2,2,3) -> flattened
        cross_per_face_dup = jnp.repeat(cross_face_local_per_face[..., None, :], 2, axis=2)  # (E,2,2,3)
        cross_per_face_dup_flat = cross_per_face_dup.reshape(-1, 3)  # (E*4,3)

        # mask invalid neighbor indices
        valid_neigh = neigh_flat >= 0
        idx_safe = jnp.where(valid_neigh, neigh_flat, 0)

        # safe scalar scatter (area)
        area_add = jnp.zeros((E,), dtype=DTYPE).at[idx_safe].add(jnp.where(valid_neigh, area_per_face_dup_flat, 0.0))

        # safe vector scatter (cross)
        zero_cross_base = jnp.zeros((E, 3), dtype=DTYPE)
        cross_add = zero_cross_base.at[idx_safe].add(jnp.where(valid_neigh[:, None], cross_per_face_dup_flat, 0.0))

        # total per-edge area and cross: own responsible face + added contributions from neighbor faces
        area_final = area_face_local + area_add  # (E,)
        cross_final = cross_face_local + cross_add  # (E,3)

        # compute per-face normals (for phi) from the two faces attached to this edge:
        # cross_face has shape (E,2,3) with contributions only for faces present (maybe zero)
        # compute unit normal per face safely:
        def safe_unit(v):
            nrm = jnp.linalg.norm(v, axis=-1)
            nrm_safe = nrm + EPS
            return v / nrm_safe[..., None]

        n_faces = safe_unit(cross_face)  # (E,2,3)
        # face 0 = sender-side, face 1 = receiver-side
        n_s = n_faces[:, 0, :]  # (E,3)
        n_r = n_faces[:, 1, :]  # (E,3)

        cosphi = jnp.clip(jnp.sum(n_s * n_r, axis=1), -1.0, 1.0)
        phi = jnp.arccos(cosphi)  # (E,)

        weight = lij * phi

        # final per-edge area scaling (match your original single scaling)
        area_edge = (1.0 / 6.0) * area_final

        normal_edge = cross_final  # you previously used raw cross vectors as "normal"

        def tether_potential(l, lc1, lc0, r):
            """
            Continuous tether potential T(l) from TriMem (Eq. 14).
            Vectorized, branch-free, JAX-compatible.

            Parameters
            ----------
            l : jnp.ndarray
                Edge lengths (E,)
            lc1 : float
                Lower threshold
            lc0 : float
                Upper threshold
            r : int
                Exponent of potential well

            Returns
            -------
            jnp.ndarray
                Potential values T(l) for each edge
            """

            # Region 1: l <= lc1  (repulsive barrier)
            mask1 = (l <= lc1)
            # T1(l) = e * l / (l - lc1) * l^(-r)
            T1 = jnp.exp( l / (l - lc1)) * (l ** (-r))

            # Region 2: l >= lc0  (stretching penalty)
            mask2 = (l >= lc0)
            # T2(l) = r/(r+1) * (l - lc0)^r
            T2 = (r ** (r + 1)) * (l - lc0) ** r

            # Interior region: 0
            return jnp.where(mask1, T1, jnp.where(mask2, T2, 0.0))

        e_tether=tether_potential(lij, lc1=0.4*globals_["l0"], lc0=1.6*globals_["l0"], r=2)
        e_tether=e_tether+(lij-globals_["l0"])**2*10

        # pack results into edges frozendict
        new_edges = {
            **edges,
            "uij": uij,
            "lij": lij,
            "e_tether":e_tether*10e3,
            "area": area_edge,
            "normal": normal_edge,
            "phi": phi,
            "weight": weight,
            "area_face_local": area_face_local,
            "cross_face_local": cross_face_local,
        }

        return frozendict(new_edges)

    def update_edge_fn(edges, edges_topo, senders, receivers, globals_):
        """
        Per-edge computation of face geometry with corrected normal & area scaling.

        Requires `edges_topo` to contain:
          - face_senders_prev/next, face_receivers_prev/next  (int arrays shape (E,))
          - face_senders_prev_orient/next_orient, face_receivers_prev_orient/next_orient (int8 ±1)
          - face_senders_orient, face_receivers_orient (int8 ±1)
          - compute_face_senders, compute_face_receivers (bool or int)
        """
        sender_pos = senders["position"]
        receiver_pos = receivers["position"]

        uij = receiver_pos - sender_pos
        lij = jnp.linalg.norm(uij, axis=1)

        # topo and flags as before...
        fs_prev = edges_topo["face_senders_prev"]
        fs_next = edges_topo["face_senders_next"]
        fr_prev = edges_topo["face_receivers_prev"]
        fr_next = edges_topo["face_receivers_next"]

        fs_prev_or = edges_topo["face_senders_prev_orient"]
        fs_next_or = edges_topo["face_senders_next_orient"]
        fr_prev_or = edges_topo["face_receivers_prev_orient"]
        fr_next_or = edges_topo["face_receivers_next_orient"]
        fs_or = edges_topo["face_senders_orient"]
        fr_or = edges_topo["face_receivers_orient"]

        compute_fs = edges_topo["compute_face_senders"]
        compute_fr = edges_topo["compute_face_receivers"]

        E = uij.shape[0]
        idx = jnp.arange(E)
        DTYPE = uij.dtype
        EPS = jnp.array(1e-12, dtype=DTYPE)

        # ---------------------------------------------------------
        # Build local face edge vectors in face-cyclic order.
        # For a given edge e in face f at position pos we want the
        # two consecutive edges (u_self, u_next) in face CCW order.
        # edges_topo must provide orient signs so that:
        #   u_local = uij[edge_index] * orient   --> points in face CCW direction
        # ---------------------------------------------------------
        def gather_face_pair(e, prev_e, next_e, prev_or, next_or, self_or):
            # Note: prev_e/next_e may be -1 for boundary; caller will guard.
            u_self = uij[e] * self_or  # this edge oriented to face CCW
            u_next = uij[next_e] * next_or
            return u_self, u_next

        # ---------------------------------------------------------
        # compute face area & cross using (u_self, u_next)
        # area = 0.5 * || cross(u_self, u_next) ||
        # return area (scalar), cross_vec (3,), and cotangents (3,)
        # ---------------------------------------------------------
        def compute_face(e, prev_e, next_e, prev_or, next_or, self_or):
            valid = (prev_e >= 0) & (next_e >= 0)

            def _body():
                ue, un = gather_face_pair(e, prev_e, next_e, prev_or, next_or, self_or)
                cross_vec = jnp.cross(ue, un)
                area = jnp.array(0.5, dtype=DTYPE) * jnp.linalg.norm(cross_vec)

                # safe cotangents using same geometry (use denom = ||cross|| or EPS)
                denom = jnp.linalg.norm(cross_vec)
                denom = denom + EPS

                # we also need the third edge (prev) for cots: prev = - (ue + un) in triangle loop
                # but easier: reconstruct prev = - (ue + un) because ue points v_i->v_{i+1}, un points v_{i+1}->v_{i+2}
              #  up = -(ue + un)
              #  cot0 = jnp.dot(un, -up) / denom  # opposite ue
              #  cot1 = jnp.dot(up, -ue) / denom  # opposite un
              #  cot2 = jnp.dot(ue, -un) / denom  # opposite up
                #cot = jnp.stack([cot0, cot1, cot2])
                return area, cross_vec #, cot

            def _zero():
                return jnp.array(0.0, dtype=DTYPE), jnp.zeros(3, dtype=DTYPE)

            return jax.lax.cond(valid, _body, _zero)

        # ---------------------------------------------------------
        # vmap compute for sender-side faces and receiver-side faces
        # The vmap returns area_fs/norm_fs/cot_fs arrays (E,)
        # where a responsible edge has area A_f, others 0.
        # ---------------------------------------------------------
        area_fs, cross_fs = jax.vmap(
            lambda e, p, n, po, no, so: compute_face(e, p, n, po, no, so),
            in_axes=(0, 0, 0, 0, 0, 0)
        )(idx, fs_prev, fs_next, fs_prev_or, fs_next_or, fs_or)

        area_fr, cross_fr = jax.vmap(
            lambda e, p, n, po, no, so: compute_face(e, p, n, po, no, so),
            in_axes=(0, 0, 0, 0, 0, 0)
        )(idx, fr_prev, fr_next, fr_prev_or, fr_next_or, fr_or)

        # ---------------------------------------------------------
        # local per-edge face contributions from responsible edges:
        # area_face holds A_f for responsible edges (sender or receiver) else 0
        # cross_face holds cross vector (2 * face normal magnitude) likewise
        # ---------------------------------------------------------
        area_face = jnp.where(compute_fs, area_fs, jnp.array(0.0, dtype=DTYPE)) + \
                    jnp.where(compute_fr, area_fr, jnp.array(0.0, dtype=DTYPE))

        cross_face = jnp.where(compute_fs[:, None], cross_fs, jnp.zeros((E, 3), dtype=DTYPE)) + \
                     jnp.where(compute_fr[:, None], cross_fr, jnp.zeros((E, 3), dtype=DTYPE))

        # ---------------------------------------------------------
        # safe scatter-add (handles scalar/vector values and -1 indices)
        # ---------------------------------------------------------
        def safe_scatter_add(target_idx, values):
            cond = target_idx >= 0
            idx_safe = jnp.where(cond, target_idx, 0)
            if values.ndim == 1:
                vals_safe = jnp.where(cond, values, jnp.zeros_like(values))
                base = jnp.zeros((E,), dtype=DTYPE)
            else:
                cond_exp = cond[:, None]
                vals_safe = jnp.where(cond_exp, values, jnp.zeros_like(values))
                base = jnp.zeros((E, values.shape[1]), dtype=DTYPE)
            return base.at[idx_safe].add(vals_safe)

        # ---------------------------------------------------------
        # scatter each responsible face area/cross to prev & next edges
        # After this each of 3 edges in a triangle will have A_f included in area_final
        # ---------------------------------------------------------
        area_add = jnp.zeros(E, dtype=DTYPE)
        area_add = area_add + safe_scatter_add(fs_prev, jnp.where(compute_fs, area_fs, jnp.array(0.0, dtype=DTYPE)))
        area_add = area_add + safe_scatter_add(fs_next, jnp.where(compute_fs, area_fs, jnp.array(0.0, dtype=DTYPE)))
        area_add = area_add + safe_scatter_add(fr_prev, jnp.where(compute_fr, area_fr, jnp.array(0.0, dtype=DTYPE)))
        area_add = area_add + safe_scatter_add(fr_next, jnp.where(compute_fr, area_fr, jnp.array(0.0, dtype=DTYPE)))

        cross_add = jnp.zeros((E, 3), dtype=DTYPE)
        cross_add = cross_add + safe_scatter_add(fs_prev, jnp.where(compute_fs[:, None], cross_fs,
                                                                    jnp.zeros((E, 3), dtype=DTYPE)))
        cross_add = cross_add + safe_scatter_add(fs_next, jnp.where(compute_fs[:, None], cross_fs,
                                                                    jnp.zeros((E, 3), dtype=DTYPE)))
        cross_add = cross_add + safe_scatter_add(fr_prev, jnp.where(compute_fr[:, None], cross_fr,
                                                                    jnp.zeros((E, 3), dtype=DTYPE)))
        cross_add = cross_add + safe_scatter_add(fr_next, jnp.where(compute_fr[:, None], cross_fr,
                                                                    jnp.zeros((E, 3), dtype=DTYPE)))

        # total (unscaled) per-edge area & cross sum: for an edge belonging to faces f,g, area_final = A_f + A_g
        area_final = area_face + area_add
        cross_final = cross_face + cross_add

        # compute phi from unit normals of face-local cross vectors
        def safe_unit(v):
            nrm = jnp.linalg.norm(v, axis=-1)
            nrm_safe = nrm + EPS
            return v / nrm_safe[:, None]

        n_s = safe_unit(cross_fs)
        n_r = safe_unit(cross_fr)
        cosphi = jnp.clip(jnp.sum(n_s * n_r, axis=1), -1.0, 1.0)
        phi = jnp.arccos(cosphi)
        weight = lij * phi

        # *** CORRECT single scaling: area_edge is area_final * (1/6) ***
        area_edge = (1.0 / 6.0) * area_final

        normal_edge = cross_final

        return frozendict({
            **edges,
            #"uij": uij,
            "lij": lij,
            "area": area_edge,  # <<< fixed: do NOT scale again
            "normal": normal_edge,
            #"phi": phi,
            "weight": weight,
            "e_tether":(lij-globals_["l0"])**2*1,
           # "area_face_local": area_face,
           # "cross_face_local": cross_face,
        })

    def update_node_fn(nodes, sent_edges, received_edges, globals_):
        DTYPE = nodes["position"].dtype
        EPS = jnp.array(1e-12, dtype=DTYPE)

        # per-vertex area (as before)
        area = sent_edges["area"] + received_edges["area"]  # (N,)

        # build vector sum of edge cross contributions = approx (2 * area_i * normal_i) maybe
        normal_vec = sent_edges["normal"] + received_edges["normal"]  # (N,3)

        # compute per-vertex normal unit vector safely
        normal_norm = jnp.linalg.norm(normal_vec, axis=1, keepdims=True)  # (N,1)
        normal_unit = normal_vec / (normal_norm + EPS)  # (N,3)

        # If you want a normal scaled by vertex area (useful for e.g. pressure), do:
        normal_area_scaled = normal_unit * area[:, None]  # (N,3)

        # Vertex contribution to volume: for closed surface,
        # V = (1/3) * sum_i ( normal_vec_i dot r_i )
        # where normal_vec_i is the *area-weighted outward normal vector* (i.e., cross_sum).
        # Here we use normal_area_scaled (area times unit normal) or directly use normal_vec depending on units.
        # Using normal_vec (un-normalized cross sums) is safe if consistent with your earlier conventions:
        volume_i = (1.0 / 3.0) * jnp.sum(normal_vec * nodes["position"], axis=1)  # (N,)

        # Mean curvature / curvature weight: keep your previous definition (but ensure phi/weight are positive)
        curvature = 0.25 * (sent_edges["weight"] + received_edges["weight"])

        return frozendict({
            **nodes,
            "area": area,
            "normal": normal_unit,  # unit normals per-vertex
            "normal_area": normal_area_scaled,  # useful if you need area-weighted normals
            "volume": volume_i,
            "curvature": curvature,
            "c2":curvature**2*300,
        })

    def update_global_fn(nodes, edges,faces, globals_):

        area=nodes["area"]
       # areag=faces["area"]*6
        volume=1/12*nodes["volume"]

        #the mean curvature H_i*A_i element (divide by vertex curvature to get local mean curvature)
        curvature=nodes["curvature"]


        c2=nodes["c2"]
        e_tether=edges["e_tether"]
        e_area=(area-globals_["A0"])**2*10e3
        e_volume = (volume - globals_["V0"])**2 * 10e3

        return frozendict({**globals_,"Rv":jnp.sqrt(1/(4*jnp.pi)*area),"curvature":curvature,"volume":volume,"area":area,"energy":e_tether+e_area+e_volume+c2})



    gn = gn_models.MeshGraphNetwork(
        update_edge_fn=update_edge_fn_fused,
        update_face_fn=None,
        update_edge_from_face_fn=None,
        update_node_fn=update_node_fn,
        update_global_fn=update_global_fn)

    return gn(graph)

def mesh_flip_fn(graph: gn_graph.MeshGraphsTuple,n_node_max,n_edge_max) -> gn_graph.MeshGraphsTuple:


    def update_edge_fn(edges, senders, receivers, globals_):
        rn_min=jnp.minimum(senders["rn_min"],receivers["rn_min"])
        return frozendict({**edges,"rn_min":rn_min})

    def update_node_fn(nodes, sent_edges, received_edges, globals_):
        rn_min=jnp.minimum(sent_edges["rn_min"],received_edges["rn_min"])
        rn_min_edge = jnp.minimum(sent_edges["rn"], received_edges["rn"])
        return frozendict({**nodes,"rn_min":rn_min,"rn_min_edge":rn_min_edge})

    def update_edge_flip_selection_fn(edges, senders, receivers, globals_):
        return frozendict({**edges,"flip_site":jnp.where(edges["rn"]==jnp.minimum(senders["rn_min_edge"],receivers["rn_min_edge"]),1,0)})



    def flip_topology_by_indexing(
            senders, receivers,
            face_senders, face_receivers,
            face_edges, face_vertices, face_orient,
            edge_flip_mask
    ):
        """
        Rotation-free flip using the computed col* indices (pos1/pos2 -> col1_s0/1/2, col2_r0/1/2).
        - senders/receivers: (n_edge,)
        - face_senders/face_receivers: (n_edge,) per-primal-edge adjacent faces
        - face_edges: (n_face,3) edge indices per face (in some ordering)
        - face_vertices: (n_face,3) vertex indices per face (matching the same ordering)
        - face_orient: (n_face,3) optional orientation per face (same shape as face_edges)
        - edge_flip_mask: (n_edge,) 0/1 mask of which primal edges to flip

        Returns updated (senders_new, receivers_new, face_senders_new, face_receivers_new,
                         face_edges_new, face_vertices_new, face_orient_new)
        """
        flip = (edge_flip_mask == 1)
        n_edge = senders.shape[0]
        row = jnp.arange(n_edge)

        # Faces of each edge (may contain -1 for boundary)
        f1 = face_senders  # (n_edge,)
        f2 = face_receivers

        # clamp negative faces to zero for indexing but NEVER write into them:
        f1s = jnp.where(f1 >= 0, f1, 0)
        f2s = jnp.where(f2 >= 0, f2, 0)


        # Snapshot originals (read-only)
        fe_old = face_edges.copy()
        fo_old = face_orient.copy()
        fv_old = face_vertices.copy()
        fs_old = face_senders
        fr_old = face_receivers

        # compute where edge i sits in its two faces
        fe1 = fe_old[f1s]
        fe2 = fe_old[f2s]
        pos1 = jnp.argmax(fe1 == row[:, None], axis=1)  # 0..2
        pos2 = jnp.argmax(fe2 == row[:, None], axis=1)

        # column indices
        left1 = (pos1 + 2) % 3
        center1 = pos1
        right1 = (pos1 + 1) % 3

        left2 = (pos2 + 2) % 3
        center2 = pos2
        right2 = (pos2 + 1) % 3

        # --- New diagonal endpoints (from opposite vertices) ---
        v1_other = fv_old[f1s, left1]  # shape (n_edge,)
        v2_other = fv_old[f2s, left2]

        senders_new = jnp.where(flip, v2_other, senders)
        receivers_new = jnp.where(flip, v1_other, receivers)

        # --- Now compute new fv entries (read-only from fv_old) ---
        fv_new = fv_old.copy()
        # place opposite from other face into the slot that will be between the two remaining edges:
        fv_new = fv_new.at[f1s, right1].set(jnp.where(flip, fv_old[f2s, left2], fv_old[f1s, right1]))
        fv_new = fv_new.at[f2s, right2].set(jnp.where(flip, fv_old[f1s, left1], fv_old[f2s, right2]))

        # --- Edge-slot mutual swaps (read from fe_old/fo_old, write to fe_new/fo_new) ---
        fe_new = fe_old.copy()
        fo_new = fo_old.copy()

        # Pair A swap: fe[f1,left1] <-> fe[f2,right2]
        a_f1_left = fe_old[f1s, left1]
        b_f2_right = fe_old[f2s, right2]

        a_o_f1_left = fo_old[f1s, left1]
        b_o_f2_right = fo_old[f2s, right2]

        fe_new = fe_new.at[f1s, left1].set(jnp.where(flip, b_f2_right, a_f1_left))
        fe_new = fe_new.at[f2s, right2].set(jnp.where(flip, a_f1_left, b_f2_right))

        fo_new = fo_new.at[f1s, left1].set(jnp.where(flip, b_o_f2_right, a_o_f1_left))
        fo_new = fo_new.at[f2s, right2].set(jnp.where(flip, a_o_f1_left, b_o_f2_right))

        # Pair B swap: fe[f1,right1] <-> fe[f2,left2]
        a_f1_right = fe_old[f1s, right1]
        b_f2_left = fe_old[f2s, left2]

        a_o_f1_right = fo_old[f1s, right1]
        b_o_f2_left = fo_old[f2s, left2]

        fe_new = fe_new.at[f1s, right1].set(jnp.where(flip, b_f2_left, a_f1_right))
        fe_new = fe_new.at[f2s, left2].set(jnp.where(flip, a_f1_right, b_f2_left))

        fo_new = fo_new.at[f1s, right1].set(jnp.where(flip, b_o_f2_left, a_o_f1_right))
        fo_new = fo_new.at[f2s, left2].set(jnp.where(flip, a_o_f1_right, b_o_f2_left))

        fs_new=face_senders
        fr_new=face_receivers

        # --------------------------------------------
        # Done — return updated arrays
        # --------------------------------------------
        return (
            senders_new, receivers_new,
            fs_new, fr_new,
            fe_new, fv_new, fo_new
        )



    def update_global_fn(nodes, edges,faces, globals_):
        return frozendict({**globals_})



    gn = gn_models.MeshGraphNetworkFlip(
        update_edge_fn=update_edge_fn_fused,
        update_edge_flip_selection_fn=update_edge_flip_selection_fn,
        update_face_fn=None,
        update_edge_from_face_fn=None,
        update_node_fn=update_node_fn,
        update_global_fn=None,
        flip_edge_fn=flip_topology_by_indexing,
        n_node_max=n_node_max,
    n_edge_max=n_edge_max)

    return gn(graph)


def set_system_state(static_graph, position):
    nodes = dict(static_graph.nodes)
    nodes["position"] = jnp.asarray(position)
    return static_graph._replace(nodes=frozendict(nodes))

# getting/setting static graphs subjected to autograd
def set_system_parameters(
    static_graph: gn_graph.MeshGraphsTuple) -> gn_graph.MeshGraphsTuple:
  """Sets the non-static parameters of the graph (momentum, position)."""
  nodes = frozendict({
      **static_graph.nodes
  })
  #nodes = static_graph.nodes.copy(position=position, momentum=momentum)
  return static_graph._replace(nodes=nodes)


def get_system_state(graph: gn_graph.MeshGraphsTuple) -> Tuple[jnp.ndarray, jnp.ndarray]:
  return graph.nodes["position"]


#def get_static_graph(graph: gn_graph.MeshGraphsTuple) -> gn_graph.MeshGraphsTuple:
 # """Returns the graph with the static parts of a system only."""
  #nodes = dict(graph.nodes)
  #del nodes["position"]
  #return graph._replace(nodes=frozendict(nodes))

def get_static_graph(graph: gn_graph.MeshGraphsTuple) -> gn_graph.MeshGraphsTuple:
    return graph   # DO NOT delete position!



#  HAMILTONIAN

def get_hamiltonian_from_state_fn(
    static_graph: gn_graph.MeshGraphsTuple,
    hamiltonian_from_graph_fn: Callable[[gn_graph.MeshGraphsTuple],gn_graph.MeshGraphsTuple],
    ) -> Callable[[jnp.ndarray], float]:
  """Returns fn such that fn(position, momentum) -> scalar Hamiltonian.

  Args:
      static_graph: `GraphsTuple` containing per-particle static parameters and
         connectivity, such as a full graph of the state can be build by calling
         `set_system_state(static_graph, position, momentum)`.
      hamiltonian_from_graph_fn: callable that given an input `GraphsTuple`
         returns a `GraphsTuple` with a "hamiltonian" field in the globals.

  Returns:
      Function that given a state (position, momentum) returns the scalar
      Hamiltonian.
  """

  def hamiltonian_from_state_fn(position):
    # Note we sum along the batch dimension to get the total energy in the batch
    # so get can easily get the gradient.
    graph = set_system_state(static_graph, position)
    output_graph = hamiltonian_from_graph_fn(graph)
    return output_graph.globals["energy"][0],output_graph

  return hamiltonian_from_state_fn

def get_state_derivatives_from_hamiltonian_fn(
    hamiltonian_from_state_fn: Callable[[np.ndarray], float],
    ) -> Callable[[np.ndarray], Tuple[ np.ndarray]]:
  """Returns fn(position, momentum, ...) -> (dposition_dt, dmomentum_dt).

  Args:
      hamiltonian_from_state_fn: Function that given a state
          (position, momentum)  returns the scalar Hamiltonian.

  Returns:
      Function that given a state (position, momentum) returns the time
      derivatives of the state (dposition_dt, dmomentum_dt) by applying
      Hamilton equations.

  """
  #hamiltonian_gradients_fn= jax.grad(hamiltonian_from_state_fn, argnums=[0, 1])
  hamiltonian_gradients_fn= jax.value_and_grad(hamiltonian_from_state_fn, argnums=[0],has_aux=True)

  #def state_derivatives_from_hamiltonian_fn(
  #    position: np.ndarray, momentum: np.ndarray
  #    ) -> Tuple[np.ndarray, np.ndarray]:
    # Take the derivatives against position and momentum.
  #  (H,graph),(dh_dposition, dh_dmomentum) = hamiltonian_gradients_fn(position, momentum)

    # Hamilton equations.
  #  dposition_dt = dh_dmomentum
  #  dmomentum_dt = - dh_dposition
  #  return (H,graph),(dposition_dt, dmomentum_dt)

  def state_derivatives_from_hamiltonian_gd_fn(
      position: jnp.ndarray
      ) -> Tuple[np.ndarray]:
    # Take the derivatives against position and momentum.
      (H,graph),(dh_dposition,) = hamiltonian_gradients_fn(position)

    # Hamilton equations.
    #dposition_dt = dh_dmomentum
    #dmomentum_dt = - dh_dposition

      return (H,graph),(-dh_dposition)
  return state_derivatives_from_hamiltonian_gd_fn


StateDerivativesFnType = Callable[
    [ jnp.ndarray],
    Tuple[Tuple[float,jraph.GraphsTuple], Tuple[ jnp.ndarray]]
]



# Modification for Verlet
LangevinIntegratorType = Callable[
    [jnp.ndarray,  float, StateDerivativesFnType,float,float,jax.Array],
     Tuple[jnp.ndarray, jax.Array,gn_graph.MeshGraphsTuple]
]


def single_langevin_integration_step(
    graph: gn_graph.GraphsTuple, time_step: float,gamma:float,temperature:float,key:jax.Array,
    integrator_fn: LangevinIntegratorType,
    hamiltonian_from_graph_fn: Callable[[gn_graph.MeshGraphsTuple], gn_graph.MeshGraphsTuple],
    ) -> Tuple[float, gn_graph.MeshGraphsTuple,jax.Array,gn_graph.MeshGraphsTuple]:
  """Updates a graph state integrating by a single step.

  Args:
    graph: `GraphsTuple` representing a system state at time t.
    time_step: size of the timestep to integrate for.
    integrator_fn: Integrator to use. A function fn such that
       fn(position_t, momentum_t, time_step, state_derivatives_fn) ->
           (position_tp1, momentum_tp1)
    hamiltonian_from_graph_fn: Function that given a `GraphsTuple`, returns
        another one with a "hamiltonian" global field.

  Returns:
    `GraphsTuple` representing a system state at time `t + time_step`.

  """

  # Template graph with particle/interactions parameters and connectiviity
  # but without the state (position/momentum).
  static_graph = get_static_graph(graph)

  # Get the Hamiltonian function, and the function that returns the state
  # derivatives.
  hamiltonian_fn = get_hamiltonian_from_state_fn(
      static_graph=static_graph,
      hamiltonian_from_graph_fn=hamiltonian_from_graph_fn)
  state_derivatives_fn = get_state_derivatives_from_hamiltonian_fn(
      hamiltonian_fn)

  # Get the current state.

  position = get_system_state(graph)


  # Calling the integrator to get the next state.
  next_position,next_key,next_graph_passed = integrator_fn(
      position, time_step, state_derivatives_fn,gamma,temperature,key)
  next_graph_passed = set_system_state(next_graph_passed, next_position)

  # Return the energy of the next state too for plotting.
  energy = hamiltonian_fn(next_position)


  return energy, next_graph_passed,next_key,next_graph_passed


def brownian_dynamics_integrator(
        position: jnp.ndarray,
        time_step: float,
        state_derivatives_fn: StateDerivativesFnType,
        gamma: float,
        temperature: float,
        rng_key: jax.Array
) -> Tuple[jnp.ndarray,  jax.Array, jraph.GraphsTuple]:
    """Unified Langevin/Brownian dynamics integrator."""

    kB = 1.0  # Boltzmann constant

    sigma = jnp.sqrt(2 * kB * temperature / gamma)

    (H, next_graph), (dposition_dt) = state_derivatives_fn(position)
    rng_key, subkey = jax.random.split(rng_key)
    noise = jax.random.normal(subkey, shape=position.shape)

    next_position = position + (dposition_dt / gamma) * time_step + sigma * jnp.sqrt(time_step) * noise

    # Momentum is unchanged in overdamped limit
    return next_position, rng_key, next_graph





step_fn_graph = functools.partial(
        single_langevin_integration_step,
        hamiltonian_from_graph_fn=mesh_geometry_fn,
        integrator_fn=brownian_dynamics_integrator)

step_fn_graph = jax.jit(step_fn_graph)









next_key=jax.random.PRNGKey(42)
time_step=1e-6
temperature=0.001
gamma0=1.0



mesh = trimesh.creation.icosphere(subdivisions=7, radius=1.0)
t1=time.time()
graph = build_meshgraph_from_trimesh(mesh)



t2=time.time()
print(f"conversion: {t2-t1}")


#print(graph["edge_face_index"])


t1=time.time()
meshgraph=gn_graph.MeshGraphsTuple(**graph)
t2=time.time()
#check_mesh_consistency(meshgraph,  verbose=True)



# call directly (no jax.jit)
energy, returned_graph, next_key, returned_graph2 = step_fn_graph(meshgraph, time_step, gamma0, temperature, next_key)

print("TYPE next_position in graph nodes:", type(returned_graph.nodes["position"]), returned_graph.nodes["position"].dtype)
print("SHAPE next_position in graph nodes:", returned_graph.nodes["position"].shape)
print("Example next_position[0]:", returned_graph.nodes["position"][0])

# Also call integrator directly to inspect next_position:
static_graph = get_static_graph(meshgraph)
ham_fn = get_hamiltonian_from_state_fn(static_graph, mesh_geometry_fn)
state_deriv = get_state_derivatives_from_hamiltonian_fn(ham_fn)
next_position_from_integrator, rng_after, graph_from_integrator = brownian_dynamics_integrator(
    meshgraph.nodes["position"], time_step, state_deriv, gamma0, temperature, next_key)

print("next_position_from_integrator[0]:", next_position_from_integrator[0])
print("equal? graph.nodes['position'] == next_position_from_integrator:",
      jnp.allclose(returned_graph.nodes["position"], next_position_from_integrator))



#plot_meshgraph_3d(meshgraph)
print(f"construction: {t2-t1}")

t3=time.time()
mesh_geometry_jitted=jax.jit(mesh_geometry_fn)


meshgraph_save=meshgraph
t3=time.time()
dosteps=5000
for i in range(dosteps):
    _, _, key, meshgraph_new = step_fn_graph(meshgraph, time_step, gamma0, temperature, next_key)
    next_key, key = jax.random.split(key)
    meshgraph=meshgraph_new
t4=time.time()

print(f"ERROR: {jnp.sum(meshgraph_save.nodes["position"]-meshgraph.nodes["position"],axis=1)}")
print(f"time integration per step: {(t4-t3)/dosteps}")
#print(f"old mesh:\nvertices:{meshgraph.faces["vertices"]},edges:{meshgraph.faces["edges"]}")

t3=time.time()
mesh_flip_jitted=jax.jit(mesh_flip_fn,static_argnums=(1,2))

#meshgraph=mesh_flip_jm(meshgraph)

flip_site_proc=[]
n_node_max_static = int(meshgraph.n_node_max)
n_edge_max_static = int(meshgraph.n_edge_max)


print(f"nodes: {n_node_max_static}, edges: {n_edge_max_static},faces: {meshgraph.faces["vertices"].shape[0]} ")
#mesh_flip_jitted_static = functools.partial(
#    mesh_flip_jitted,
#    n_node_max=n_node_max_static,
#    n_edge_max=n_edge_max_static
#)

#def step(carry, _):
#    new_carry = mesh_flip_jitted_static(carry)
#    return new_carry, None
t4=time.time()
#meshgraph, _ = jax.lax.scan(step, meshgraph, xs=None, length=10000)
#for i in range(1):

#    meshgraph = mesh_flip_jitted(meshgraph,n_node_max_static,n_edge_max_static)

t5=time.time()

#print(f"new mesh:\nvertices:{meshgraph.faces["vertices"]},edges:{meshgraph.faces["edges"]}")
#check_mesh_consistency(meshgraph, verbose=True)
#flip_site_proc.append(meshgraph.nodes["flip_site"].sum())
print(f"calculation flip sites: {(t5-t4)/10000}")

#fig,ax=plt.subplots()
#plt.hist(flip_site_proc)
#plt.show()
#print(np.mean(flip_site_proc)/meshgraph.n_node_max)

#plot_meshgraph_3d(meshgraph)
dosteps_geom=10

plot_mesh_fast(meshgraph)

print(f"compilation: {t4-t3}")
t4=time.time()
for i in range(dosteps_geom):

    meshgraph=mesh_geometry_jitted(meshgraph)
t5=time.time()

print(f"calculation for: {(t5-t4)/dosteps_geom}")


def step(carry, _):
    new_carry = mesh_geometry_jitted(carry)
    return new_carry, None
t4=time.time()
meshgraph, _ = jax.lax.scan(step, meshgraph, xs=None, length=dosteps_geom)
t5=time.time()
print(f"calculation lax: {(t5-t4)/dosteps_geom}")





#print(meshgraph.edges)



print(f"compute sphere radius:\nvia vertex area: {meshgraph.globals["Rv"]}\nvia integrated curvature: {meshgraph.globals["curvature"]/(4*jnp.pi)}\nvia mean local curvature: {1/(jnp.mean(meshgraph.nodes["curvature"]/meshgraph.nodes["area"])),} ")

print("Error volume: ", meshgraph.globals["volume"]-meshgraph.globals["V0"])
print("Error area: ", meshgraph.globals["area"]-meshgraph.globals["A0"])
print("sum vertex area:", jnp.sum(meshgraph.nodes["area"]))
print("sphere area:", 4*jnp.pi)
print("sum volume:", jnp.sum(meshgraph.nodes["volume"]) / 3)
print("sphere volume:", 4/3*jnp.pi)
print(f"volume from global: {meshgraph.globals["volume"]}")
print(f"via volume: {(meshgraph.globals["volume"]/(4*jnp.pi/3))**(1/3)}")



edges_num = meshgraph.edges
edges_topo = meshgraph.edges_topo

# 1) face-local totals (responsible-edge contributions)
sum_face_local = float(jnp.sum(edges_num["area_face_local"]))
n_nonzero_face_local = int(jnp.sum(edges_num["area_face_local"] > 1e-12))

# 2) sum of (unscaled) cross-face magnitudes (sanity)
sum_cross_norm = float(jnp.sum(jnp.linalg.norm(edges_num["cross_face_local"], axis=1)))

# 3) edges area summary and relation to node areas
sum_edges_area = float(jnp.sum(edges_num["area"]))
sum_nodes_area = float(jnp.sum(meshgraph.nodes["area"]))

# 4) expected totals
expected_area = float(4.0 * jnp.pi)
expected_volume = float(4.0/3.0 * jnp.pi)

# 5) counts of responsible flags (should equal number of faces)
if edges_topo is not None:
    # count how many faces each edge claims (sum of booleans over edges) but we want per-face check
    count_compute_fs = int(jnp.sum(edges_topo["compute_face_senders"]))
    count_compute_fr = int(jnp.sum(edges_topo["compute_face_receivers"]))
else:
    count_compute_fs = count_compute_fr = None

print("=== Diagnostics ===")
print("expected total face area (4π):", expected_area)
print("sum_face_local (sum of A_f on responsible edges):", sum_face_local)
print("n_nonzero_face_local:", n_nonzero_face_local)
print("count_compute_face_senders:", count_compute_fs, " count_compute_face_receivers:", count_compute_fr," total (should be faces):", count_compute_fr+count_compute_fs)
print("sum_cross_norm (sanity):", sum_cross_norm)
print("--- edges/nodes area ---")
print("sum_edges_area (sum edges['area']):", sum_edges_area)
print("sum_nodes_area (sum nodes['area']):", sum_nodes_area)
print("sum_nodes_area / expected_area:", sum_nodes_area / expected_area)
print("sum_edges_area * 2 (should equal sum_nodes_area):", sum_edges_area * 2.0)
#print(np.mean(meshgraph.faces["area"]))