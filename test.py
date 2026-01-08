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
    Plot a MeshGraph in 3D without faces["vertices"].
    Face vertices are reconstructed from face_edges + edge_orientation.
    """

    # ---------------------------------------------------------
    # Convert JAX → numpy
    # ---------------------------------------------------------
    pos = np.asarray(mesh.nodes["position"])
    flip_id = np.asarray(mesh.nodes["flip_id"])

    senders   = np.asarray(mesh.senders)
    receivers = np.asarray(mesh.receivers)

    edge_flip = np.asarray(mesh.edges["flip_site"])

    face_senders   = np.asarray(mesh.face_senders)
    face_receivers = np.asarray(mesh.face_receivers)

    face_edges = np.asarray(mesh.faces["edges"])              # [n_faces, 3]
    face_ori   = np.asarray(mesh.faces["edge_orientation"])   # [n_faces, 3]

    n_faces = face_edges.shape[0]

    # ---------------------------------------------------------
    # 1) RECONSTRUCT FACE VERTICES (topological)
    # ---------------------------------------------------------

    def reconstruct_face_vertices(face_edges, face_ori, senders, receivers):
        """
        Robust reconstruction of triangle vertices from edges + orientations.
        Returns [n_faces, 3] array of vertex indices.
        """

        n_faces = face_edges.shape[0]
        face_vertices = np.empty((n_faces, 3), dtype=np.int32)

        for f in range(n_faces):
            e0, e1, e2 = face_edges[f]
            o0, o1, o2 = face_ori[f]

            # Edge 0 gives v0 -> v1
            if o0 == 1:
                v0 = senders[e0]
                v1 = receivers[e0]
            else:
                v0 = receivers[e0]
                v1 = senders[e0]

            # Edge 1 contains (v1, v2)
            s1 = senders[e1]
            r1 = receivers[e1]

            v2 = r1 if s1 == v1 else s1

            face_vertices[f] = [v0, v1, v2]

        return face_vertices

    def reconstruct_face_vertices_walk(face_edges, senders, receivers):
        """
        Reconstruct face vertices by walking the triangle.
        Does NOT assume cyclic order or orientation correctness.
        Safe for debugging / plotting.
        """
        n_faces = face_edges.shape[0]
        face_vertices = np.empty((n_faces, 3), dtype=np.int32)

        for f in range(n_faces):
            e0, e1, e2 = face_edges[f]

            # start with e0 arbitrarily
            a, b = senders[e0], receivers[e0]

            # find which edge connects to b
            if senders[e1] == b:
                c = receivers[e1]
            elif receivers[e1] == b:
                c = senders[e1]
            elif senders[e2] == b:
                c = receivers[e2]
            elif receivers[e2] == b:
                c = senders[e2]
            else:
                raise RuntimeError(f"Face {f} is not a closed triangle")

            face_vertices[f] = [a, b, c]

        return face_vertices

    face_vertices = reconstruct_face_vertices(
        face_edges,
        face_ori,
        senders,
        receivers
    )

    for f in range(len(face_vertices)):
        print(
            f,
            sorted(face_vertices[f]),
            sorted(reconstruct_face_vertices_walk(mesh.faces["edges"][f:f + 1], senders, receivers)[0])
        )
        break
    # ---------------------------------------------------------
    # 2) FACE CENTERS (dual vertices)
    # ---------------------------------------------------------

    def face_center_from_edges(face_edges, pos, senders, receivers):
        e0, e1, e2 = face_edges

        a, b = senders[e0], receivers[e0]

        # find c by walking
        for e in (e1, e2):
            s, r = senders[e], receivers[e]
            if s == b:
                c = r
                break
            if r == b:
                c = s
                break
        else:
            raise RuntimeError("Non-cyclic face")

        return (pos[a] + pos[b] + pos[c]) / 3

    face_centers = np.stack([
        face_center_from_edges(mesh.faces["edges"][f], pos, senders, receivers)
        for f in range(n_faces)
    ])

    dual_s = face_senders
    dual_r = face_receivers
    dual_mask = (dual_s >= 0) & (dual_r >= 0)
    dual_s = dual_s[dual_mask]
    dual_r = dual_r[dual_mask]

    # ---------------------------------------------------------
    # 3) Plot setup
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
    # 4) Primal semi-transparent faces
    # ---------------------------------------------------------
    face_tris = pos[face_vertices]   # [n_faces, 3, 3]

    tri_collection = Poly3DCollection(
        face_tris,
        facecolors=(0.4, 0.6, 1.0, 0.25),
        edgecolors=(0.2, 0.3, 0.5, 0.5),
        linewidths=0.5
    )
    ax.add_collection3d(tri_collection)

    # ---------------------------------------------------------
    # 5) Primal edges (colored by flip_site)
    # ---------------------------------------------------------
    norm = plt.Normalize(vmin=edge_flip.min(), vmax=edge_flip.max())
    cmap = plt.cm.viridis

    for s, r, val in zip(senders, receivers, edge_flip):
        p1 = pos[s]
        p2 = pos[r]
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color=cmap(norm(val)),
            linewidth=2,
            alpha=0.9
        )

    # ---------------------------------------------------------
    # 6) Dual mesh
    # ---------------------------------------------------------
    ax.scatter(
        face_centers[:, 0],
        face_centers[:, 1],
        face_centers[:, 2],
        c="red",
        s=20,
        alpha=0.8
    )

    # Optional debug highlighting
    fail = [32, 35, 37, 39, 42]
    ax.scatter(
        face_centers[fail, 0],
        face_centers[fail, 1],
        face_centers[fail, 2],
        c="yellow",
        s=60,
        alpha=0.8
    )

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

import numpy as np
import matplotlib.pyplot as plt


def plot_primal_dual_minimal(mesh):

    pos        = np.asarray(mesh.nodes["position"])
    senders    = np.asarray(mesh.senders)
    receivers  = np.asarray(mesh.receivers)
    edge_flip  = np.asarray(mesh.edges["flip_site"]).copy()

    face_edges     = np.asarray(mesh.faces["edges"])
    face_senders   = np.asarray(mesh.face_senders)
    face_receivers = np.asarray(mesh.face_receivers)

    # mask invalid dual edges
    mask = (face_senders >= 0) & (face_receivers >= 0)
    face_senders   = face_senders[mask]
    face_receivers = face_receivers[mask]

    # -------------------------------------------------------
    # reconstruct face centers via unique vertex extraction
    # -------------------------------------------------------
    face_centers = []
    for e0, e1, e2 in face_edges:
        vids = {
            senders[e0], receivers[e0],
            senders[e1], receivers[e1],
            senders[e2], receivers[e2],
        }
        if len(vids) != 3:
            print("DEBUG FACE:", e0, e1, e2)
            print("  edge", e0, "=", senders[e0], receivers[e0])
            print("  edge", e1, "=", senders[e1], receivers[e1])
            print("  edge", e2, "=", senders[e2], receivers[e2])
            edge_flip[e0]=0.5
            edge_flip[e1]=0.5
            edge_flip[e2]=0.5


            verts = {
                senders[e0], receivers[e0],
                senders[e1], receivers[e1],
                senders[e2], receivers[e2],
            }
            print("  unique vertices:", verts)
           # raise RuntimeError(f"Face edges {e0,e1,e2} do not define 3 unique vertices")

        face_centers.append(np.mean(pos[list(vids)], axis=0))

    face_centers = np.asarray(face_centers)

    # -------------------------------------------------------
    # plotting
    # -------------------------------------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # primal directed arrows
    for s, r, flip in zip(senders, receivers, edge_flip):
        p1, p2 = pos[s], pos[r]
        color = "yellow" if flip > 0 else "blue"

        ax.quiver(
            p1[0], p1[1], p1[2],
            *(p2 - p1),
            color=color,
            arrow_length_ratio=0.12,
            linewidth=1.5
        )

    # dual directed arrows
    for fs, fr in zip(face_senders, face_receivers):
        c1, c2 = face_centers[fs], face_centers[fr]

        ax.quiver(
            c1[0], c1[1], c1[2],
            *(c2 - c1),
            color="red",
            arrow_length_ratio=0.12,
            linewidth=1.5
        )

    # visualize primal nodes & face centers
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], c="black", s=20)
    ax.scatter(face_centers[:,0], face_centers[:,1], face_centers[:,2], c="red", s=30)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_box_aspect([1,1,1])
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




































# Tell tree_util how to navigate frozendicts.
jax.tree_util.register_pytree_node(
    frozendict,
    flatten_func=lambda s: (tuple(s.values()), tuple(s.keys())),
    unflatten_func=lambda k, xs: frozendict(zip(k, xs)))
import numpy as np
import jax.numpy as jnp
from frozendict import frozendict

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
    # Vertex neighbor list (1-ring), fixed size
    # ------------------------------------------------------------
    n_neigh_max = 8
    neigh_lists = [[] for _ in range(n_nodes)]

    for e in range(n_edges):
        i = senders[e]
        j = receivers[e]
        neigh_lists[i].append(j)
        neigh_lists[j].append(i)

    node_neighbors = np.full((n_nodes, n_neigh_max), -1, dtype=np.int32)
    for v in range(n_nodes):
        nl = list(dict.fromkeys(neigh_lists[v]))
        if len(nl) > n_neigh_max:
            raise ValueError(
                f"Vertex {v} valence {len(nl)} exceeds n_neigh_max={n_neigh_max}"
            )
        node_neighbors[v, :len(nl)] = nl

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

    face_senders = np.full(n_edges, -1, dtype=np.int32)
    face_receivers = np.full(n_edges, -1, dtype=np.int32)

    for e in range(n_edges):
        adj = edge_faces[e]

        if len(adj) == 1:
            # boundary edge
            f, _ = adj[0]
            face_receivers[e] = f  # arbitrary but consistent
            continue

        (f0, s0), (f1, s1) = adj
        u, v = senders[e], receivers[e]

        # check traversal direction in each face
        f0_forward = (
                face_edges_directed[f0, s0, 0] == u and
                face_edges_directed[f0, s0, 1] == v
        )
        f1_forward = (
                face_edges_directed[f1, s1, 0] == u and
                face_edges_directed[f1, s1, 1] == v
        )

        if f0_forward and not f1_forward:
            face_receivers[e] = f0
            face_senders[e] = f1
        elif f1_forward and not f0_forward:
            face_receivers[e] = f1
            face_senders[e] = f0
        else:
            raise ValueError(
                f"Inconsistent orientation on edge {e}: "
                f"faces {f0}, {f1} both forward or both backward"
            )

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








    nodes = frozendict({"position": jnp.asarray(vertices),"flip_id":jnp.zeros((vertices.shape[0],))})
    edges = frozendict({})

    faces_dict = frozendict({
        "vertices": jnp.asarray(face_vertices),
        "edges":jnp.asarray(face_edges),
        "edge_orientation":jnp.asarray(face_edge_orientation)
    })

    globals_dict = frozendict({"l0":jnp.mean(jnp.linalg.norm(vertices[senders]-vertices[receivers],axis=1)),
                               "V0":4*jnp.pi/3.0,
                               "A0":4*jnp.pi/n_faces,
                               "Av0":4*jnp.pi/n_nodes,
                               "kA":0.5,
                               "kV":0,
                               "kT":1.,
                               "kTh":2.,
                               "kB":40
                               })

    def check_dual_orientation(face_senders, face_receivers):
        e = np.arange(len(face_senders))
        return np.mean(face_senders[e] == face_receivers[e])

    assert np.all(face_senders >= -1)
    assert np.all(face_receivers >= -1)
    assert np.all(face_senders != face_receivers)

    node_mask=jnp.ones((n_nodes,))
    face_mask = jnp.ones((n_faces,))
    edge_mask = jnp.ones((n_edges,))

    nodes = frozendict({**nodes, "mask": node_mask,"area":jnp.zeros((n_nodes,)),"volume":jnp.zeros((n_nodes,)),"curvature":jnp.zeros((n_nodes,)),"c2":jnp.zeros((n_nodes,)),"energy":jnp.zeros((n_nodes,)),"normal":jnp.zeros((n_nodes,3)),"neighbors": jnp.asarray(node_neighbors)})
    edges = frozendict({**edges, "mask": edge_mask,"uij":jnp.ones((n_edges,3)),"lij":jnp.ones((n_edges,)),"area":jnp.ones((n_edges,)),"weight":jnp.ones((n_edges,)),"normal":jnp.ones((n_edges,3)),"energy":jnp.ones((n_edges,))})
    faces_dict = frozendict({**faces_dict, "mask": face_mask,"area":jnp.zeros((n_faces,)),"normal":jnp.zeros((n_faces,3)),"energy":jnp.zeros((n_faces,))})

    return {
        "nodes": nodes,
        "edges": edges,

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
        "n_face_max": n_faces,

        "rng_key": jax.random.PRNGKey(43),
    }




##### HELPER FUNCTIONS:

#### MODULE LAYOUT
# mesh_ops.py

# ---------- Edge geometry ----------
def edge_vectors(pos_s, pos_r):
    """Compute edge vector u_ij and length l_ij."""
    uij = pos_r - pos_s
    lij = jnp.linalg.norm(uij, axis=1)
    return uij, lij

# ---------- Face geometry ----------
def face_normal_and_area(uij):
    """uij: (F,3,3) edge vectors"""
    e0 = uij[:, 0]
    e1 = uij[:, 1]
    e2 = uij[:, 2]
    cross = jnp.cross(e0, -e2)
    area = 0.5 * jnp.linalg.norm(cross, axis=1)
    return area, cross

# ---------- Dihedral angle + edge weight ----------
def edge_dihedral_and_weight(face_normal_s, face_normal_r, lij):
    fsn = face_normal_s / jnp.linalg.norm(face_normal_s, axis=1)[:, None]
    fsr = face_normal_r / jnp.linalg.norm(face_normal_r, axis=1)[:, None]

    cosphi = jnp.sum(fsr * fsn, axis=1)
    cosphi = jnp.clip(cosphi, -1.0 + 1e-7, 1.0 - 1e-7)

    phi = jnp.arccos(cosphi)
    weight = lij * phi
    return phi, weight

# ---------- Node curvature, normal, volume ----------
def node_geometry(sent_area, recv_area, sent_normal, recv_normal, sent_weight, recv_weight, pos):
    area = sent_area + recv_area

    normal = sent_normal + recv_normal
    normal = normal / (jnp.linalg.norm(normal, axis=1)[:, None] + 1e-12)
    normal = normal * area[:, None]

    volume = -jnp.sum(normal * pos, axis=1)

    curvature = 0.25 * (sent_weight + recv_weight)
    c2 = curvature * curvature

    return area, normal, volume, curvature, c2


#######   ENERGY FUNCTIONS

## edge tether potential
def tether_potential_trimem(l, lc1, lc0, r):
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
            T1 = jnp.exp(l / (l - lc1)) * (l ** (-r))

            # Region 2: l >= lc0  (stretching penalty)
            mask2 = (l >= lc0)
            # T2(l) = r/(r+1) * (l - lc0)^r
            T2 = (r ** (r + 1)) * (l - lc0) ** r

            # Interior region: 0
            return jnp.where(mask1, T1, jnp.where(mask2, T2, 0.0))



def tether_potential(l, l0=1, kT=5.0, kTh=20.0):
    d = l - 1
    return 0.5 * kTh * d*d + 0.25 * kTh * d*d*d*d





def mesh_geometry_fn(graph):

    def update_edge_fn(edges, senders, receivers, globals_):
        edge_mask=edges["mask"]
        pos_s = senders["position"]
        pos_r = receivers["position"]

        uij, lij = edge_vectors(pos_s, pos_r)

        return frozendict({
            **edges,
            "uij": jnp.where(edge_mask[:, None], uij, edges["uij"]),
            "lij": jnp.where(edge_mask, lij, edges["lij"]),
        })

    def update_face_fn(faces, edges, sign, globals_):
        face_mask=faces["mask"]


        uij = edges["uij"] * sign[:, :, None]

        area, cross = face_normal_and_area(uij)


        energy = globals_["kA"]*(area/globals_["A0"]-1)**2



        return frozendict({
            **faces,
            "area": jnp.where(face_mask, area / 6.0, faces["area"]),
            "normal": jnp.where(face_mask[:, None], cross, faces["normal"]),
            "energy": jnp.where(face_mask, energy, faces["energy"]),
        })

    def update_edge_from_face_fn(edges, face_senders, face_receivers, globals_):
        edge_mask=edges["mask"]

        phi, weight = edge_dihedral_and_weight(
            face_senders["normal"],
            face_receivers["normal"],
            edges["lij"]
        )

        area = face_senders["area"] + face_receivers["area"]
        normal = face_senders["normal"] + face_receivers["normal"]

        lijs = edges["lij"] / globals_["l0"]

        energy = tether_potential(lijs, l0=globals_["l0"], kT=globals_["kT"], kTh=globals_["kTh"])
        #energy = tether_potential(lijs, lc1=0.3, lc0=1.7, r=2)
       # energy = energy * globals_["kT"] + (lijs - 1) ** 2 * 0.5*globals_["kTh"]*1e-2

        return frozendict({
            **edges,
            "area": jnp.where(edge_mask, area, edges["area"]),
            "weight": jnp.where(edge_mask, weight, edges["weight"]),
            "normal": jnp.where(edge_mask[:, None], normal, edges["normal"]),
            "energy": jnp.where(edge_mask, energy, edges["energy"]),
        })

    def update_node_fn(nodes, sent_edges, received_edges, globals_):
        node_mask=nodes["mask"]

        area, normal, volume, curvature, c2 = node_geometry(
            sent_edges["area"], received_edges["area"],
            sent_edges["normal"], received_edges["normal"],
            sent_edges["weight"], received_edges["weight"],
            nodes["position"]
        )
        #sel=jnp.zeros((nodes["position"].shape[0],))
        #sel.at[0].set(1)
        #pull=jnp.where(sel,-50.0*nodes["position"][jnp.arange((nodes["position"].shape[0])),0],jnp.zeros((nodes["position"].shape[0],)))

        def gravity(position, G=0.0):
            pos_z = position[:, 2]
            return G * pos_z

        def wca_wall_energy(
                x,  # (3,) particle position
                z0=-1.1,
                sigma=0.1,  # effective particle radius
                epsilon=0.2,  # wall strength
        ):
            """
            WCA hard wall at z = 0.

            Returns wall energy for a single particle.
            """
            z = x[:, 2] - z0

            zc = (2.0 ** (1.0 / 6.0)) * sigma

            def active():
                inv = sigma / z
                inv6 = inv ** 6
                return 4.0 * epsilon * (inv6 ** 2 - inv6) + epsilon

            return jnp.where(
                z < zc,
                active(),
                0.0,
            )

        gravity = gravity(nodes["position"])
        floor = wca_wall_energy(nodes["position"])

        energy=-globals_["kV"]*volume+2*globals_["kB"]*c2/area*globals_["Av0"]+gravity+floor

        return frozendict({
            **nodes,
            "area": jnp.where(node_mask, area, nodes["area"]),
            "normal": jnp.where(node_mask[:, None], normal, nodes["normal"]),
            "volume": jnp.where(node_mask, volume, nodes["volume"]),
            "curvature": jnp.where(node_mask, curvature, nodes["curvature"]),
            "c2": jnp.where(node_mask, c2, nodes["c2"]),
            "energy": jnp.where(node_mask, energy, nodes["energy"]),
        })

    def update_global_fn(nodes, edges, faces, globals_):
        area = nodes["area"]
        areag = faces["area"] * 6
        volume = nodes["volume"] / 3
        curvature = nodes["curvature"]
        c2 = nodes["c2"]

        energy=nodes["energy"]+edges["energy"]+faces["energy"]+(volume-globals_["V0"])**2*1


        return frozendict({
            **globals_,
            "Rv": jnp.sqrt(1/(4*jnp.pi) * area),
            "Rf": jnp.sqrt(1/(4*jnp.pi) * areag),
            "curvature": curvature,
            "volume": volume,
            "energy": energy,
            "area": area,
        })

    gn = gn_models.MeshGraphNetwork(
        update_edge_fn=update_edge_fn,
        update_face_fn=update_face_fn,
        update_edge_from_face_fn=update_edge_from_face_fn,
        update_node_fn=update_node_fn,
        update_global_fn=update_global_fn,
    )

    return gn(graph)


def mesh_flip_fn(graph: gn_graph.MeshGraphsTuple,n_node_max,n_edge_max,n_face_max) -> gn_graph.MeshGraphsTuple:

    def update_edge_fn(edges, senders, receivers, globals_):
        # propagate vertex-min from node rn_min
        rn_min = jnp.minimum(senders["rn_min"], receivers["rn_min"])
       # flip = rn_min == edges["rn"]


        return frozendict({**edges, "rn_min": rn_min})

    def update_edge_from_face_fn(edges, face_senders, face_receivers, globals_):
        # propagate face-min

        rn_min = jnp.minimum(face_senders["rn_min"], face_receivers["rn_min"])

        flip = rn_min==edges["rn"]

        return frozendict({**edges, "rn_min":rn_min,"flip_site": flip.astype(jnp.int32)})

    def update_node_fn(nodes, sent_edges, received_edges, globals_):
        # nodes compute minimal rn from incident edges
        rn_min = jnp.minimum(sent_edges["rn_min"], received_edges["rn_min"])
        return frozendict({**nodes, "rn_min": rn_min})

    def update_face_fn(faces,face_edges,nodes,senders,receivers, globals_):

        fe = face_edges
        vertex_priority=nodes["rn_min"]
        ps = vertex_priority[senders[fe]]
        pr = vertex_priority[receivers[fe]]
        rn_min = jnp.min(jnp.concatenate([ps, pr], axis=1), axis=1)
        # faces compute min over 3 incident edges
        #rn_min = jnp.min(edges["rn_min_f"], axis=1)
        return frozendict({**faces, "rn_min": rn_min})

    def update_face_preselection_fn(faces,edges,globals_):

        rn_min=jnp.min(edges["rn_min"],axis=1)
        return frozendict({**faces, "rn_min": rn_min})

    def extract_flip_edges(flip_mask, K):
        flip_edges = jnp.nonzero(
            flip_mask,
            size=K,
            fill_value=-1
        )[0]

        valid = flip_edges >= 0
        safe_edges = jnp.where(valid, flip_edges, 0)

        return flip_edges, safe_edges, valid

    def build_masks(aggregate, n_nodes, n_edges, n_faces):
        quad = aggregate[:, :4]  # (K,4)
        e = aggregate[:, 4]  # (K,)
        f = aggregate[:, 5:7]  # (K,2)

        # -------------------------
        # node mask
        # -------------------------
        quad_nodes = quad.ravel()  # (K*4,)
        quad_valid = quad_nodes >= 0  # boolean mask
        node_mask = jnp.zeros(n_nodes, dtype=bool).at[
            quad_nodes * quad_valid  # invalid entries become zero
            ].set(True)

        # -------------------------
        # edge mask
        # -------------------------
        edge_valid = e >= 0
        edge_mask = jnp.zeros(n_edges, dtype=bool).at[
            e * edge_valid
            ].set(True)

        # -------------------------
        # face mask
        # -------------------------
        f_ids = f.ravel()
        face_valid = f_ids >= 0
        face_mask = jnp.zeros(n_faces, dtype=bool).at[
            f_ids * face_valid
            ].set(True)

        return node_mask, edge_mask, face_mask

    def flip_edges_topology(
            senders,
            receivers,
            faces,
            face_senders,
            face_receivers,
            flip_edges,
            node_neighbors
    ):
        """
        Apply edge flips atomically.
        Assumes flip_edges is already a safe, non-interacting set.
        """

        faces_edges = faces["edges"]
        old_s = senders
        old_r = receivers

        # ------------------------------------------------------------
        # 0. Valid flip indices
        # ------------------------------------------------------------
        valid = flip_edges >= 0
        e = jnp.where(valid, flip_edges, -1)

        fL = jnp.where(valid, face_senders[e], -1)
        fR = jnp.where(valid, face_receivers[e], -1)
        valid = valid & (fL >= 0) & (fR >= 0)

        i = jnp.where(valid, old_s[e], -1)
        j = jnp.where(valid, old_r[e], -1)

        # ------------------------------------------------------------
        # 1. Opposing vertices k, l (READ-ONLY)
        # ------------------------------------------------------------
        def opposing_vertex(f_ids, a, b):
            fe = faces_edges[f_ids]  # (K,3)
            s = old_s[fe]
            r = old_r[fe]
            ov = jnp.where(
                (s != a[:, None]) & (s != b[:, None]), s,
                jnp.where((r != a[:, None]) & (r != b[:, None]), r, -1)
            )
            return ov.max(axis=1)

        k = opposing_vertex(fL, i, j)
        l = opposing_vertex(fR, i, j)

        valid = valid & (k >= 0) & (l >= 0)
        valid = valid & (k != l)

        # ------------------------------------------------------------
        # 2. Neighbor edges in the two faces (READ-ONLY)
        # ------------------------------------------------------------
        def find_edge_with(f_ids, a, b):
            fe = faces_edges[f_ids]
            s = old_s[fe]
            r = old_r[fe]
            mask = ((s == a[:, None]) & (r == b[:, None])) | \
                   ((s == b[:, None]) & (r == a[:, None]))
            idx = jnp.argmax(mask, axis=1)
            exists = jnp.any(mask, axis=1)
            return jnp.where(exists, fe[jnp.arange(fe.shape[0]), idx], -1)

        e_jk = find_edge_with(fL, j, k)
        e_ki = find_edge_with(fL, k, i)
        e_il = find_edge_with(fR, i, l)
        e_lj = find_edge_with(fR, l, j)

        valid = valid & (e_jk >= 0) & (e_ki >= 0) & (e_il >= 0) & (e_lj >= 0)

        # Mask everything consistently
        e = jnp.where(valid, e, -1)
        fL = jnp.where(valid, fL, -1)
        fR = jnp.where(valid, fR, -1)
        e_jk = jnp.where(valid, e_jk, -1)
        e_ki = jnp.where(valid, e_ki, -1)
        e_il = jnp.where(valid, e_il, -1)
        e_lj = jnp.where(valid, e_lj, -1)

        # ------------------------------------------------------------
        # 3. GATHER all updates into temporary arrays
        # ------------------------------------------------------------

        # Edge endpoints
        edge_sender_updates = jnp.full_like(senders, -1)
        edge_receiver_updates = jnp.full_like(receivers, -1)

        edge_sender_updates = edge_sender_updates.at[e].set(k)
        edge_receiver_updates = edge_receiver_updates.at[e].set(l)

        # Face edge lists
        face_edges_updates = jnp.full_like(faces_edges, -1)

        face_edges_updates = face_edges_updates.at[fL].set(
            jnp.stack([e, e_lj, e_jk], axis=1)
        )
        face_edges_updates = face_edges_updates.at[fR].set(
            jnp.stack([e, e_ki, e_il], axis=1)
        )

        # Face adjacency per edge
        face_senders_updates = jnp.full_like(face_senders, -1)
        face_receivers_updates = jnp.full_like(face_receivers, -1)

        face_senders_updates = face_senders_updates.at[e].set(fL)
        face_receivers_updates = face_receivers_updates.at[e].set(fR)

        def assign_face(edge_ids, old_face, new_face,
                        fsu, fru,
                        face_senders, face_receivers):
            mask = edge_ids >= 0

            fs_old = face_senders[edge_ids]
            fr_old = face_receivers[edge_ids]

            fs_new = jnp.where(mask & (fs_old == old_face), new_face, fs_old)
            fr_new = jnp.where(mask & (fr_old == old_face), new_face, fr_old)

            fsu = fsu.at[edge_ids].set(fs_new)
            fru = fru.at[edge_ids].set(fr_new)

            return fsu, fru

        face_senders_updates, face_receivers_updates = assign_face(
            e_jk, fL, fL, face_senders_updates, face_receivers_updates,face_senders,face_receivers
        )
        face_senders_updates, face_receivers_updates = assign_face(
            e_lj, fR, fL, face_senders_updates, face_receivers_updates,face_senders,face_receivers
        )
        face_senders_updates, face_receivers_updates = assign_face(
            e_ki, fL, fR, face_senders_updates, face_receivers_updates,face_senders,face_receivers
        )
        face_senders_updates, face_receivers_updates = assign_face(
            e_il, fR, fR, face_senders_updates, face_receivers_updates,face_senders,face_receivers
        )

        # ------------------------------------------------------------
        # 4. COMMIT all updates atomically
        # ------------------------------------------------------------
        new_senders = jnp.where(edge_sender_updates >= 0,
                                edge_sender_updates, senders)

        new_receivers = jnp.where(edge_receiver_updates >= 0,
                                  edge_receiver_updates, receivers)

        new_faces_edges = jnp.where(face_edges_updates >= 0,
                                    face_edges_updates, faces_edges)

        new_face_senders = jnp.where(face_senders_updates >= 0,
                                     face_senders_updates, face_senders)

        new_face_receivers = jnp.where(face_receivers_updates >= 0,
                                       face_receivers_updates, face_receivers)

        faces = frozendict({**faces, "edges": new_faces_edges})

        aggregate = jnp.stack([i, j, k, l, e, fL, fR], axis=1)
        aggregate = jnp.where(valid[:, None], aggregate, -1)

        return new_senders, new_receivers,faces,new_face_senders,new_face_receivers,aggregate,node_neighbors


    def flip_edges_topology(
            senders,
            receivers,
            faces,
            face_senders,
            face_receivers,
            flip_edges,
            node_neighbors
    ):
        faces_edges = faces["edges"]
        old_s = senders
        old_r = receivers

        valid = flip_edges >= 0
        e = jnp.where(valid, flip_edges, -1)

        fL = jnp.where(valid, face_senders[e], -1)
        fR = jnp.where(valid, face_receivers[e], -1)
        interior = (fL >= 0) & (fR >= 0)
        valid = valid & interior

        i = jnp.where(valid, old_s[e], -1)
        j = jnp.where(valid, old_r[e], -1)

        def opposing_vertex(f_ids, a, b):
            fe = faces_edges[f_ids]
            s = old_s[fe]
            r = old_r[fe]
            ov = jnp.where((s != a[:, None]) & (s != b[:, None]), s,
                           jnp.where((r != a[:, None]) & (r != b[:, None]), r, -1))
            return ov.max(axis=1)

        k = opposing_vertex(fL, i, j)
        l = opposing_vertex(fR, i, j)


        # tightened quad checks
        quad_ok = (
                (k >= 0) & (l >= 0) &
                (k != l) &
                (i != k) & (i != l) &
                (j != k) & (j != l)
        )
        valid = valid & quad_ok

        def find_edge_with(f_ids, a, b):
            fe = faces_edges[f_ids]
            s = old_s[fe]
            r = old_r[fe]
            mask = ((s == a[:, None]) & (r == b[:, None])) | \
                   ((s == b[:, None]) & (r == a[:, None]))
            idx = jnp.argmax(mask, axis=1)
            exists = jnp.any(mask, axis=1)
            edge = fe[jnp.arange(fe.shape[0]), idx]
            return jnp.where(exists, edge, -1)

        e_jk = find_edge_with(fL, j, k)
        e_ki = find_edge_with(fL, k, i)
        e_il = find_edge_with(fR, i, l)
        e_lj = find_edge_with(fR, l, j)

        neigh_ok = (e_jk >= 0) & (e_ki >= 0) & (e_il >= 0) & (e_lj >= 0)
        valid = valid & neigh_ok

        # NEW: detect degenerate "fan" configuration
        vj, vk = old_s[e_jk], old_r[e_jk]
        vk2, vi = old_s[e_ki], old_r[e_ki]
        vi2, vl = old_s[e_il], old_r[e_il]
        vl2, vj2 = old_s[e_lj], old_r[e_lj]

        fan_reject = (
                (i == k) | (i == l) |
                (j == k) | (j == l)
        )



        cycle_ok = ~fan_reject

        valid = valid & cycle_ok



        # mask all dependent indices
        e_jk = jnp.where(valid, e_jk, -1)
        e_ki = jnp.where(valid, e_ki, -1)
        e_il = jnp.where(valid, e_il, -1)
        e_lj = jnp.where(valid, e_lj, -1)

        v = valid
        vb = v[:, None]

        new_senders = senders.at[e].set(jnp.where(v, k, senders[e]))
        new_receivers = receivers.at[e].set(jnp.where(v, l, receivers[e]))

        new_faces_edges = faces_edges
        new_faces_edges = new_faces_edges.at[fL].set(
            jnp.where(vb, jnp.stack([e, e_lj, e_jk], axis=1), new_faces_edges[fL])
        )
        new_faces_edges = new_faces_edges.at[fR].set(
            jnp.where(vb, jnp.stack([e, e_ki, e_il], axis=1), new_faces_edges[fR])
        )



        def replace_role(fs, fr, edge_ids, old_face, new_face):
            valid_edge = edge_ids >= 0
            edge_ids = jnp.where(valid_edge, edge_ids, -1)

            fs_edge = fs[edge_ids]
            fr_edge = fr[edge_ids]

            match_fs = valid_edge & (fs_edge == old_face)
            match_fr = valid_edge & (fr_edge == old_face)

            fs_edge = jnp.where(match_fs, new_face, fs_edge)
            fr_edge = jnp.where(match_fr, new_face, fr_edge)

            fs = fs.at[edge_ids].set(jnp.where(valid_edge, fs_edge, fs[edge_ids]))
            fr = fr.at[edge_ids].set(jnp.where(valid_edge, fr_edge, fr[edge_ids]))
            return fs, fr

        new_fs = face_senders.at[e].set(jnp.where(v, fL, face_senders[e]))
        new_fr = face_receivers.at[e].set(jnp.where(v, fR, face_receivers[e]))

        new_fs, new_fr = replace_role(new_fs, new_fr, e_jk, fL, fL)
        new_fs, new_fr = replace_role(new_fs, new_fr, e_lj, fR, fL)
        new_fs, new_fr = replace_role(new_fs, new_fr, e_ki, fL, fR)
        new_fs, new_fr = replace_role(new_fs, new_fr, e_il, fR, fR)

        aggregate = jnp.stack([i, j, k, l, e, fL, fR], axis=1)
        aggregate = jnp.where(valid[:, None], aggregate, -1)

        faces = frozendict({**faces, "edges": new_faces_edges})

        return new_senders, new_receivers, faces, new_fs, new_fr, aggregate, node_neighbors

    def unique_owner(mask, face_ids, num_faces):
        """
        mask      : (K,) bool
        face_ids  : (K,) int, may contain -1
        returns   : (K,) bool — True only for the first owner of each face
        """
        valid = mask & (face_ids >= 0)

        # give each lane an id
        lane_id = jnp.arange(face_ids.shape[0])

        # for each face, find minimum lane id
        owner = jnp.full(num_faces, face_ids.shape[0], dtype=jnp.int32)
        owner = owner.at[face_ids].min(lane_id)

        # keep only the owner
        is_owner = valid & (lane_id == owner[face_ids])
        return is_owner


    def flip_edges_topology_realdeal(
            senders,
            receivers,
            faces,
            face_senders,
            face_receivers,
            flip_edges,
            node_neighbors,
    ):
        faces_edges = faces["edges"]
        old_s = senders
        old_r = receivers
        num_faces = faces_edges.shape[0]

        # ------------------------------------------------------------
        # 0. Basic validity
        # ------------------------------------------------------------
        valid = flip_edges >= 0
        e = jnp.where(valid, flip_edges, -1)

        fL = jnp.where(valid, face_senders[e], -1)
        fR = jnp.where(valid, face_receivers[e], -1)

        interior = (fL >= 0) & (fR >= 0)
        valid = valid & interior

        i = jnp.where(valid, old_s[e], -1)
        j = jnp.where(valid, old_r[e], -1)

        # ------------------------------------------------------------
        # 1. Opposing vertices
        # ------------------------------------------------------------
        def opposing_vertex(f_ids, a, b):
            fe = faces_edges[f_ids]
            s = old_s[fe]
            r = old_r[fe]
            ov = jnp.where(
                (s != a[:, None]) & (s != b[:, None]), s,
                jnp.where((r != a[:, None]) & (r != b[:, None]), r, -1)
            )
            return ov.max(axis=1)

        k = opposing_vertex(fL, i, j)
        l = opposing_vertex(fR, i, j)

        quad_ok = (
                (k >= 0) & (l >= 0) &
                (k != l) &
                (i != k) & (i != l) &
                (j != k) & (j != l)
        )
        valid = valid & quad_ok

        # ------------------------------------------------------------
        # 2. One flip per face (CRITICAL FIX)
        # ------------------------------------------------------------
        own_L = unique_owner(valid, fL, num_faces)
        own_R = unique_owner(valid, fR, num_faces)

        # flip must own BOTH faces
        valid = valid & own_L & own_R

        # ------------------------------------------------------------
        # 3. Neighbor edges (unchanged)
        # ------------------------------------------------------------
        def find_edge_with(f_ids, a, b):
            fe = faces_edges[f_ids]
            s = old_s[fe]
            r = old_r[fe]
            mask = ((s == a[:, None]) & (r == b[:, None])) | \
                   ((s == b[:, None]) & (r == a[:, None]))
            idx = jnp.argmax(mask, axis=1)
            exists = jnp.any(mask, axis=1)
            edge = fe[jnp.arange(fe.shape[0]), idx]
            return jnp.where(exists, edge, -1)

        e_jk = find_edge_with(fL, j, k)
        e_ki = find_edge_with(fL, k, i)
        e_il = find_edge_with(fR, i, l)
        e_lj = find_edge_with(fR, l, j)

        neigh_ok = (e_jk >= 0) & (e_ki >= 0) & (e_il >= 0) & (e_lj >= 0)
        valid = valid & neigh_ok

        # ------------------------------------------------------------
        # 4. Mask everything downstream
        # ------------------------------------------------------------
        v = valid
        vb = v[:, None]

        e = jnp.where(v, e, -1)
        fL = jnp.where(v, fL, -1)
        fR = jnp.where(v, fR, -1)
        e_jk = jnp.where(v, e_jk, -1)
        e_ki = jnp.where(v, e_ki, -1)
        e_il = jnp.where(v, e_il, -1)
        e_lj = jnp.where(v, e_lj, -1)

        # ------------------------------------------------------------
        # 5. Apply updates (NOW SAFE)
        # ------------------------------------------------------------
        new_senders = senders.at[e].set(jnp.where(v, k, senders[e]))
        new_receivers = receivers.at[e].set(jnp.where(v, l, receivers[e]))

        new_faces_edges = faces_edges
        new_faces_edges = new_faces_edges.at[fL].set(
            jnp.where(vb, jnp.stack([e, e_lj, e_jk], axis=1), new_faces_edges[fL])
        )
        new_faces_edges = new_faces_edges.at[fR].set(
            jnp.where(vb, jnp.stack([e, e_ki, e_il], axis=1), new_faces_edges[fR])
        )

        # ------------------------------------------------------------
        # 6. Edge–face adjacency updates (safe now)
        # ------------------------------------------------------------
        def replace_role(fs, fr, edge_ids, old_face, new_face):
            valid_edge = edge_ids >= 0
            fs_edge = fs[edge_ids]
            fr_edge = fr[edge_ids]

            fs_edge = jnp.where(valid_edge & (fs_edge == old_face), new_face, fs_edge)
            fr_edge = jnp.where(valid_edge & (fr_edge == old_face), new_face, fr_edge)

            fs = fs.at[edge_ids].set(fs_edge)
            fr = fr.at[edge_ids].set(fr_edge)
            return fs, fr

        new_fs = face_senders.at[e].set(jnp.where(v, fL, face_senders[e]))
        new_fr = face_receivers.at[e].set(jnp.where(v, fR, face_receivers[e]))

        new_fs, new_fr = replace_role(new_fs, new_fr, e_jk, fL, fL)
        new_fs, new_fr = replace_role(new_fs, new_fr, e_lj, fR, fL)
        new_fs, new_fr = replace_role(new_fs, new_fr, e_ki, fL, fR)
        new_fs, new_fr = replace_role(new_fs, new_fr, e_il, fR, fR)

        faces = frozendict({**faces, "edges": new_faces_edges})

        aggregate = jnp.stack([i, j, k, l, e, fL, fR], axis=1).astype(jnp.int32)
        aggregate = jnp.where(valid[:, None], aggregate, jnp.int32(-1))

        return new_senders, new_receivers, faces, new_fs, new_fr, aggregate, node_neighbors

    def update_edge_local_fn(edges,  senders, receivers,edge_mask, globals_):

        pos_s = senders["position"]
        pos_r = receivers["position"]

        uij, lij = edge_vectors(pos_s, pos_r)

        return frozendict({
            **edges,
            "uij": jnp.where(edge_mask[:, None], uij, edges["uij"]),
            "lij": jnp.where(edge_mask, lij, edges["lij"]),
        })

    def update_face_local_fn(faces, edges, sign,face_mask, globals_):

        uij = edges["uij"] * sign[:, :, None]

        area, cross = face_normal_and_area(uij)

        energy = globals_["kA"] * (area/globals_["A0"] - 1) ** 2

        return frozendict({
            **faces,
            "area": jnp.where(face_mask, area / 6.0, faces["area"]),
            "normal": jnp.where(face_mask[:, None], cross, faces["normal"]),
            "energy": jnp.where(face_mask, energy, faces["energy"]),
        })

    def update_edge_from_face_local_fn(edges, face_senders, face_receivers,edge_mask, globals_):

        phi, weight = edge_dihedral_and_weight(
            face_senders["normal"],
            face_receivers["normal"],
            edges["lij"]
        )

        area = face_senders["area"] + face_receivers["area"]
        normal = face_senders["normal"] + face_receivers["normal"]


        lijs = edges["lij"] / globals_["l0"]
        #energy = tether_potential(lijs, lc1=0.3, lc0=1.7, r=2)
        energy=tether_potential(lijs, l0=globals_["l0"], kT=globals_["kT"], kTh=globals_["kTh"])
        #energy = energy * globals_["kT"] + (lijs - 1) ** 2 * 0.5*globals_["kTh"]*1e-2

        return frozendict({
            **edges,
            "area": jnp.where(edge_mask, area, edges["area"]),
            "weight": jnp.where(edge_mask, weight, edges["weight"]),
            "normal": jnp.where(edge_mask[:, None], normal, edges["normal"]),
            "energy": jnp.where(edge_mask, energy, edges["energy"]),
        })

    def update_node_local_fn(nodes, sent_edges, received_edges,node_mask, globals_):

        area, normal, volume, curvature, c2 = node_geometry(
            sent_edges["area"], received_edges["area"],
            sent_edges["normal"], received_edges["normal"],
            sent_edges["weight"], received_edges["weight"],
            nodes["position"]
        )

        energy = -globals_["kV"] * volume + 2*globals_["kB"] * c2/area*globals_["Av0"]

        return frozendict({
            **nodes,
            "area": jnp.where(node_mask, area, nodes["area"]),
            "normal": jnp.where(node_mask[:, None], normal, nodes["normal"]),
            "volume": jnp.where(node_mask, volume, nodes["volume"]),
            "curvature": jnp.where(node_mask, curvature, nodes["curvature"]),
            "c2": jnp.where(node_mask, c2, nodes["c2"]),
            "energy": jnp.where(node_mask, energy, nodes["energy"]),
        })

    def metropolis_fn(
            flip_edges,
            E_before,
            E_after,
            beta,
            rng_key,
    ):
        """
        flip_edges:      (K,) int32, -1 padded
        E_before:        (K,) float
        E_after:         (K,) float
        beta:            float
        rng_key:         PRNGKey

        Returns:
            rollback_edges:      (K,) int32
            acceptance_fraction: float  in [0,1]
        """

        # which elements correspond to real flips
        valid = flip_edges >= 0

        dE = E_after - E_before
        rng_key, subkey = jax.random.split(rng_key)
        u = jax.random.uniform(subkey, shape=dE.shape)
      #  u = jax.random.uniform(rng_key, shape=dE.shape)

        # Metropolis rule
        accept = (dE <= 0.0) | (jnp.exp(-beta * dE) > u)

        # only count acceptances for real (non-padded) edges
        accept_valid = accept & valid

        # compute fraction
        n_valid = jnp.sum(valid)
        n_accept = jnp.sum(accept_valid)

        # safe division: if no valid flips -> 0.0
        acceptance_fraction = jnp.where(
            n_valid > 0,
            n_accept.astype(jnp.float32) / n_valid.astype(jnp.float32),
            0.0
        )

        # rejected flips retain edge id, accepted become -1
        reject = valid & (~accept)
        rollback_edges = jnp.where(reject, flip_edges, -1)

        return rollback_edges, acceptance_fraction,rng_key



    def update_global_fn(nodes, edges,faces, globals_):
        return frozendict({**globals_})



    gn = gn_models.MeshGraphNetworkFlip(
        # FLIPS SITE SELECTION
        update_edge_selection_fn=update_edge_fn,
        update_node_selection_fn=update_node_fn,
        update_face_selection_fn=update_face_fn,
        update_edge_from_face_selection_fn=update_edge_from_face_fn,
        update_face_preselection_fn=update_face_preselection_fn,
        # FLIP FUNCTION
        flip_edge_fn=flip_edges_topology_realdeal,
        #LOCAL ENERGY RECOMPUTE
        update_edge_local_fn=update_edge_local_fn,
        update_face_local_fn=update_face_local_fn,
        update_edge_from_face_local_fn=update_edge_from_face_local_fn,
        update_node_local_fn=update_node_local_fn,
        build_local_mask_fn=build_masks,
        #
        metropolis_fn=metropolis_fn,

        update_global_fn=None,

        n_node_max=n_node_max,
    n_edge_max=n_edge_max,
    n_face_max=n_face_max)

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





import numpy as np

def write_mesh_vtk(meshgraph, filename, binary=False):
    """
    Write meshgraph as VTK POLYDATA (.vtk), reconstructing face vertices
    from faces['edges'] and edges senders/receivers.

    No need for faces['vertices'].
    """

    pos          = np.asarray(meshgraph.nodes["position"])      # (N,3)
    face_edges   = np.asarray(meshgraph.faces["edges"])         # (F,3)
    senders      = np.asarray(meshgraph.senders)       # (E,)
    receivers    = np.asarray(meshgraph.receivers)     # (E,)

    # ---- reconstruct vertex triplets ----
    F = face_edges.shape[0]
    faces_vertices = np.empty((F, 3), dtype=np.int32)

    for f in range(F):
        e0, e1, e2 = face_edges[f]
        verts = [
            senders[e0], receivers[e0],
            senders[e1], receivers[e1],
            senders[e2], receivers[e2],
        ]

        # extract 3 unique vertex ids
        uniq = np.unique(verts)

        if uniq.shape[0] != 3:
            e0, e1, e2 = face_edges[f]
            msg = (
                f"Invalid face topology at face {f}\n"
                f"  edges: {e0}, {e1}, {e2}\n"
                f"  edge0: ({senders[e0]}, {receivers[e0]})\n"
                f"  edge1: ({senders[e1]}, {receivers[e1]})\n"
                f"  edge2: ({senders[e2]}, {receivers[e2]})\n"
                f"  vertices from edges: {verts}\n"
                f"  unique vertices extracted: {uniq}\n"
                f"  expected exactly 3 unique vertices for a triangular face."
            )
            raise RuntimeError(msg)

        faces_vertices[f] = uniq

    N = pos.shape[0]

    mode = "wb" if binary else "w"
    with open(filename, mode) as f:

        # Header
        header = (
            "# vtk DataFile Version 3.0\n"
            "MeshGraph export\n"
        )
        header += "BINARY\n" if binary else "ASCII\n"
        header += "DATASET POLYDATA\n"

        if binary:
            f.write(header.encode())
        else:
            f.write(header)

        # Points
        if binary:
            f.write(f"POINTS {N} float\n".encode())
            arr = pos.astype(">f4").tobytes()
            f.write(arr + b"\n")
        else:
            f.write(f"POINTS {N} float\n")
            for x, y, z in pos:
                f.write(f"{x} {y} {z}\n")

        # Polygons
        total_ints = F * 4
        if binary:
            f.write(f"POLYGONS {F} {total_ints}\n".encode())
            tris = np.hstack(
                [np.full((F,1), 3, np.int32), faces_vertices]
            ).astype(">i4")
            f.write(tris.tobytes() + b"\n")
        else:
            f.write(f"POLYGONS {F} {total_ints}\n")
            for (i, j, k) in faces_vertices:
                f.write(f"3 {i} {j} {k}\n")

    print(f"[VTK] wrote mesh to {filename} ({'binary' if binary else 'ascii'})")




### INTEGRATOR PARAMETERS

next_key=jax.random.PRNGKey(42)
time_step=1.e-6
temperature=0.00
gamma0=1.0


### BUILD INITIAL STATE FROM TRIMESH
mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)

#mesh = trimesh.creation.icosahedron(subdivisions=5, radius=1.0)


graph = build_meshgraph_from_trimesh(mesh)






meshgraph=gn_graph.MeshGraphsTuple(**graph)




###### SETUP INTEGRATOR
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



### CREATE ENERGY FUNCTION AND INITIALIZE GEOMETRY
mesh_geometry_jitted=jax.jit(mesh_geometry_fn)

meshgraph=mesh_geometry_jitted(meshgraph)


dosteps_geom=5
def step_geometry(carry, _):
    new_carry = mesh_geometry_jitted(carry)
    return new_carry, None


#meshgraph, _ = jax.lax.scan(step_geometry, meshgraph, xs=None, length=dosteps_geom)



globals_=meshgraph.globals

globals_=frozendict({**meshgraph.globals,
                     "V0":meshgraph.globals["volume"]*0.55,
                     "A0":jnp.mean(meshgraph.faces["area"]),
                     "l0":jnp.mean(meshgraph.edges['lij']),
                     "Av0":jnp.mean(meshgraph.nodes["area"])}

                    )
meshgraph._replace(globals=globals_)


#### BUILD FLIP FUNCTIONS

mesh_flip_jitted=jax.jit(mesh_flip_fn,static_argnums=(1,2,3))

#meshgraph=mesh_flip_jm(meshgraph)

flip_site_proc=[]
n_node_max_static = int(meshgraph.n_node_max)
n_edge_max_static = int(meshgraph.n_edge_max)
n_face_max_static = int(meshgraph.n_face_max)



print(f"nodes: {n_node_max_static}, edges: {n_edge_max_static},faces: {n_face_max_static} ")


mesh_flip_jitted_static = functools.partial(
    mesh_flip_jitted,
    n_node_max=n_node_max_static,
    n_edge_max=n_edge_max_static,
    n_face_max=n_face_max_static
)

def step_topology(carry, _):
    new_carry = mesh_flip_jitted_static(carry)
    return new_carry, None

#meshgraph, _ = jax.lax.scan(step, meshgraph, xs=None, length=10000)
for i in range(0):
    meshgraph = mesh_flip_jitted_static(meshgraph)#,n_node_max_static,n_edge_max_static)
    flip_site_proc.append(meshgraph.nodes["flip_site"].sum())




dosteps=10000
accep=0
#write_mesh_vtk(meshgraph, f"mesh_00000.vtu")
for i in range(dosteps):
    for j in range(1000):

       _, _, key, meshgraph = step_fn_graph(meshgraph, time_step, gamma0, temperature, next_key)
       next_key, key = jax.random.split(key)
       if j % 10==0:
      # for k in range(1):
           meshgraph = mesh_flip_jitted_static(meshgraph)
         #  meshgraph = mesh_geometry_jitted(meshgraph)

           accep+=meshgraph.globals["flips_acc"]



    #meshgraph, _ = jax.lax.scan(step_topology, meshgraph, xs=None, length=dosteps_10)
    #globals_=frozendict({**meshgraph.globals,"V0":0.999*meshgraph.globals["V0"]})
    #meshgraph._replace(globals=globals_)
    if i % 1 == 0:
       # if i==36:
       #     plot_primal_dual_minimal(meshgraph)print("selected edges per: ",meshgraph.globals["flips_per"]," accepted proposed flips per:",meshgraph.globals["flips_acc"])
        print("selected edges per: ",meshgraph.globals["flips_per"]," accepted proposed flips per:",meshgraph.globals["flips_acc"])

        print(f"E_b:{meshgraph.globals["E_b"]}, E_a:{meshgraph.globals["E_a"]}")
        write_mesh_vtk(meshgraph, f"mesh_{i:05d}.vtk",binary=False)



print("total accepted: ",accep)




dosteps_geom=0

#plot_mesh_fast(meshgraph)


t4=time.time()
for i in range(dosteps_geom):

    meshgraph=mesh_geometry_jitted(meshgraph)
t5=time.time()



#print(f"ERROR: {jnp.sum(meshgraph_save.nodes["position"]-meshgraph.nodes["position"],axis=1)}")
#print(f"time integration per step: {(t4-t3)/(dosteps+1)*1}")
#print(f"old mesh:\nvertices:{meshgraph.faces["vertices"]},edges:{meshgraph.faces["edges"]}")



#print(meshgraph.globals["rejected"])
#plot_meshgraph_3d(meshgraph)
#plot_primal_dual_minimal(meshgraph)
#print(f"new mesh:\nvertices:{meshgraph.faces["vertices"]},edges:{meshgraph.faces["edges"]}")
#ok,result=check_mesh_consistency_edge_centric(meshgraph)


#fig,ax=plt.subplots()
#plt.hist(flip_site_proc)
#plt.show()
print(f"percet flipped on average:{np.mean(flip_site_proc)/meshgraph.n_node_max}")


dosteps_geom=1

#plot_mesh_fast(meshgraph)


t4=time.time()
for i in range(dosteps_geom):

    meshgraph=mesh_geometry_jitted(meshgraph)
t5=time.time()

#print(f"calculation for: {(t5-t4)/dosteps_geom*10}")


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