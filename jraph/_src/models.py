# Copyright 2020 DeepMind Technologies Limited.


# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A library of Graph Neural Network models."""


from frozendict import frozendict

import functools
from typing import Any, Callable, Iterable, Mapping, Optional, Union

import jax
import jax.numpy as jnp
import jax.tree_util as tree
from jraph._src import graph as gn_graph
from jraph._src import utils

# As of 04/2020 pytype doesn't support recursive types.
# pytype: disable=not-supported-yet
ArrayTree = Union[jnp.ndarray, Iterable['ArrayTree'], Mapping[Any, 'ArrayTree']]

# All features will be an ArrayTree.
NodeFeatures = EdgeFeatures = SenderFeatures = ReceiverFeatures = FaceFeatures = FaceSenderFeatures = FaceReceiverFeatures = Globals = ArrayTree

# Signature:
# (edges of each node to be aggregated, segment ids, number of segments) ->
# aggregated edges
AggregateEdgesToNodesFn = Callable[
    [EdgeFeatures, jnp.ndarray, int], NodeFeatures]

# Signature:
# (edges of each face to be aggregated, segment ids, number of segments) ->
# aggregated edges
AggregateEdgesToFacesFn = Callable[
    [EdgeFeatures, jnp.ndarray, int], FaceFeatures]

# Signature:
# (nodes of each graph to be aggregated, segment ids, number of segments) ->
# aggregated nodes
AggregateNodesToGlobalsFn = Callable[[NodeFeatures, jnp.ndarray, int],
                                     Globals]

# Signature:
# (edges of each graph to be aggregated, segment ids, number of segments) ->
# aggregated edges
AggregateEdgesToGlobalsFn = Callable[[EdgeFeatures, jnp.ndarray, int],
                                     Globals]


AggregateFacesToGlobalsFn = Callable[[FaceFeatures, jnp.ndarray, int],
                                     Globals]
# Signature:
# (edge features, sender node features, receiver node features, globals) ->
# attention weights
AttentionLogitFn = Callable[
    [EdgeFeatures, SenderFeatures, ReceiverFeatures, Globals], ArrayTree]

# Signature:
# (edge features, weights) -> edge features for node update
AttentionReduceFn = Callable[[EdgeFeatures, ArrayTree], EdgeFeatures]

# Signature:
# (edges to be normalized, segment ids, number of segments) ->
# normalized edges
AttentionNormalizeFn = Callable[[EdgeFeatures, jnp.ndarray, int], EdgeFeatures]

# Signature:
# (edge features, sender node features, receiver node features, globals) ->
# updated edge features
GNUpdateEdgeFn = Callable[
    [EdgeFeatures, SenderFeatures, ReceiverFeatures, Globals], EdgeFeatures]


MGNUpdateFaceFn = Callable[
    [FaceFeatures, FaceSenderFeatures, FaceReceiverFeatures, Globals], FaceFeatures]


#MGNFlipEdgeFn = Callable[
#    [FaceFeatures, FaceSenderFeatures, FaceReceiverFeatures, Globals], FaceFeatures]
# Signature:
# (edge features, sender face features, receiver face features, globals) ->
# updated edge features (from adjacent faces)
MGNUpdateEdgeFromFaceFn = Callable[
    [EdgeFeatures, FaceSenderFeatures, FaceReceiverFeatures, Globals], EdgeFeatures]


# Signature:
# (node features, outgoing edge features, incoming edge features,
#  globals) -> updated node features
GNUpdateNodeFn = Callable[
    [NodeFeatures, SenderFeatures, ReceiverFeatures, Globals], NodeFeatures]

GNUpdateGlobalFn = Callable[[NodeFeatures, EdgeFeatures, Globals], Globals]


def MeshGraphNetwork(
    update_edge_fn: Optional[GNUpdateEdgeFn],
    update_face_fn: Optional[MGNUpdateFaceFn],
    update_edge_from_face_fn: Optional[MGNUpdateEdgeFromFaceFn],
    update_node_fn: Optional[GNUpdateNodeFn],
    update_global_fn: Optional[GNUpdateGlobalFn] = None,
    aggregate_edges_for_nodes_fn: AggregateEdgesToNodesFn = utils.segment_sum,
    aggregate_edges_for_faces_fn: AggregateEdgesToFacesFn = utils.segment_sum,
    aggregate_nodes_for_globals_fn: AggregateNodesToGlobalsFn = utils
    .segment_sum,
    aggregate_edges_for_globals_fn: AggregateEdgesToGlobalsFn = utils
    .segment_sum,
    aggregate_faces_for_globals_fn: AggregateFacesToGlobalsFn = utils.segment_sum,
    attention_logit_fn: Optional[AttentionLogitFn] = None,
    attention_normalize_fn: Optional[AttentionNormalizeFn] = utils
    .segment_softmax,
    attention_reduce_fn: Optional[AttentionReduceFn] = None):
  """Returns a method that applies a configured GraphNetwork.

  This implementation follows Algorithm 1 in https://arxiv.org/abs/1806.01261

  There is one difference. For the nodes update the class aggregates over the
  sender edges and receiver edges separately. This is a bit more general
  than the algorithm described in the paper. The original behaviour can be
  recovered by using only the receiver edge aggregations for the update.

  In addition this implementation supports softmax attention over incoming
  edge features.

  Example usage::

    gn = GraphNetwork(update_edge_function,
    update_node_function, **kwargs)
    # Conduct multiple rounds of message passing with the same parameters:
    for _ in range(num_message_passing_steps):
      graph = gn(graph)

  Args:
    update_edge_fn: function used to update the edges or None to deactivate edge
      updates.
    update_node_fn: function used to update the nodes or None to deactivate node
      updates.
    update_global_fn: function used to update the globals or None to deactivate
      globals updates.
    aggregate_edges_for_nodes_fn: function used to aggregate messages to each
      node.
    aggregate_nodes_for_globals_fn: function used to aggregate the nodes for the
      globals.
    aggregate_edges_for_globals_fn: function used to aggregate the edges for the
      globals.
    attention_logit_fn: function used to calculate the attention weights or
      None to deactivate attention mechanism.
    attention_normalize_fn: function used to normalize raw attention logits or
      None if attention mechanism is not active.
    attention_reduce_fn: function used to apply weights to the edge features or
      None if attention mechanism is not active.

  Returns:
    A method that applies the configured GraphNetwork.
  """
  not_both_supplied = lambda x, y: (x != y) and ((x is None) or (y is None))
  if not_both_supplied(attention_reduce_fn, attention_logit_fn):
    raise ValueError(('attention_logit_fn and attention_reduce_fn must both be'
                      ' supplied.'))

  def _ApplyMeshGraphNet(graph):
    """Applies a configured GraphNetwork to a graph.

    This implementation follows Algorithm 1 in https://arxiv.org/abs/1806.01261

    There is one difference. For the nodes update the class aggregates over the
    sender edges and receiver edges separately. This is a bit more general
    the algorithm described in the paper. The original behaviour can be
    recovered by using only the receiver edge aggregations for the update.

    In addition this implementation supports softmax attention over incoming
    edge features.

    Many popular Graph Neural Networks can be implemented as special cases of
    GraphNets, for more information please see the paper.

    Args:
      graph: a `GraphsTuple` containing the graph.

    Returns:
      Updated `GraphsTuple`.
    """
    # pylint: disable=g-long-lambda
    (nodes,
     edges,
     faces,
     globals_,
     receivers,
     senders,
     face_receivers,
     face_senders,
     face_edges,
     node_neighbors,
     node_mask,
     edge_mask,
     face_mask,
     n_node,
     n_edge,
     n_face,
     n_node_max,
     n_edge_max,
     n_face_max,
     rng_key)= graph
    # Equivalent to jnp.sum(n_node), but jittable
    #face_edges=faces["edges"]
    #face_vertices=faces["vertices"]
    sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
    sum_n_face = tree.tree_leaves(faces)[0].shape[0]
    sum_n_edge = senders.shape[0]
    if not tree.tree_all(
        tree.tree_map(lambda n: n.shape[0] == sum_n_node, nodes)):
      raise ValueError(
          'All node arrays in nest must contain the same number of nodes.')

    if not tree.tree_all(
        tree.tree_map(lambda n: n.shape[0] == sum_n_face, faces)):
      raise ValueError(
          'All face arrays in nest must contain the same number of faces.')

    sent_attributes = tree.tree_map(lambda n: n[senders], nodes)
    received_attributes = tree.tree_map(lambda n: n[receivers], nodes)

    # Here we scatter the global features to the corresponding edges,
    # giving us tensors of shape [num_edges, global_feat].
    global_edge_attributes = tree.tree_map(lambda g: jnp.repeat(
        g, n_edge, axis=0, total_repeat_length=sum_n_edge), globals_)

    if update_edge_fn:
      edges = update_edge_fn(edges, sent_attributes, received_attributes,
                             global_edge_attributes,edge_mask)

    if update_face_fn:
       # sent_face_attributes = tree.tree_map(
       #     lambda e: aggregate_edges_for_faces_fn(e, face_senders, sum_n_face), edges)


        received_edge_attributes = tree.tree_map(lambda n: n[face_edges], edges)
       #     lambda e: aggregate_edges_for_faces_fn(e, face_receivers, sum_n_face),
       #     edges)
        # Here we scatter the global features to the corresponding nodes,
        # giving us tensors of shape [num_nodes, global_feat].
        f = jnp.arange(face_edges.shape[0])[:, None]  # [n_faces, 1]
        #e = face_edges  # [n_faces, 3]

        sign = jnp.where(face_receivers[face_edges] == f, 1.0, -1.0)
        global_attributes = tree.tree_map(lambda g: jnp.repeat(
            g, n_face, axis=0, total_repeat_length=sum_n_face), globals_)
        faces = update_face_fn(faces,received_edge_attributes,sign,global_attributes,face_mask)

    if update_edge_from_face_fn:
      sent_face_attributes = tree.tree_map(lambda n: n[face_senders], faces)
      received_face_attributes = tree.tree_map(lambda n: n[face_receivers], faces)
      edges = update_edge_from_face_fn(edges, sent_face_attributes, received_face_attributes,
                             global_edge_attributes,edge_mask)

    if attention_logit_fn:
      logits = attention_logit_fn(edges, sent_attributes, received_attributes,
                                  global_edge_attributes)
      tree_calculate_weights = functools.partial(
          attention_normalize_fn,
          segment_ids=receivers,
          num_segments=sum_n_node)
      weights = tree.tree_map(tree_calculate_weights, logits)
      edges = attention_reduce_fn(edges, weights)

    if update_node_fn:
      sent_attributes = tree.tree_map(
          lambda e: aggregate_edges_for_nodes_fn(e, senders, sum_n_node), edges)
      received_attributes = tree.tree_map(
          lambda e: aggregate_edges_for_nodes_fn(e, receivers, sum_n_node),
          edges)
      # Here we scatter the global features to the corresponding nodes,
      # giving us tensors of shape [num_nodes, global_feat].
      global_attributes = tree.tree_map(lambda g: jnp.repeat(
          g, n_node, axis=0, total_repeat_length=sum_n_node), globals_)
      nodes = update_node_fn(nodes, sent_attributes,
                             received_attributes, global_attributes,node_mask)

    if update_global_fn:
      n_graph = n_node.shape[0]
      graph_idx = jnp.arange(n_graph)
      # To aggregate nodes and edges from each graph to global features,
      # we first construct tensors that map the node to the corresponding graph.
      # For example, if you have `n_node=[1,2]`, we construct the tensor
      # [0, 1, 1]. We then do the same for edges.
      node_gr_idx = jnp.repeat(
          graph_idx, n_node, axis=0, total_repeat_length=sum_n_node)
      edge_gr_idx = jnp.repeat(
          graph_idx, n_edge, axis=0, total_repeat_length=sum_n_edge)
      face_gr_idx = jnp.repeat(
          graph_idx, n_face, axis=0, total_repeat_length=sum_n_face)

      # We use the aggregation function to pool the nodes/edges per graph.
      node_attributes = tree.tree_map(
          lambda n: aggregate_nodes_for_globals_fn(n, node_gr_idx, n_graph),
          nodes)
      edge_attribtutes = tree.tree_map(
          lambda e: aggregate_edges_for_globals_fn(e, edge_gr_idx, n_graph),
          edges)
      face_attribtutes = tree.tree_map(
          lambda e: aggregate_faces_for_globals_fn(e, face_gr_idx, n_graph),
          faces)
      # These pooled nodes are the inputs to the global update fn.
      globals_ = update_global_fn(node_attributes, edge_attribtutes,face_attribtutes, globals_)
    # pylint: enable=g-long-lambda
    return gn_graph.MeshGraphsTuple(
        nodes=nodes,
        edges=edges,
        faces=faces,
        globals=globals_,

        receivers=receivers,
        senders=senders,
        face_receivers=face_receivers,
        face_senders=face_senders,
        face_edges=face_edges,
        node_neighbors=node_neighbors,
        node_mask=node_mask,
        edge_mask=edge_mask,
        face_mask=face_mask,

        n_node=n_node,
        n_edge=n_edge,
        n_face=n_face,

    n_node_max=n_node_max,
    n_edge_max=n_edge_max,
    n_face_max=n_face_max,
    rng_key=rng_key)

  return _ApplyMeshGraphNet



def MeshGraphNetworkFlip(
    update_edge_selection_fn: Optional[GNUpdateEdgeFn],
update_node_selection_fn: Optional[GNUpdateNodeFn],

update_face_preselection_fn: Optional[GNUpdateEdgeFn],
update_face_selection_fn=None,
        update_edge_from_face_selection_fn=None,
    update_global_fn: Optional[GNUpdateGlobalFn]=None,


    aggregate_edges_for_nodes_fn: AggregateEdgesToNodesFn = utils.segment_sum,
    aggregate_edges_for_faces_fn: AggregateEdgesToFacesFn = utils.segment_sum,
    aggregate_nodes_for_globals_fn: AggregateNodesToGlobalsFn = utils
    .segment_sum,
        aggregate_edges_for_nodes_rflood_fn =utils.segment_min,
    aggregate_edges_for_globals_fn: AggregateEdgesToGlobalsFn = utils
    .segment_sum,
    aggregate_faces_for_globals_fn: AggregateFacesToGlobalsFn = utils.segment_sum,
    attention_logit_fn: Optional[AttentionLogitFn] = None,
    attention_normalize_fn: Optional[AttentionNormalizeFn] = utils
    .segment_softmax,
    attention_reduce_fn: Optional[AttentionReduceFn] = None,
rng_key: Optional[jax.Array] = None,
        flip_edge_fn=None,
        update_node_local_fn =None,
        update_edge_local_fn =None,
        update_face_local_fn= None,
        update_edge_from_face_local_fn= None,
        build_local_mask_fn=None,
        metropolis_fn=None,


        n_node_max:int=None,
        n_edge_max:int=None,
        n_face_max:int=None,
):




  """Returns a method that applies a configured GraphNetwork.

  This implementation follows Algorithm 1 in https://arxiv.org/abs/1806.01261

  There is one difference. For the nodes update the class aggregates over the
  sender edges and receiver edges separately. This is a bit more general
  than the algorithm described in the paper. The original behaviour can be
  recovered by using only the receiver edge aggregations for the update.

  In addition this implementation supports softmax attention over incoming
  edge features.

  Example usage::

    gn = GraphNetwork(update_edge_function,
    update_node_function, **kwargs)
    # Conduct multiple rounds of message passing with the same parameters:
    for _ in range(num_message_passing_steps):
      graph = gn(graph)

  Args:
    update_edge_fn: function used to update the edges or None to deactivate edge
      updates.
    update_node_fn: function used to update the nodes or None to deactivate node
      updates.
    update_global_fn: function used to update the globals or None to deactivate
      globals updates.
    aggregate_edges_for_nodes_fn: function used to aggregate messages to each
      node.
    aggregate_nodes_for_globals_fn: function used to aggregate the nodes for the
      globals.
    aggregate_edges_for_globals_fn: function used to aggregate the edges for the
      globals.
    attention_logit_fn: function used to calculate the attention weights or
      None to deactivate attention mechanism.
    attention_normalize_fn: function used to normalize raw attention logits or
      None if attention mechanism is not active.
    attention_reduce_fn: function used to apply weights to the edge features or
      None if attention mechanism is not active.

  Returns:
    A method that applies the configured GraphNetwork.
  """


  not_both_supplied = lambda x, y: (x != y) and ((x is None) or (y is None))
  if not_both_supplied(attention_reduce_fn, attention_logit_fn):
    raise ValueError(('attention_logit_fn and attention_reduce_fn must both be'
                      ' supplied.'))



  def _ApplyMeshGraphNetFlip(graph,n_node_max,n_edge_max,n_face_max):


    """Applies a configured GraphNetwork to a graph.

    This implementation follows Algorithm 1 in https://arxiv.org/abs/1806.01261

    There is one difference. For the nodes update the class aggregates over the
    sender edges and receiver edges separately. This is a bit more general
    the algorithm described in the paper. The original behaviour can be
    recovered by using only the receiver edge aggregations for the update.

    In addition this implementation supports softmax attention over incoming
    edge features.

    Many popular Graph Neural Networks can be implemented as special cases of
    GraphNets, for more information please see the paper.

    Args:
      graph: a `GraphsTuple` containing the graph.

    Returns:
      Updated `GraphsTuple`.
    """



    # Example: generate a single random vector




    # pylint: disable=g-long-lambda
    (nodes,
     edges,
     faces,
     globals_,
     receivers,
     senders,
     face_receivers,
     face_senders,
     face_edges,
     node_neighbors,
     node_mask,
     edge_mask,
     face_mask,
     n_node,
     n_edge,
     n_face,
     _,
     _,
     _,
     rng_key) = graph

   # face_edges = faces["edges"]
   # node_neighbors=nodes["neighbors"]

    # Equivalent to jnp.sum(n_node), but
    rng_key, rng_sample = jax.random.split(rng_key)


    rn = jax.random.uniform(rng_sample, shape=(n_edge_max,))

    rng_key, rng_sample = jax.random.split(rng_key)

   # edges=frozendict({**edges,"rn":rn})

    #rng_key, rng_sample = jax.random.split(rng_key)
    #rn = jax.random.uniform(rng_sample, shape=(n_edge_max,))

    #nodes=frozendict({**nodes,"rn":rn,"rn_min":rn,"flip_id":jnp.zeros((n_node_max))})
    edges = frozendict({**edges, "rn": rn, "rn_min": rn})
    faces = frozendict({**faces, "rn_min": jnp.ones(n_face_max)*jnp.inf})
   # nodes = frozendict({**nodes, "rn_min": jnp.ones(n_node_max)*jnp.inf})
    flip_mask = edges["rn"] == edges["rn_min"]
    sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
    sum_n_face = tree.tree_leaves(faces)[0].shape[0]
    sum_n_edge = senders.shape[0]
    if not tree.tree_all(
        tree.tree_map(lambda n: n.shape[0] == sum_n_node, nodes)):
      raise ValueError(
          'All node arrays in nest must contain the same number of nodes.')

    if not tree.tree_all(
        tree.tree_map(lambda n: n.shape[0] == sum_n_face, faces)):
      raise ValueError(
          'All face arrays in nest must contain the same number of faces.')




    if update_face_preselection_fn:
        # sent_face_attributes = tree.tree_map(
        #     lambda e: aggregate_edges_for_faces_fn(e, face_senders, sum_n_face), edges)

        received_edge_attributes = tree.tree_map(lambda n: n[face_edges], edges)
        #     lambda e: aggregate_edges_for_faces_fn(e, face_receivers, sum_n_face),
        #     edges)
        # Here we scatter the global features to the corresponding nodes,
        # giving us tensors of shape [num_nodes, global_feat].
        #  f = jnp.arange(face_edges.shape[0])[:, None]  # [n_faces, 1]
        # e = face_edges  # [n_faces, 3]

        global_attributes = tree.tree_map(lambda g: jnp.repeat(
            g, n_face, axis=0, total_repeat_length=sum_n_face), globals_)

        faces = update_face_preselection_fn(faces, received_edge_attributes, global_attributes)

    sent_face_attributes = tree.tree_map(lambda n: n[face_senders], faces)
    received_face_attributes = tree.tree_map(lambda n: n[face_receivers], faces)

    global_edge_attributes = tree.tree_map(lambda g: jnp.repeat(
        g, n_edge, axis=0, total_repeat_length=sum_n_edge), globals_)

    if update_edge_from_face_selection_fn:
      sent_face_attributes = tree.tree_map(lambda n: n[face_senders], faces)
      received_face_attributes = tree.tree_map(lambda n: n[face_receivers], faces)
      edges = update_edge_from_face_selection_fn(edges, sent_face_attributes, received_face_attributes,global_edge_attributes)

    flip_mask = edges["rn"] == edges["rn_min"]


   # flip_mask=edges["rn"]==edges["rn_min"]


   # if update_edge_flip_selection_fn:
   #     edges = update_edge_flip_selection_fn(edges, sent_attributes, received_attributes,sent_face_attributes,received_face_attributes,
   #                          global_edge_attributes)



    K = n_edge_max // 4

    def extract_flip_edges(flip_mask, K):
        # get candidate flip edges (possibly padded)
        flip_edges = jnp.nonzero(flip_mask, size=K, fill_value=-1)[0]

        # valid ones: >= 0
        valid = flip_edges >= 0

        # JAX-friendly "safe" set: keep invalid as -1 sentinel, not filtered away
        safe_edges = jnp.where(valid, flip_edges, -1)

        return flip_edges, safe_edges, valid

    flip_edges, safe_edges, valid0 = extract_flip_edges(flip_mask, K)

    def resolve_conflicts_randomized(
            writes,  # (K, R) int32, vertex IDs, padded with -1
            valid,  # (K,) bool
            priority,  # (K,) float32
            num_nodes,  # total number of vertices
    ):
        """
        Randomized MIS conflict resolution with PURE VERTEX ownership.

        A flip survives iff it owns ALL vertices in its write-set.
        """

        K, R = writes.shape

        # ------------------------------------------------------------
        # Flatten write-set
        # ------------------------------------------------------------
        resources = writes.reshape(-1)  # (K*R,)
        op_ids = jnp.repeat(jnp.arange(K), R)  # (K*R,)

        # ------------------------------------------------------------
        # Valid entries only
        # ------------------------------------------------------------
        valid_flat = (resources >= 0) & jnp.repeat(valid, R)

        # ------------------------------------------------------------
        # Shift resources so 0 is reserved dummy
        # ------------------------------------------------------------
        resources_shifted = resources + 1  # real nodes: 1..num_nodes

        resources_safe = jnp.where(valid_flat, resources_shifted, 0)
        prios_safe = jnp.where(valid_flat, priority[op_ids], jnp.inf)

        # ------------------------------------------------------------
        # One owner per vertex
        # ------------------------------------------------------------
        owner_prio = jax.ops.segment_min(
            prios_safe,
            resources_safe,
            num_segments=num_nodes + 1
        )

        owns = prios_safe == owner_prio[resources_safe]

        # ------------------------------------------------------------
        # Flip survives only if it owns ALL its vertices
        # ------------------------------------------------------------
        owns_per_op = jax.ops.segment_min(
            owns.astype(jnp.int32),
            op_ids,
            num_segments=K
        )

        return owns_per_op.astype(bool)

    def compute_flip_aggregate(
            senders,
            receivers,
            face_edges,
            face_senders,
            face_receivers,
            flip_edges,
    ):
        """
        Pure computation of flip aggregates.
        No conflict resolution.
        No mutation.
        """

       # faces_edges = faces["edges"]
        old_s = senders
        old_r = receivers
        num_faces = face_edges.shape[0]

        valid = flip_edges >= 0
        e = jnp.where(valid, flip_edges, -1)

        fR = jnp.where(valid, face_senders[e], -1)
        fL = jnp.where(valid, face_receivers[e], -1)
        valid &= (fL >= 0) & (fR >= 0)

        i = jnp.where(valid, old_s[e], -1)
        j = jnp.where(valid, old_r[e], -1)

        def opposing_vertex(f_ids, a, b):
            fe = face_edges[f_ids]
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
        valid &= quad_ok

        def find_edge_with(f_ids, a, b, face_edges, senders, receivers):
            """
            For each face f_ids[t], find the edge in that face
            connecting vertices a[t] and b[t].
            Returns edge id or -1.
            """
            fe = face_edges[f_ids]  # (K, 3)
            s = senders[fe]  # (K, 3)
            r = receivers[fe]  # (K, 3)

            mask = ((s == a[:, None]) & (r == b[:, None])) | \
                   ((s == b[:, None]) & (r == a[:, None]))

            idx = jnp.argmax(mask, axis=1)
            exists = jnp.any(mask, axis=1)

            edge = fe[jnp.arange(fe.shape[0]), idx]
            return jnp.where(exists, edge, -1)

        e_jk = find_edge_with(fL, j, k, face_edges, old_s, old_r)
        e_ki = find_edge_with(fL, k, i, face_edges, old_s, old_r)
        e_il = find_edge_with(fR, i, l, face_edges, old_s, old_r)
        e_lj = find_edge_with(fR, l, j, face_edges, old_s, old_r)

        neigh_ok = (e_jk >= 0) & (e_ki >= 0) & (e_il >= 0) & (e_lj >= 0)
        valid &= neigh_ok

        def m(x): return jnp.where(valid, x, -1)

        aggregate = jnp.stack(
            [
                m(i), m(j), m(k), m(l),
                m(e), m(fL), m(fR),
                m(e_jk), m(e_ki), m(e_il), m(e_lj),
                m(old_s[e]), m(old_r[e]),
            ],
            axis=1,
        ).astype(jnp.int32)

        return aggregate

    def resolve_flip_conflicts(
            aggregate,
            node_neighbors,
            face_senders,
            face_receivers,
            rng_key,
            n_node_max,
    ):
        K = aggregate.shape[0]

        quad = aggregate[:, :4]
        e = aggregate[:, 4]
        valid = e >= 0

        nbr_i = node_neighbors[quad[:, 0]]
        nbr_j = node_neighbors[quad[:, 1]]
        nbr_k = node_neighbors[quad[:, 2]]
        nbr_l = node_neighbors[quad[:, 3]]

        writes = jnp.concatenate(
            [quad, nbr_i, nbr_j, nbr_k, nbr_l],
            axis=1,
        )

        rng_key, subkey = jax.random.split(rng_key)
        priority = jax.random.uniform(subkey, (K,))
        priority = jnp.where(valid, priority, jnp.inf)

        keep = resolve_conflicts_randomized(
            writes,
            valid,
            priority,
            n_node_max,
        )

        mask = keep & valid
        aggregate = jnp.where(mask[:, None], aggregate, -1)

        return aggregate, rng_key

    def compactify_aggregate(aggregate, Kf, valid_col=4):
        """
        aggregate : (K, R) int array with -1 padding
        Kf        : fixed output size
        valid_col: column that contains edge_ids (-1 = invalid)

        returns  : (Kf, R) compactified aggregate, padded with -1
        """
        K, R = aggregate.shape

        # 1. row is valid if it has a non-negative edge id
        valid_mask = aggregate[:, valid_col] >= 0
        valid_idx = jnp.where(valid_mask, size=Kf, fill_value=-1)[0]

        # 2. initialize output
        out = -jnp.ones((Kf, R), dtype=aggregate.dtype)

        # 3. scatter valid rows
        valid_rows = aggregate[valid_idx]
        out = out.at[jnp.arange(Kf)].set(
            jnp.where(valid_idx[:, None] >= 0, valid_rows, out)
        )

        return out

        ################# PARALLEL

    def masked_scatter(base, idx, values, mask):
        safe_idx = jnp.where(mask, idx, -1)
        old = base[safe_idx]

        # broadcast mask to value rank
        m = mask
        while m.ndim < values.ndim:
            m = m[..., None]

        updates = jnp.where(m, values, old)
        return base.at[safe_idx].set(updates)

    def replace_role_masked(face_senders, face_receivers, e, old_f, new_f, mask):
        fs = face_senders
        fr = face_receivers

        fs_e = fs[e]
        fr_e = fr[e]

        new_fs_e = jnp.where(
            mask,
            jnp.where(fs_e == old_f, new_f, fs_e),
            fs_e,
        )
        new_fr_e = jnp.where(
            mask,
            jnp.where(fr_e == old_f, new_f, fr_e),
            fr_e,
        )

        fs = fs.at[e].set(new_fs_e)
        fr = fr.at[e].set(new_fr_e)
        return fs, fr

    def remove_neighbor_masked(neigh, src, dst, mask):
        # neigh: (N_nodes, max_deg)
        row = neigh[src]
        new_row = jnp.where(mask[:, None] & (row == dst[:, None]), -1, row)
        return neigh.at[src].set(new_row)

    def add_neighbor_masked(neigh, src, dst, mask):
        row = neigh[src]
        empty = row == -1
        idx = jnp.argmax(empty, axis=1)
        updates = jnp.where(mask, dst, row[jnp.arange(row.shape[0]), idx])
        return neigh.at[src, idx].set(updates)

    def apply_flip_parallel_full(
            senders,
            receivers,
            face_edges,
            face_senders,
            face_receivers,
            node_neighbors,
            aggregate,
    ):
        (
            i, j, k, l,
            e, fL, fR,
            e_jk, e_ki, e_il, e_lj,
            s_old, r_old,
        ) = aggregate.T

        mask = e >= 0

        # --------------------------------------------------
        # Edge orientation
        # --------------------------------------------------
        senders = masked_scatter(senders, e, k, mask)
        receivers = masked_scatter(receivers, e, l, mask)

        # --------------------------------------------------
        # Face edges
        # --------------------------------------------------
        #  faces_edges = faces["edges"]

        fL_vals = jnp.stack([e, e_lj, e_jk], axis=1)
        fR_vals = jnp.stack([e, e_ki, e_il], axis=1)

        face_edges = masked_scatter(face_edges, fL, fL_vals, mask)
        face_edges = masked_scatter(face_edges, fR, fR_vals, mask)

        #  faces = frozendict({**faces, "edges": faces_edges})

        # --------------------------------------------------
        # Face incidence
        # --------------------------------------------------
        face_senders, face_receivers = replace_role_masked(
            face_senders, face_receivers, e_ki, fL, fR, mask
        )
        face_senders, face_receivers = replace_role_masked(
            face_senders, face_receivers, e_lj, fR, fL, mask
        )

        # --------------------------------------------------
        # Neighbors
        # --------------------------------------------------
        node_neighbors = remove_neighbor_masked(node_neighbors, i, j, mask)
        node_neighbors = remove_neighbor_masked(node_neighbors, j, i, mask)
        node_neighbors = add_neighbor_masked(node_neighbors, k, l, mask)
        node_neighbors = add_neighbor_masked(node_neighbors, l, k, mask)

        return (
            senders,
            receivers,
            face_edges,
            face_senders,
            face_receivers,
            node_neighbors,
        )

    def undo_flip_parallel_full(
            senders,
            receivers,
            face_edges,
            face_senders,
            face_receivers,
            node_neighbors,
            aggregate,
    ):
        (
            i, j, k, l,
            e, fL, fR,
            e_jk, e_ki, e_il, e_lj,
            s_old, r_old,
        ) = aggregate.T

        mask = e >= 0

        # --------------------------------------------------
        # Restore edge orientation
        # --------------------------------------------------
        senders = masked_scatter(senders, e, s_old, mask)
        receivers = masked_scatter(receivers, e, r_old, mask)

        # --------------------------------------------------
        # Restore face edges
        # --------------------------------------------------
        #   faces_edges = faces["edges"]

        fL_vals = jnp.stack([e, e_jk, e_ki], axis=1)
        fR_vals = jnp.stack([e, e_il, e_lj], axis=1)

        face_edges = masked_scatter(face_edges, fL, fL_vals, mask)
        face_edges = masked_scatter(face_edges, fR, fR_vals, mask)

        # faces = frozendict({**faces, "edges": faces_edges})

        # --------------------------------------------------
        # Restore face incidence
        # --------------------------------------------------
        face_senders, face_receivers = replace_role_masked(
            face_senders, face_receivers, e_ki, fR, fL, mask
        )
        face_senders, face_receivers = replace_role_masked(
            face_senders, face_receivers, e_lj, fL, fR, mask
        )

        # --------------------------------------------------
        # Restore neighbors
        # --------------------------------------------------
        node_neighbors = remove_neighbor_masked(node_neighbors, k, l, mask)
        node_neighbors = remove_neighbor_masked(node_neighbors, l, k, mask)
        node_neighbors = add_neighbor_masked(node_neighbors, i, j, mask)
        node_neighbors = add_neighbor_masked(node_neighbors, j, i, mask)

        return (
            senders,
            receivers,
            face_edges,
            face_senders,
            face_receivers,
            node_neighbors,
        )

    def per_flip_energy_sum(node_energy, edge_energy, face_energy, aggregate):
        """
        Per-flip local energy using the aggregate layout (K, 11).

        NOTE: This is mechanically correct but NOT sufficient for
        physically correct Metropolis in most mesh energies.
        """

        # vertices involved in the quad
        quad_nodes = aggregate[:, 0:4]  # (K, 4)

        # edges involved
        quad_edges = aggregate[:, [4, 7, 8, 9, 10]]  # (K, 5)

        # faces involved
        quad_faces = aggregate[:, 5:7]  # (K, 2)

        # ------------------------
        # validity masks
        # ------------------------
        valid_nodes = quad_nodes >= 0
        valid_edges = quad_edges >= 0
        valid_faces = quad_faces >= 0

        # ------------------------
        # energy sums
        # ------------------------
        E_nodes = jnp.where(valid_nodes, node_energy[quad_nodes], 0.0).sum(axis=1)
        E_edges = jnp.where(valid_edges, edge_energy[quad_edges], 0.0).sum(axis=1)
        E_faces = jnp.where(valid_faces, face_energy[quad_faces], 0.0).sum(axis=1)

        return E_nodes + E_edges + E_faces, node_energy[quad_nodes], edge_energy[quad_edges], face_energy[
            quad_faces]  # (K,)

    def scatter_quad_energies_overwrite(
            nodes,
            edges,
            faces,
            aggregate,
            quad_node_energy,
            quad_edge_energy,
            quad_face_energy,
    ):
        """
        Overwrite per-quad energies back into node/edge/face arrays.

        Semantics:
        - If an index is valid (>= 0): overwrite with quad energy
        - If invalid (-1): keep the old stored energy
        """

        # ------------------------
        # unpack indices
        # ------------------------
        quad_nodes = aggregate[:, 0:4]  # (K, 4)
        quad_edges = aggregate[:, [4, 7, 8, 9, 10]]  # (K, 5)
        quad_faces = aggregate[:, 5:7]  # (K, 2)

        # ------------------------
        # masks
        # ------------------------
        node_mask = quad_nodes >= 0
        edge_mask = quad_edges >= 0
        face_mask = quad_faces >= 0

        # ------------------------
        # flatten
        # ------------------------
        node_idx = quad_nodes.reshape(-1)
        edge_idx = quad_edges.reshape(-1)
        face_idx = quad_faces.reshape(-1)

        node_val = quad_node_energy.reshape(-1)
        edge_val = quad_edge_energy.reshape(-1)
        face_val = quad_face_energy.reshape(-1)

        node_mask = node_mask.reshape(-1)
        edge_mask = edge_mask.reshape(-1)
        face_mask = face_mask.reshape(-1)

        # ------------------------
        # overwrite but preserve old values if invalid
        # ------------------------
        node_energy_old = nodes["energy"]
        edge_energy_old = edges["energy"]
        face_energy_old = faces["energy"]

        new_node_energy = node_energy_old.at[node_idx].set(
            jnp.where(node_mask, node_val, node_energy_old[node_idx])
        )

        new_edge_energy = edge_energy_old.at[edge_idx].set(
            jnp.where(edge_mask, edge_val, edge_energy_old[edge_idx])
        )

        new_face_energy = face_energy_old.at[face_idx].set(
            jnp.where(face_mask, face_val, face_energy_old[face_idx])
        )

        # ------------------------
        # return updated pytrees
        # ------------------------
        nodes = {**nodes, "energy": new_node_energy}
        edges = {**edges, "energy": new_edge_energy}
        faces = {**faces, "energy": new_face_energy}

        return nodes, edges, faces

    def mask_aggregate_by_flip_edges(aggregate, flip_edges):
        """
        Disable aggregate rows where flip_edges[t] < 0.

        Args:
            aggregate: (K, A) int32
            flip_edges: (K,) int32, -1 means rejected

        Returns:
            aggregate_masked: (K, A) int32
        """
        keep = flip_edges >= 0
        return jnp.where(keep[:, None], aggregate, -1)



    def deactivate_aggregate_rows(aggregate, flip_edges):
        """
        Set aggregate[valid_edge_id, :] = -1 for valid_edge_id in flip_edges.
        Rows corresponding to flip_edges == -1 are untouched.
        """

        # mask of valid indices
        valid_mask = flip_edges >= 0  # (M,)

        # replace invalid indices with 0 to keep indexing safe
        safe_idx = jnp.where(valid_mask, flip_edges, 0)

        # value to write
        new_rows = -jnp.ones((flip_edges.shape[0], aggregate.shape[1]),
                             dtype=aggregate.dtype)

        # overwrite semantics with preservation
        aggregate = aggregate.at[safe_idx, :].set(
            jnp.where(valid_mask[:, None], new_rows, aggregate[safe_idx, :])
        )

        return aggregate

    aggregate = compute_flip_aggregate(
        senders,
        receivers,
        face_edges,
        face_senders,
        face_receivers,
        flip_edges,
    )


    count=0

  #  init_state=(aggregate,count,senders,receivers,face_senders,face_receivers,face_edges,node_neighbors,nodes,edges,faces,rng_key)

    def cond_fn(state,max_count=1):
        (
            aggregate,
            count,
            senders,
            receivers,
            face_senders,
            face_receivers,
            face_edges,
            node_neighbors,
            nodes,
            edges,
            faces,
            rng_key,
        ) = state
        return jnp.any(aggregate[:, 4] != -1) & (count<max_count)


    #def body_fn(state,Kcp=80):

    #(aggregate, count, senders, receivers, face_senders, face_receivers, face_edges, node_neighbors, nodes, edges, faces,rng_key)=state




    aggregate, rng_key = resolve_flip_conflicts(
        aggregate,
        node_neighbors,
        face_senders,
        face_receivers,
        rng_key,
        n_node_max,
    )

    flip_edges = aggregate[:, 4]

   # aggregate =deactivate_aggregate_rows(aggregate,flip_edges)




    Kf = n_edge_max // 80
    aggregate=compactify_aggregate(aggregate,Kf)


    senders, receivers, face_edges, face_senders, face_receivers, node_neighbors = (
       apply_flip_parallel_full(
           senders,
           receivers,
           face_edges,
           face_senders,
           face_receivers,
           node_neighbors,
           aggregate,

       )
    )

    energy_before,E_nodes,E_edges,E_faces=per_flip_energy_sum(nodes["energy"],edges['energy'],faces['energy'],aggregate)


    ### RECOMPUTE
    node_mask_local,edge_mask_local,face_mask_local=build_local_mask_fn(aggregate,n_node_max,n_edge_max,n_face_max)


    if not tree.tree_all(
            tree.tree_map(lambda n: n.shape[0] == sum_n_node, nodes)):
        raise ValueError(
            'All node arrays in nest must contain the same number of nodes.')

    if not tree.tree_all(
            tree.tree_map(lambda n: n.shape[0] == sum_n_face, faces)):
        raise ValueError(
            'All face arrays in nest must contain the same number of faces.')

    sent_attributes = tree.tree_map(lambda n: n[senders], nodes)
    received_attributes = tree.tree_map(lambda n: n[receivers], nodes)

    # Here we scatter the global features to the corresponding edges,
    # giving us tensors of shape [num_edges, global_feat].
    global_edge_attributes = tree.tree_map(lambda g: jnp.repeat(
        g, n_edge, axis=0, total_repeat_length=sum_n_edge), globals_)


    edges = update_edge_local_fn(edges,  sent_attributes, received_attributes,
                           edge_mask_local,global_edge_attributes)


    # sent_face_attributes = tree.tree_map(
    #     lambda e: aggregate_edges_for_faces_fn(e, face_senders, sum_n_face), edges)

    received_edge_attributes = tree.tree_map(lambda n: n[face_edges], edges)
    #     lambda e: aggregate_edges_for_faces_fn(e, face_receivers, sum_n_face),
    #     edges)
    # Here we scatter the global features to the corresponding nodes,
    # giving us tensors of shape [num_nodes, global_feat].
    f = jnp.arange(face_edges.shape[0])[:, None]  # [n_faces, 1]
    # e = face_edges  # [n_faces, 3]

    sign = jnp.where(face_receivers[face_edges] == f, 1.0, -1.0)
    global_attributes = tree.tree_map(lambda g: jnp.repeat(
        g, n_face, axis=0, total_repeat_length=sum_n_face), globals_)
    faces = update_face_local_fn(faces, received_edge_attributes, sign, face_mask_local, global_attributes)


    sent_face_attributes = tree.tree_map(lambda n: n[face_senders], faces)
    received_face_attributes = tree.tree_map(lambda n: n[face_receivers], faces)
    edges = update_edge_from_face_local_fn(edges, sent_face_attributes, received_face_attributes,
                                     edge_mask_local,global_edge_attributes)



    sent_attributes = tree.tree_map(
        lambda e: aggregate_edges_for_nodes_fn(e, senders, sum_n_node), edges)
    received_attributes = tree.tree_map(
        lambda e: aggregate_edges_for_nodes_fn(e, receivers, sum_n_node),
        edges)
    # Here we scatter the global features to the corresponding nodes,
    # giving us tensors of shape [num_nodes, global_feat].
    global_attributes = tree.tree_map(lambda g: jnp.repeat(
        g, n_node, axis=0, total_repeat_length=sum_n_node), globals_)
    nodes = update_node_local_fn(nodes, sent_attributes,
                           received_attributes, node_mask_local,global_attributes)



    energy_after,_,_,_ = per_flip_energy_sum(nodes["energy"], edges['energy'], faces['energy'], aggregate)



    flip_edges,accepted_fraction,rng_key=metropolis_fn(aggregate[:, 4],
        energy_before,
        energy_after,
        10,
        rng_key,
    )

    aggregate = mask_aggregate_by_flip_edges(aggregate, flip_edges)
    flip_edges=aggregate[:, 4]



    senders, receivers, face_edges, face_senders, face_receivers, node_neighbors = (
    undo_flip_parallel_full(
        senders,
        receivers,
        face_edges,
        face_senders,
        face_receivers,
        node_neighbors,
        aggregate,

    )
    )

    scatter_quad_energies_overwrite(
        nodes,
        edges,
        faces,
        aggregate,
        E_nodes,
        E_edges,
        E_faces,
    )

    count += 1

      #  return (aggregate,count,senders,receivers,face_senders,face_receivers,face_edges,node_neighbors,nodes,edges,faces,rng_key)

    #(aggregate,count,senders,receivers,face_senders,face_receivers,face_edges,node_neighbors,nodes,edges,faces,rng_key) = jax.lax.while_loop(
    #    cond_fn,
    #    body_fn,
    #    init_state,
    #)
   # (aggregate, count, senders, receivers, face_senders, face_receivers, face_edges, node_neighbors, nodes, edges, faces,
   # rng_key)=body_fn(init_state)

   # globals_=frozendict({**globals_,"flips_per":jnp.sum(flip_mask)/n_edge_max,"flip_edge_per":jnp.sum(jnp.int32(flip_edges>=0))/n_edge_max,"flips_acc":accepted_fraction,"E_b":jnp.sum(energy_before),"E_a":jnp.sum(energy_after)})




    # pylint: enable=g-long-lambda
    return gn_graph.MeshGraphsTuple(
        nodes=nodes,
        edges=edges,
        faces=faces,
        globals=globals_,
        receivers=receivers,
        senders=senders,
        face_receivers=face_receivers,
        face_senders=face_senders,
        face_edges=face_edges,
        node_neighbors=node_neighbors,
        node_mask=node_mask,
        edge_mask=edge_mask,
        face_mask=face_mask,
        n_node=n_node,
        n_edge=n_edge,
        n_face=n_face,
        n_node_max=n_node_max,
        n_edge_max=n_edge_max,
        n_face_max=n_face_max,
        rng_key=rng_key,)

  return lambda n: _ApplyMeshGraphNetFlip(n,n_node_max,n_edge_max,n_face_max)




def GraphNetwork(
    update_edge_fn: Optional[GNUpdateEdgeFn],
    update_node_fn: Optional[GNUpdateNodeFn],
    update_global_fn: Optional[GNUpdateGlobalFn] = None,
    aggregate_edges_for_nodes_fn: AggregateEdgesToNodesFn = utils.segment_sum,
    aggregate_nodes_for_globals_fn: AggregateNodesToGlobalsFn = utils
    .segment_sum,
    aggregate_edges_for_globals_fn: AggregateEdgesToGlobalsFn = utils
    .segment_sum,
    attention_logit_fn: Optional[AttentionLogitFn] = None,
    attention_normalize_fn: Optional[AttentionNormalizeFn] = utils
    .segment_softmax,
    attention_reduce_fn: Optional[AttentionReduceFn] = None):
  """Returns a method that applies a configured GraphNetwork.

  This implementation follows Algorithm 1 in https://arxiv.org/abs/1806.01261

  There is one difference. For the nodes update the class aggregates over the
  sender edges and receiver edges separately. This is a bit more general
  than the algorithm described in the paper. The original behaviour can be
  recovered by using only the receiver edge aggregations for the update.

  In addition this implementation supports softmax attention over incoming
  edge features.

  Example usage::

    gn = GraphNetwork(update_edge_function,
    update_node_function, **kwargs)
    # Conduct multiple rounds of message passing with the same parameters:
    for _ in range(num_message_passing_steps):
      graph = gn(graph)

  Args:
    update_edge_fn: function used to update the edges or None to deactivate edge
      updates.
    update_node_fn: function used to update the nodes or None to deactivate node
      updates.
    update_global_fn: function used to update the globals or None to deactivate
      globals updates.
    aggregate_edges_for_nodes_fn: function used to aggregate messages to each
      node.
    aggregate_nodes_for_globals_fn: function used to aggregate the nodes for the
      globals.
    aggregate_edges_for_globals_fn: function used to aggregate the edges for the
      globals.
    attention_logit_fn: function used to calculate the attention weights or
      None to deactivate attention mechanism.
    attention_normalize_fn: function used to normalize raw attention logits or
      None if attention mechanism is not active.
    attention_reduce_fn: function used to apply weights to the edge features or
      None if attention mechanism is not active.

  Returns:
    A method that applies the configured GraphNetwork.
  """
  not_both_supplied = lambda x, y: (x != y) and ((x is None) or (y is None))
  if not_both_supplied(attention_reduce_fn, attention_logit_fn):
    raise ValueError(('attention_logit_fn and attention_reduce_fn must both be'
                      ' supplied.'))

  def _ApplyGraphNet(graph):
    """Applies a configured GraphNetwork to a graph.

    This implementation follows Algorithm 1 in https://arxiv.org/abs/1806.01261

    There is one difference. For the nodes update the class aggregates over the
    sender edges and receiver edges separately. This is a bit more general
    the algorithm described in the paper. The original behaviour can be
    recovered by using only the receiver edge aggregations for the update.

    In addition this implementation supports softmax attention over incoming
    edge features.

    Many popular Graph Neural Networks can be implemented as special cases of
    GraphNets, for more information please see the paper.

    Args:
      graph: a `GraphsTuple` containing the graph.

    Returns:
      Updated `GraphsTuple`.
    """
    # pylint: disable=g-long-lambda
    nodes, edges, receivers, senders, globals_, n_node, n_edge = graph
    # Equivalent to jnp.sum(n_node), but jittable
    sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
    sum_n_edge = senders.shape[0]
    if not tree.tree_all(
        tree.tree_map(lambda n: n.shape[0] == sum_n_node, nodes)):
      raise ValueError(
          'All node arrays in nest must contain the same number of nodes.')

    sent_attributes = tree.tree_map(lambda n: n[senders], nodes)
    received_attributes = tree.tree_map(lambda n: n[receivers], nodes)
    # Here we scatter the global features to the corresponding edges,
    # giving us tensors of shape [num_edges, global_feat].
    global_edge_attributes = tree.tree_map(lambda g: jnp.repeat(
        g, n_edge, axis=0, total_repeat_length=sum_n_edge), globals_)

    if update_edge_fn:
      edges = update_edge_fn(edges, sent_attributes, received_attributes,
                             global_edge_attributes)

    if attention_logit_fn:
      logits = attention_logit_fn(edges, sent_attributes, received_attributes,
                                  global_edge_attributes)
      tree_calculate_weights = functools.partial(
          attention_normalize_fn,
          segment_ids=receivers,
          num_segments=sum_n_node)
      weights = tree.tree_map(tree_calculate_weights, logits)
      edges = attention_reduce_fn(edges, weights)

    if update_node_fn:
      sent_attributes = tree.tree_map(
          lambda e: aggregate_edges_for_nodes_fn(e, senders, sum_n_node), edges)
      received_attributes = tree.tree_map(
          lambda e: aggregate_edges_for_nodes_fn(e, receivers, sum_n_node),
          edges)
      # Here we scatter the global features to the corresponding nodes,
      # giving us tensors of shape [num_nodes, global_feat].
      global_attributes = tree.tree_map(lambda g: jnp.repeat(
          g, n_node, axis=0, total_repeat_length=sum_n_node), globals_)
      nodes = update_node_fn(nodes, sent_attributes,
                             received_attributes, global_attributes)

    if update_global_fn:
      n_graph = n_node.shape[0]
      graph_idx = jnp.arange(n_graph)
      # To aggregate nodes and edges from each graph to global features,
      # we first construct tensors that map the node to the corresponding graph.
      # For example, if you have `n_node=[1,2]`, we construct the tensor
      # [0, 1, 1]. We then do the same for edges.
      node_gr_idx = jnp.repeat(
          graph_idx, n_node, axis=0, total_repeat_length=sum_n_node)
      edge_gr_idx = jnp.repeat(
          graph_idx, n_edge, axis=0, total_repeat_length=sum_n_edge)
      # We use the aggregation function to pool the nodes/edges per graph.
      node_attributes = tree.tree_map(
          lambda n: aggregate_nodes_for_globals_fn(n, node_gr_idx, n_graph),
          nodes)
      edge_attribtutes = tree.tree_map(
          lambda e: aggregate_edges_for_globals_fn(e, edge_gr_idx, n_graph),
          edges)
      # These pooled nodes are the inputs to the global update fn.
      globals_ = update_global_fn(node_attributes, edge_attribtutes, globals_)
    # pylint: enable=g-long-lambda
    return gn_graph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        receivers=receivers,
        senders=senders,
        globals=globals_,
        n_node=n_node,
        n_edge=n_edge)

  return _ApplyGraphNet



InteractionUpdateNodeFn = Callable[
    [NodeFeatures,
     Mapping[str, SenderFeatures],
     Mapping[str, ReceiverFeatures]],
    NodeFeatures]
InteractionUpdateNodeFnNoSentEdges = Callable[
    [NodeFeatures,
     Mapping[str, ReceiverFeatures]],
    NodeFeatures]

InteractionUpdateEdgeFn = Callable[
    [EdgeFeatures, SenderFeatures, ReceiverFeatures], EdgeFeatures]


def InteractionNetwork(
    update_edge_fn: InteractionUpdateEdgeFn,
    update_node_fn: Union[InteractionUpdateNodeFn,
                          InteractionUpdateNodeFnNoSentEdges],
    aggregate_edges_for_nodes_fn: AggregateEdgesToNodesFn = utils.segment_sum,
    include_sent_messages_in_node_update: bool = False):
  """Returns a method that applies a configured InteractionNetwork.

  An interaction network computes interactions on the edges based on the
  previous edges features, and on the features of the nodes sending into those
  edges. It then updates the nodes based on the incoming updated edges.
  See https://arxiv.org/abs/1612.00222 for more details.

  This implementation adds an option not in https://arxiv.org/abs/1612.00222,
  which is to include edge features for which a node is a sender in the
  arguments to the node update function.

  Args:
    update_edge_fn: a function mapping a single edge update inputs to a single
      edge feature.
    update_node_fn: a function mapping a single node update input to a single
      node feature.
    aggregate_edges_for_nodes_fn: function used to aggregate messages to each
      node.
    include_sent_messages_in_node_update: pass edge features for which a node is
      a sender to the node update function.
  """
  # An InteractionNetwork is a GraphNetwork without globals features,
  # so we implement the InteractionNetwork as a configured GraphNetwork.

  # An InteractionNetwork edge function does not have global feature inputs,
  # so we filter the passed global argument in the GraphNetwork.
  wrapped_update_edge_fn = lambda e, s, r, g: update_edge_fn(e, s, r)

  # Similarly, we wrap the update_node_fn to ensure only the expected
  # arguments are passed to the Interaction net.
  if include_sent_messages_in_node_update:
    wrapped_update_node_fn = lambda n, s, r, g: update_node_fn(n, s, r)  # pytype: disable=wrong-arg-count
  else:
    wrapped_update_node_fn = lambda n, s, r, g: update_node_fn(n, r)  # pytype: disable=wrong-arg-count
  return GraphNetwork(
      update_edge_fn=wrapped_update_edge_fn,
      update_node_fn=wrapped_update_node_fn,
      aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn)


# Signature:
# edge features -> embedded edge features
EmbedEdgeFn = Callable[[EdgeFeatures], EdgeFeatures]

# Signature:
# node features -> embedded node features
EmbedNodeFn = Callable[[NodeFeatures], NodeFeatures]

# Signature:
# globals features -> embedded globals features
EmbedGlobalFn = Callable[[Globals], Globals]


def GraphMapFeatures(embed_edge_fn: Optional[EmbedEdgeFn] = None,
                     embed_node_fn: Optional[EmbedNodeFn] = None,
                     embed_global_fn: Optional[EmbedGlobalFn] = None):
  """Returns function which embeds the components of a graph independently.

  Args:
    embed_edge_fn: function used to embed the edges.
    embed_node_fn: function used to embed the nodes.
    embed_global_fn: function used to embed the globals.
  """
  identity = lambda x: x
  embed_edges_fn = embed_edge_fn if embed_edge_fn else identity
  embed_nodes_fn = embed_node_fn if embed_node_fn else identity
  embed_global_fn = embed_global_fn if embed_global_fn else identity

  def Embed(graphs_tuple):
    return graphs_tuple._replace(
        nodes=embed_nodes_fn(graphs_tuple.nodes),
        edges=embed_edges_fn(graphs_tuple.edges),
        globals=embed_global_fn(graphs_tuple.globals))

  return Embed


def RelationNetwork(
    update_edge_fn: Callable[[SenderFeatures, ReceiverFeatures], EdgeFeatures],
    update_global_fn: Callable[[EdgeFeatures], NodeFeatures],
    aggregate_edges_for_globals_fn:
        AggregateEdgesToGlobalsFn = utils.segment_sum):
  """Returns a method that applies a Relation Network.

  See https://arxiv.org/abs/1706.01427 for more details.

  This implementation has one more argument, `aggregate_edges_for_globals_fn`,
  which changes how edge features are aggregated. The paper uses the default -
  `utils.segment_sum`.

  Args:
    update_edge_fn: function used to update the edges.
    update_global_fn: function used to update the globals.
    aggregate_edges_for_globals_fn: function used to aggregate the edges for the
      globals.
  """
  return GraphNetwork(
      update_edge_fn=lambda e, s, r, g: update_edge_fn(s, r),
      update_node_fn=None,
      update_global_fn=lambda n, e, g: update_global_fn(e),
      attention_logit_fn=None,
      aggregate_edges_for_globals_fn=aggregate_edges_for_globals_fn)


def DeepSets(
    update_node_fn: Callable[[NodeFeatures, Globals], NodeFeatures],
    update_global_fn: Callable[[NodeFeatures], Globals],
    aggregate_nodes_for_globals_fn:
        AggregateNodesToGlobalsFn = utils.segment_sum):
  """Returns a method that applies a DeepSets layer.

  Implementation for the model described in https://arxiv.org/abs/1703.06114
  (M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Poczos, R. Salakhutdinov, A. Smola).
  See also PointNet (https://arxiv.org/abs/1612.00593, C. Qi, H. Su, K. Mo,
  L. J. Guibas) for a related model.

  This module operates on sets, which can be thought of as graphs without
  edges. The nodes features are first updated based on their value and the
  globals features, and new globals features are then computed based on the
  updated nodes features.

  Args:
    update_node_fn: function used to update the nodes.
    update_global_fn: function used to update the globals.
    aggregate_nodes_for_globals_fn: function used to aggregate the nodes for the
      globals.
  """
  # DeepSets can be implemented with a GraphNetwork, with just a node
  # update function that takes nodes and globals, and a global update
  # function based on the updated node features.
  return GraphNetwork(
      update_edge_fn=None,
      update_node_fn=lambda n, s, r, g: update_node_fn(n, g),
      update_global_fn=lambda n, e, g: update_global_fn(n),
      aggregate_nodes_for_globals_fn=aggregate_nodes_for_globals_fn)


def GraphNetGAT(
    update_edge_fn: GNUpdateEdgeFn,
    update_node_fn: GNUpdateNodeFn,
    attention_logit_fn: AttentionLogitFn,
    attention_reduce_fn: AttentionReduceFn,
    update_global_fn: Optional[GNUpdateGlobalFn] = None,
    aggregate_edges_for_nodes_fn: AggregateEdgesToNodesFn = utils.segment_sum,
    aggregate_nodes_for_globals_fn: AggregateNodesToGlobalsFn = utils.
    segment_sum,
    aggregate_edges_for_globals_fn: AggregateEdgesToGlobalsFn = utils.
    segment_sum
    ):
  """Returns a method that applies a GraphNet with attention on edge features.

  Args:
    update_edge_fn: function used to update the edges.
    update_node_fn: function used to update the nodes.
    attention_logit_fn: function used to calculate the attention weights.
    attention_reduce_fn: function used to apply attention weights to the edge
      features.
    update_global_fn: function used to update the globals or None to deactivate
      globals updates.
    aggregate_edges_for_nodes_fn: function used to aggregate attention-weighted
      messages to each node.
    aggregate_nodes_for_globals_fn: function used to aggregate the nodes for the
      globals.
    aggregate_edges_for_globals_fn: function used to aggregate
      attention-weighted edges for the globals.

  Returns:
    A function that applies a GraphNet Graph Attention layer.
  """
  if (attention_logit_fn is None) or (attention_reduce_fn is None):
    raise ValueError(('`None` value not supported for `attention_logit_fn` or '
                      '`attention_reduce_fn` in a Graph Attention network.'))
  return GraphNetwork(
      update_edge_fn=update_edge_fn,
      update_node_fn=update_node_fn,
      update_global_fn=update_global_fn,
      attention_logit_fn=attention_logit_fn,
      attention_reduce_fn=attention_reduce_fn,
      aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn,
      aggregate_nodes_for_globals_fn=aggregate_nodes_for_globals_fn,
      aggregate_edges_for_globals_fn=aggregate_edges_for_globals_fn)


GATAttentionQueryFn = Callable[[NodeFeatures], NodeFeatures]
GATAttentionLogitFn = Callable[
    [SenderFeatures, ReceiverFeatures, EdgeFeatures], EdgeFeatures]
GATNodeUpdateFn = Callable[[NodeFeatures], NodeFeatures]


def GAT(attention_query_fn: GATAttentionQueryFn,
        attention_logit_fn: GATAttentionLogitFn,
        node_update_fn: Optional[GATNodeUpdateFn] = None):
  """Returns a method that applies a Graph Attention Network layer.

  Graph Attention message passing as described in
  https://arxiv.org/abs/1710.10903. This model expects node features as a
  jnp.array, may use edge features for computing attention weights, and
  ignore global features. It does not support nests.

  NOTE: this implementation assumes that the input graph has self edges. To
  recover the behavior of the referenced paper, please add self edges.

  Args:
    attention_query_fn: function that generates attention queries
      from sender node features.
    attention_logit_fn: function that converts attention queries into logits for
      softmax attention.
    node_update_fn: function that updates the aggregated messages. If None,
      will apply leaky relu and concatenate (if using multi-head attention).

  Returns:
    A function that applies a Graph Attention layer.
  """
  # pylint: disable=g-long-lambda
  if node_update_fn is None:
    # By default, apply the leaky relu and then concatenate the heads on the
    # feature axis.
    node_update_fn = lambda x: jnp.reshape(
        jax.nn.leaky_relu(x), (x.shape[0], -1))
  def _ApplyGAT(graph):
    """Applies a Graph Attention layer."""
    nodes, edges, receivers, senders, _, _, _ = graph
    # Equivalent to the sum of n_node, but statically known.
    try:
      sum_n_node = nodes.shape[0]
    except IndexError:
      raise IndexError('GAT requires node features')  # pylint: disable=raise-missing-from

    # First pass nodes through the node updater.
    nodes = attention_query_fn(nodes)
    # pylint: disable=g-long-lambda
    # We compute the softmax logits using a function that takes the
    # embedded sender and receiver attributes.
    sent_attributes = nodes[senders]
    received_attributes = nodes[receivers]
    softmax_logits = attention_logit_fn(
        sent_attributes, received_attributes, edges)

    # Compute the softmax weights on the entire tree.
    weights = utils.segment_softmax(softmax_logits, segment_ids=receivers,
                                    num_segments=sum_n_node)
    # Apply weights
    messages = sent_attributes * weights
    # Aggregate messages to nodes.
    nodes = utils.segment_sum(messages, receivers, num_segments=sum_n_node)

    # Apply an update function to the aggregated messages.
    nodes = node_update_fn(nodes)
    return graph._replace(nodes=nodes)
  # pylint: enable=g-long-lambda
  return _ApplyGAT


def GraphConvolution(
    update_node_fn: Callable[[NodeFeatures], NodeFeatures],
    aggregate_nodes_fn: AggregateEdgesToNodesFn = utils.segment_sum,
    add_self_edges: bool = False,
    symmetric_normalization: bool = True):
  """Returns a method that applies a Graph Convolution layer.

  Graph Convolutional layer as in https://arxiv.org/abs/1609.02907,

  NOTE: This implementation does not add an activation after aggregation.
  If you are stacking layers, you may want to add an activation between
  each layer.

  Args:
    update_node_fn: function used to update the nodes. In the paper a single
      layer MLP is used.
    aggregate_nodes_fn: function used to aggregates the sender nodes.
    add_self_edges: whether to add self edges to nodes in the graph as in the
      paper definition of GCN. Defaults to False.
    symmetric_normalization: whether to use symmetric normalization. Defaults
      to True. Note that to replicate the fomula of the linked paper, the
      adjacency matrix must be symmetric. If the adjacency matrix is not
      symmetric the data is prenormalised by the sender degree matrix and post
      normalised by the receiver degree matrix.

  Returns:
    A method that applies a Graph Convolution layer.
  """
  def _ApplyGCN(graph):
    """Applies a Graph Convolution layer."""
    nodes, _, receivers, senders, _, _, _ = graph

    # First pass nodes through the node updater.
    nodes = update_node_fn(nodes)
    # Equivalent to jnp.sum(n_node), but jittable
    total_num_nodes = tree.tree_leaves(nodes)[0].shape[0]
    if add_self_edges:
      # We add self edges to the senders and receivers so that each node
      # includes itself in aggregation.
      # In principle, a `GraphsTuple` should partition by n_edge, but in
      # this case it is not required since a GCN is agnostic to whether
      # the `GraphsTuple` is a batch of graphs or a single large graph.
      conv_receivers = jnp.concatenate((receivers, jnp.arange(total_num_nodes)),
                                       axis=0)
      conv_senders = jnp.concatenate((senders, jnp.arange(total_num_nodes)),
                                     axis=0)
    else:
      conv_senders = senders
      conv_receivers = receivers

    # pylint: disable=g-long-lambda
    if symmetric_normalization:
      # Calculate the normalization values.
      count_edges = lambda x: utils.segment_sum(
          jnp.ones_like(conv_senders), x, total_num_nodes)
      sender_degree = count_edges(conv_senders)
      receiver_degree = count_edges(conv_receivers)

      # Pre normalize by sqrt sender degree.
      # Avoid dividing by 0 by taking maximum of (degree, 1).
      nodes = tree.tree_map(
          lambda x: x * jax.lax.rsqrt(jnp.maximum(sender_degree, 1.0))[:, None],
          nodes,
      )
      # Aggregate the pre normalized nodes.
      nodes = tree.tree_map(
          lambda x: aggregate_nodes_fn(x[conv_senders], conv_receivers,
                                       total_num_nodes), nodes)
      # Post normalize by sqrt receiver degree.
      # Avoid dividing by 0 by taking maximum of (degree, 1).
      nodes = tree.tree_map(
          lambda x:
          (x * jax.lax.rsqrt(jnp.maximum(receiver_degree, 1.0))[:, None]),
          nodes,
      )
    else:
      nodes = tree.tree_map(
          lambda x: aggregate_nodes_fn(x[conv_senders], conv_receivers,
                                       total_num_nodes), nodes)
    # pylint: enable=g-long-lambda
    return graph._replace(nodes=nodes)

  return _ApplyGCN
