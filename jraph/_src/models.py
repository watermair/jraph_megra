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
    nodes, edges, faces, receivers, senders,face_receivers,face_senders,face_vertices, globals_, n_node, n_edge, n_face,rng_key,n_node_max,n_edge_max,n_face_max= graph
    # Equivalent to jnp.sum(n_node), but jittable
    face_edges=faces["edges"]
    face_vertices=faces["vertices"]
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
                             global_edge_attributes)

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
        faces = update_face_fn(faces,received_edge_attributes,sign,global_attributes)

    if update_edge_from_face_fn:
      sent_face_attributes = tree.tree_map(lambda n: n[face_senders], faces)
      received_face_attributes = tree.tree_map(lambda n: n[face_receivers], faces)
      edges = update_edge_from_face_fn(edges, sent_face_attributes, received_face_attributes,
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
        receivers=receivers,
        senders=senders,
        face_receivers=face_receivers,
        face_senders=face_senders,
        face_vertices=face_vertices,
        globals=globals_,
        n_node=n_node,
        n_edge=n_edge,
        n_face=n_face,
    rng_key=rng_key,
    n_node_max=n_node_max,
    n_edge_max=n_edge_max,
    n_face_max=n_face_max)

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
    nodes, edges, faces, receivers, senders,face_receivers,face_senders,face_vertices, globals_, n_node, n_edge, n_face,rng_key,_,_,_= graph

    face_edges = faces["edges"]
    node_neighbors=nodes["neighbors"]

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
            faces,
            face_senders,
            face_receivers,
            flip_edges,
    ):
        """
        Pure computation of flip aggregates.
        No conflict resolution.
        No mutation.
        """

        faces_edges = faces["edges"]
        old_s = senders
        old_r = receivers
        num_faces = faces_edges.shape[0]

        valid = flip_edges >= 0
        e = jnp.where(valid, flip_edges, -1)

        fR = jnp.where(valid, face_senders[e], -1)
        fL = jnp.where(valid, face_receivers[e], -1)
        valid &= (fL >= 0) & (fR >= 0)

        i = jnp.where(valid, old_s[e], -1)
        j = jnp.where(valid, old_r[e], -1)

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
        valid &= quad_ok

        def find_edge_with(f_ids, a, b, faces_edges, senders, receivers):
            """
            For each face f_ids[t], find the edge in that face
            connecting vertices a[t] and b[t].
            Returns edge id or -1.
            """
            fe = faces_edges[f_ids]  # (K, 3)
            s = senders[fe]  # (K, 3)
            r = receivers[fe]  # (K, 3)

            mask = ((s == a[:, None]) & (r == b[:, None])) | \
                   ((s == b[:, None]) & (r == a[:, None]))

            idx = jnp.argmax(mask, axis=1)
            exists = jnp.any(mask, axis=1)

            edge = fe[jnp.arange(fe.shape[0]), idx]
            return jnp.where(exists, edge, -1)

        e_jk = find_edge_with(fL, j, k, faces_edges, old_s, old_r)
        e_ki = find_edge_with(fL, k, i, faces_edges, old_s, old_r)
        e_il = find_edge_with(fR, i, l, faces_edges, old_s, old_r)
        e_lj = find_edge_with(fR, l, j, faces_edges, old_s, old_r)

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

    def remove_face_from_edge(fs, fr, e, face):
        fs = fs.at[e].set(jnp.where(fs[e] == face, -1, fs[e]))
        fr = fr.at[e].set(jnp.where(fr[e] == face, -1, fr[e]))
        return fs, fr

    def add_face_to_edge(fs, fr, e, face):
        def add_to_receiver():
            return fs, fr.at[e].set(face)

        def add_to_sender():
            return fs.at[e].set(face), fr

        return jax.lax.cond(
            fr[e] == -1,
            lambda _: add_to_receiver(),
            lambda _: add_to_sender(),
            operand=None,
        )

    def remove_neighbor(neigh, a, b):
        row = neigh[a]
        row = jnp.where(row == b, -1, row)
        return neigh.at[a].set(row)

    def add_neighbor(neigh, a, b):
        row = neigh[a]
        idx = jnp.argmax(row == -1)
        row = row.at[idx].set(b)
        return neigh.at[a].set(row)

    def replace_role(fs, fr, edge_ids, old_face, new_face):
        """
        For each edge in edge_ids:
          if fs[e] == old_face -> replace with new_face
          if fr[e] == old_face -> replace with new_face
        """
        valid_edge = edge_ids >= 0

        fs_edge = fs[edge_ids]
        fr_edge = fr[edge_ids]

        fs_edge = jnp.where(valid_edge & (fs_edge == old_face), new_face, fs_edge)
        fr_edge = jnp.where(valid_edge & (fr_edge == old_face), new_face, fr_edge)

        fs = fs.at[edge_ids].set(fs_edge)
        fr = fr.at[edge_ids].set(fr_edge)

        return fs, fr

    def apply_flip_serial_full(
            senders,
            receivers,
            faces,
            face_senders,
            face_receivers,
            node_neighbors,
            agg_row,
    ):
        (
            i, j, k, l,
            e, fL, fR,
            e_jk, e_ki, e_il, e_lj,
            s_old, r_old,
        ) = agg_row

        def no_op():
            return senders, receivers, faces, face_senders, face_receivers, node_neighbors

        def do_flip():
            s = senders
            r = receivers
            fs = face_senders
            fr = face_receivers
            neigh = node_neighbors

            # --------------------------------------------------
            # Edge orientation
            # --------------------------------------------------
            s = s.at[e].set(k)
            r = r.at[e].set(l)

            # --------------------------------------------------
            # Face edges
            # --------------------------------------------------
            faces_edges = faces["edges"]
            faces_edges = (
                faces_edges
                .at[fL].set(jnp.array([e, e_lj, e_jk], dtype=jnp.int32))
                .at[fR].set(jnp.array([e, e_ki, e_il], dtype=jnp.int32))
            )
            new_faces = frozendict({**faces, "edges": faces_edges})

            # --------------------------------------------------
            # Face incidence (replace_role — SAME AS BEFORE)
            # --------------------------------------------------
            # fL -> fR
           # fs, fr = replace_role(fs, fr, e_jk, fL, fR)
            fs, fr = replace_role(fs, fr, e_ki, fL, fR)

            # fR -> fL
           # fs, fr = replace_role(fs, fr, e_il, fR, fL)
            fs, fr = replace_role(fs, fr, e_lj, fR, fL)

            # --------------------------------------------------
            # Neighbors
            # --------------------------------------------------
            neigh = remove_neighbor(neigh, i, j)
            neigh = remove_neighbor(neigh, j, i)
            neigh = add_neighbor(neigh, k, l)
            neigh = add_neighbor(neigh, l, k)

            return s, r, new_faces, fs, fr, neigh

        return jax.lax.cond(e >= 0, do_flip, no_op)

    def undo_flip_serial_full(
            senders,
            receivers,
            faces,
            face_senders,
            face_receivers,
            node_neighbors,
            agg_row,
    ):
        (
            i, j, k, l,
            e, fL, fR,
            e_jk, e_ki, e_il, e_lj,
            s_old, r_old,
        ) = agg_row

        def no_op():
            return senders, receivers, faces, face_senders, face_receivers, node_neighbors

        def undo():
            s = senders
            r = receivers
            fs = face_senders
            fr = face_receivers
            neigh = node_neighbors

            # --------------------------------------------------
            # Restore edge orientation
            # --------------------------------------------------
            s = s.at[e].set(s_old)
            r = r.at[e].set(r_old)

            # --------------------------------------------------
            # Restore face edges
            # --------------------------------------------------
            faces_edges = faces["edges"]
            faces_edges = (
                faces_edges
                .at[fL].set(jnp.array([e, e_jk, e_ki], dtype=jnp.int32))
                .at[fR].set(jnp.array([e, e_il, e_lj], dtype=jnp.int32))
            )
            new_faces = frozendict({**faces, "edges": faces_edges})

            # --------------------------------------------------
            # Restore face incidence (replace_role — inverse)
            # --------------------------------------------------
            # fR -> fL

         #   new_fs = face_senders.copy()
         #   new_fr = face_receivers.copy()
         #   new_fs, new_fr = replace_role(new_fs, new_fr, e_lj, fR, fL)
         #   new_fs, new_fr = replace_role(new_fs, new_fr, e_ki, fL, fR)

          #  fs, fr = replace_role(fs, fr, e_jk, fR, fL)
            fs, fr = replace_role(fs, fr, e_ki, fR, fL)

            # fL -> fR
           # fs, fr = replace_role(fs, fr, e_il, fL, fR)
            fs, fr = replace_role(fs, fr, e_lj, fL, fR)

            # --------------------------------------------------
            # Restore neighbors
            # --------------------------------------------------
            neigh = remove_neighbor(neigh, k, l)
            neigh = remove_neighbor(neigh, l, k)
            neigh = add_neighbor(neigh, i, j)
            neigh = add_neighbor(neigh, j, i)

            return s, r, new_faces, fs, fr, neigh

        return jax.lax.cond(e >= 0, undo, no_op)

    def apply_aggregate_serial_full(
            senders,
            receivers,
            faces,
            face_senders,
            face_receivers,
            node_neighbors,
            aggregate,
            forward=True,
    ):
        def body(state, agg_row):
            if forward:
                return apply_flip_serial_full(*state, agg_row), None
            else:
                return undo_flip_serial_full(*state, agg_row), None

        init = (
            senders,
            receivers,
            faces,
            face_senders,
            face_receivers,
            node_neighbors,
        )

        (senders, receivers, faces, face_senders, face_receivers, node_neighbors), _ = (
            jax.lax.scan(body, init, aggregate)
        )

        return senders, receivers, faces, face_senders, face_receivers, node_neighbors



    def commit_flip_from_aggregate(
            senders,
            receivers,
            faces,
            node_neighbors,
            agg_row,
    ):
        (
            i, j, k, l,
            e, fL, fR,
            e_jk, e_ki, e_il, e_lj,
            _, _,
        ) = agg_row

        def no_op():
            return senders, receivers, faces, node_neighbors

        def do_flip():
            # --- edge orientation ---
            new_s = senders.at[e].set(k)
            new_r = receivers.at[e].set(l)

            # --- faces ---
            faces_edges = faces["edges"]
            faces_edges = (
                faces_edges
                .at[fL].set(jnp.array([e, e_lj, e_jk], dtype=jnp.int32))
                .at[fR].set(jnp.array([e, e_ki, e_il], dtype=jnp.int32))
            )
            new_faces = frozendict({**faces, "edges": faces_edges})

            # --- neighbors ---
            neigh = node_neighbors
            neigh = remove_neighbor(neigh, i, j)
            neigh = remove_neighbor(neigh, j, i)
            neigh = add_neighbor(neigh, k, l)
            neigh = add_neighbor(neigh, l, k)

            return new_s, new_r, new_faces, neigh

        return jax.lax.cond(e >= 0, do_flip, no_op)

    def undo_flip_from_aggregate(
            senders,
            receivers,
            faces,
            node_neighbors,
            agg_row,
    ):
        (
            i, j, k, l,
            e, fL, fR,
            e_jk, e_ki, e_il, e_lj,
            s_old, r_old,
        ) = agg_row

        def no_op():
            return senders, receivers, faces, node_neighbors

        def undo():
            # --- restore edge orientation ---
            new_s = senders.at[e].set(s_old)
            new_r = receivers.at[e].set(r_old)

            # --- restore faces ---
            faces_edges = faces["edges"]
            faces_edges = (
                faces_edges
                .at[fL].set(jnp.array([e, e_jk, e_ki], dtype=jnp.int32))
                .at[fR].set(jnp.array([e, e_il, e_lj], dtype=jnp.int32))
            )
            new_faces = frozendict({**faces, "edges": faces_edges})

            # --- restore neighbors ---
            neigh = node_neighbors
            neigh = remove_neighbor(neigh, k, l)
            neigh = remove_neighbor(neigh, l, k)
            neigh = add_neighbor(neigh, i, j)
            neigh = add_neighbor(neigh, j, i)

            return new_s, new_r, new_faces, neigh

        return jax.lax.cond(e >= 0, undo, no_op)

    def apply_aggregate_serial(
            senders,
            receivers,
            faces,
            node_neighbors,
            aggregate,
            forward=True,
    ):
        def body(state, agg_row):
            s, r, f, nbh = state
            if forward:
                return commit_flip_from_aggregate(s, r, f, nbh, agg_row), None
            else:
                return undo_flip_from_aggregate(s, r, f, nbh, agg_row), None

        init = (senders, receivers, faces, node_neighbors)
        (senders, receivers, faces, node_neighbors), _ = jax.lax.scan(
            body, init, aggregate
        )

        return senders, receivers, faces, node_neighbors

    aggregate = compute_flip_aggregate(
        senders,
        receivers,
        faces,
        face_senders,
        face_receivers,
        flip_edges,
    )

    aggregate, rng_key = resolve_flip_conflicts(
        aggregate,
        node_neighbors,
        face_senders,
        face_receivers,
        rng_key,
        n_node_max,
    )

   # import jax
   # import jax.numpy as jnp

    def compactify_aggregate(aggregate, Kf, valid_col=5):
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
            faces,
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
        faces_edges = faces["edges"]

        fL_vals = jnp.stack([e, e_lj, e_jk], axis=1)
        fR_vals = jnp.stack([e, e_ki, e_il], axis=1)

        faces_edges = masked_scatter(faces_edges, fL, fL_vals, mask)
        faces_edges = masked_scatter(faces_edges, fR, fR_vals, mask)

        faces = frozendict({**faces, "edges": faces_edges})

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
            faces,
            face_senders,
            face_receivers,
            node_neighbors,
        )

    def undo_flip_parallel_full(
            senders,
            receivers,
            faces,
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
        faces_edges = faces["edges"]


        fL_vals = jnp.stack([e, e_jk, e_ki], axis=1)
        fR_vals = jnp.stack([e, e_il, e_lj], axis=1)

        faces_edges = masked_scatter(faces_edges, fL, fL_vals, mask)
        faces_edges = masked_scatter(faces_edges, fR, fR_vals, mask)

        faces = frozendict({**faces, "edges": faces_edges})

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
            faces,
            face_senders,
            face_receivers,
            node_neighbors,
        )

    Kf = n_edge_max // 80
    aggregate=compactify_aggregate(aggregate,Kf)


    senders, receivers, faces, face_senders, face_receivers, node_neighbors = (
       apply_flip_parallel_full(
           senders,
           receivers,
           faces,
           face_senders,
           face_receivers,
           node_neighbors,
           aggregate,

       )
   )

 #   senders, receivers, faces, face_senders, face_receivers, node_neighbors = (
 #       apply_aggregate_serial_full(
 #           senders,
 #           receivers,
 #           faces,
 #           face_senders,
 #           face_receivers,
 #           node_neighbors,
 #           aggregate,
 #           forward=True,
 #       )
 #   )

  #  face_senders, face_receivers = rebuild_face_incidence_from_faces(
  #      faces["edges"],
  #      n_edges=senders.shape[0],
  #  )

    # flip_mask = luby_mis_flip_mask(
   #     senders, receivers,
   #     face_senders, face_receivers,
   #     edges["flip_site"],
   #     rng_key)

  #  flip_mask = independent_flip_mask(
  #      senders, receivers,
  #      face_senders, face_receivers,
  #      edges["flip_site"], K
  #  )

    #flip_mask=flip_candidates & edges["flip_site"]
   # flip_edges,safe_edges,valid = extract_flip_edges(edges["flip_site"],K)
   # flip_edges, safe_edges, valid = extract_flip_edges(flip_mask, K)

    #flip_edges=select_independent_flips(senders, receivers,
    #face_senders, face_receivers, face_edges,
    #flip_edges)

   # flip_edges = flip_edges[:1]

    if flip_edge_fn:
        #priority = jax.random.uniform(rng_sample, shape=(n_edge_max,))
        #(
        #    senders,
        #    receivers,
        #    faces_packed,
        #    face_senders,
        #    face_receivers,face_vertices)=  flip_edge_fn(
        #senders,  # (E,) int32
        #receivers,  # (E,) int32
        #face_senders,  # (E,) int32  (face id or -1)
        #face_receivers,  # (E,) int32  (face id or -1)
        #edges_topo["faces_packed"],  # (E,2,6) int32 as described
        #edges["flip_site"],
        #priority,face_vertices)
        #edges_topo = frozendict({**edges_topo,
        #    "faces_packed":faces_packed
        #})

        def apply_flips_serial(
                senders,
                receivers,
                faces,
                face_senders,
                face_receivers,
                flip_edges,  # (K,)
                node_neighbors,
        ):
            import jax
            import jax.numpy as jnp

            """
            Serially apply edge flips in a JAX-safe way.

            Returns:
                aggregate : (K, 7) int32
                    One row per entry in flip_edges (matches vectorized version).
                    Rows are -1 for sentinel edges (e < 0).
            """

            # Shape must match flip_edge_fn aggregate output
            dummy_agg = jnp.full((1, 7), -1, dtype=jnp.int32)

            def body(state, e):
                (
                    senders,
                    receivers,
                    faces,
                    fs,
                    fr,
                    node_neighbors,
                ) = state

                def do_flip():
                    return flip_edge_fn(
                        senders,
                        receivers,
                        faces,
                        fs,
                        fr,
                        jnp.array([e], dtype=jnp.int32),  # K = 1
                        node_neighbors,
                    )

                def no_flip():
                    return (
                        senders,
                        receivers,
                        faces,
                        fs,
                        fr,
                        dummy_agg,
                        node_neighbors,
                    )

                (
                    senders2,
                    receivers2,
                    faces2,
                    fs2,
                    fr2,
                    aggregate2,  # (1, 7)
                    node_neighbors2,
                ) = jax.lax.cond(e >= 0, do_flip, no_flip)

                # Carry updated mesh state
                new_state = (
                    senders2,
                    receivers2,
                    faces2,
                    fs2,
                    fr2,
                    node_neighbors2,
                )

                # Output aggregate row for this step
                return new_state, aggregate2[0]

            init_state = (
                senders,
                receivers,
                faces,
                face_senders,
                face_receivers,
                node_neighbors,
            )

            (
                (senders, receivers, faces, face_senders, face_receivers, node_neighbors),
                aggregate,  # (K, 7)
            ) = jax.lax.scan(body, init_state, flip_edges)

            return (
                senders,
                receivers,
                faces,
                face_senders,
                face_receivers,
                aggregate,
                node_neighbors,
            )



     #   senders, receivers, faces, fs, fr = apply_flip_topology(
     #       senders,
     #       receivers,
     #       faces,
     #       face_senders,
     #       face_receivers,
     #       aggregate,
     #       final_mask
     #   )

      #  senders, receivers, faces, face_senders, face_receivers, aggregate,node_neighbors = apply_flips_serial(
      #      senders,
      #      receivers,
      #      faces,
      #      face_senders,
      #      face_receivers,
      #      flip_edges,
      #      node_neighbors
      #  )

    #    (
    #        senders,
    #        receivers,
    #        faces,
    #        face_senders,
    #        face_receivers,
    #        flip_edges,
    #        aggregate,
   #         node_neighbors,
   #         rng_key,
   #     ) = flip_edge_fn(
   #             senders,
   #             receivers,
   #             faces,
   #             face_senders,
   #             face_receivers,
   #             flip_edges,  # (K,) padded with -1
   #             node_neighbors,
   #             rng_key,
   #             n_node_max,
   #     )

    #    aggregate1=aggregate



        def per_flip_energy_sum_old(node_energy, edge_energy, face_energy, aggregate):
            quad_nodes = aggregate[:, :4]  # (K,4)
            flip_edge = aggregate[:, 4]  # (K,)
            flip_faces = aggregate[:, 5:7]  # (K,2)

            # mask invalid = -1 → contribute zero
            valid_nodes = quad_nodes >= 0
            valid_edge = flip_edge >= 0
            valid_faces = flip_faces >= 0

            E_nodes = jnp.where(valid_nodes, node_energy[quad_nodes], 0.0).sum(axis=1)
            E_edge = jnp.where(valid_edge, edge_energy[flip_edge], 0.0)
            E_faces = jnp.where(valid_faces, face_energy[flip_faces], 0.0).sum(axis=1)

            return E_nodes + E_edge + E_faces  # (K,)

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

            return E_nodes + E_edges + E_faces  # (K,)



    energy_before=per_flip_energy_sum(nodes["energy"],edges['energy'],faces['energy'],aggregate)






    ### RECOMPUTE
    node_mask,edge_mask,face_mask=build_local_mask_fn(aggregate,n_node_max,n_edge_max,n_face_max)

    if update_edge_local_fn:

      #  faces = rebuild_faces_from_dual_fast(
      #      senders,
      #      receivers,
      #      face_senders,
      #      face_receivers,
      #      faces
      #  )

        face_edges = faces["edges"]

        #face_vertices = faces["vertices"]
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

        if update_edge_local_fn:
            edges = update_edge_local_fn(edges,  sent_attributes, received_attributes,
                                   edge_mask,global_edge_attributes)

        if update_face_local_fn:
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
            faces = update_face_local_fn(faces, received_edge_attributes, sign, face_mask, global_attributes)

        if update_edge_from_face_local_fn:
            sent_face_attributes = tree.tree_map(lambda n: n[face_senders], faces)
            received_face_attributes = tree.tree_map(lambda n: n[face_receivers], faces)
            edges = update_edge_from_face_local_fn(edges, sent_face_attributes, received_face_attributes,
                                             edge_mask,global_edge_attributes)


        if update_node_local_fn:
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
                                   received_attributes, node_mask,global_attributes)



        energy_after = per_flip_energy_sum(nodes["energy"], edges['energy'], faces['energy'], aggregate)

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

        flip_edges,accepted_fraction,rng_key=metropolis_fn(aggregate[:, 4],
            energy_before,
            energy_after,
            1,
            rng_key,
        )

        aggregate = mask_aggregate_by_flip_edges(aggregate, flip_edges)
        flip_edges=aggregate[:, 4]

        #senders, receivers, faces, node_neighbors = apply_aggregate_serial(
        #    senders,
        #    receivers,
        #    faces,
        #    node_neighbors,
        #    aggregate,
        #    forward=False,  # ← undo
        #)

        #face_senders, face_receivers = rebuild_face_incidence_from_faces(
        #    faces["edges"],
        #    n_edges=senders.shape[0],
        #)
      #  flip_edges

        senders, receivers, faces, face_senders, face_receivers, node_neighbors = (
        undo_flip_parallel_full(
            senders,
            receivers,
            faces,
            face_senders,
            face_receivers,
            node_neighbors,
            aggregate,

        )
        )

    #    senders, receivers, faces, face_senders, face_receivers, node_neighbors = (
    #        apply_aggregate_serial_full(
    #            senders,
    #            receivers,
    #            faces,
    #            face_senders,
    #            face_receivers,
    #            node_neighbors,
    #            aggregate,
    #            forward=False,
    #        )
    #    )

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

        def flip_edges_topology_no_conflicts(
                senders,
                receivers,
                faces,
                face_senders,
                face_receivers,
                flip_edges,  # (K,) padded with -1
                node_neighbors,
        ):
            """
            Same semantics as flip_edges_topology_no_conflicts,
            but GUARANTEES index safety:
              - no scatter ever sees -1
              - masked flips do nothing
            """

            faces_edges = faces["edges"]
            old_s = senders
            old_r = receivers
            num_faces = faces_edges.shape[0]

            # ============================================================
            # 1. PRECOMPUTE AGGREGATE (unchanged logic)
            # ============================================================

            valid = flip_edges >= 0
            e = jnp.where(valid, flip_edges, -1)

            fL = jnp.where(valid, face_senders[e], -1)
            fR = jnp.where(valid, face_receivers[e], -1)

            valid &= (fL >= 0) & (fR >= 0)

            i = jnp.where(valid, old_s[e], -1)
            j = jnp.where(valid, old_r[e], -1)

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

            valid &= (
                    (k >= 0) & (l >= 0) &
                    (k != l) &
                    (i != k) & (i != l) &
                    (j != k) & (j != l)
            )

            own_L = unique_owner(valid, fL, num_faces)
            own_R = unique_owner(valid, fR, num_faces)
            valid &= own_L & own_R

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

            valid &= (e_jk >= 0) & (e_ki >= 0) & (e_il >= 0) & (e_lj >= 0)

            def m(x): return jnp.where(valid, x, -1)

            aggregate = jnp.stack(
                [m(i), m(j), m(k), m(l),
                 m(e), m(fL), m(fR),
                 m(e_jk), m(e_ki), m(e_il), m(e_lj)],
                axis=1
            ).astype(jnp.int32)

            new_flip_edges = aggregate[:, 4]

            # ============================================================
            # 2. COMMIT FLIPS (INDEX-SAFE)
            # ============================================================

            (
                i, j, k, l,
                e, fL, fR,
                e_jk, e_ki, e_il, e_lj
            ) = aggregate.T

            v = e >= 0
            vb = v[:, None]

            # ---- SAFE INDICES (CRITICAL FIX) ----
            safe_e = jnp.where(v, e, 0)
            safe_fL = jnp.where(v, fL, 0)
            safe_fR = jnp.where(v, fR, 0)
            safe_ejk = jnp.where(v, e_jk, 0)
            safe_eki = jnp.where(v, e_ki, 0)
            safe_eil = jnp.where(v, e_il, 0)
            safe_elj = jnp.where(v, e_lj, 0)

            # ---- edge endpoints ----
            new_senders = senders.at[safe_e].set(
                jnp.where(v, k, senders[safe_e])
            )
            new_receivers = receivers.at[safe_e].set(
                jnp.where(v, l, receivers[safe_e])
            )

            # ---- node neighbors (already safe via v) ----
            def remove_neighbor(neigh, node, nbr):
                row = neigh[node]
                row = jnp.where(row == nbr, -1, row)
                return neigh.at[node].set(row)

            def add_neighbor(neigh, node, nbr):
                row = neigh[node]
                idx = jnp.argmax(row == -1)
                row = row.at[idx].set(nbr)
                return neigh.at[node].set(row)

            def update_neighbors(neigh, a, b, mask, fn):
                def body(t, nbh):
                    return jax.lax.cond(
                        mask[t],
                        lambda x: fn(x, a[t], b[t]),
                        lambda x: x,
                        nbh
                    )

                return jax.lax.fori_loop(0, a.shape[0], body, neigh)

            node_neighbors = update_neighbors(node_neighbors, i, j, v, remove_neighbor)
            node_neighbors = update_neighbors(node_neighbors, j, i, v, remove_neighbor)
            node_neighbors = update_neighbors(node_neighbors, k, l, v, add_neighbor)
            node_neighbors = update_neighbors(node_neighbors, l, k, v, add_neighbor)

            # ---- faces["edges"] ----
            new_faces_edges = faces_edges
            new_faces_edges = new_faces_edges.at[safe_fL].set(
                jnp.where(vb, jnp.stack([e, e_lj, e_jk], axis=1),
                          new_faces_edges[safe_fL])
            )
            new_faces_edges = new_faces_edges.at[safe_fR].set(
                jnp.where(vb, jnp.stack([e, e_ki, e_il], axis=1),
                          new_faces_edges[safe_fR])
            )

            # ---- dual adjacency ----
            def replace_role(fs, fr, edge_ids, old_face, new_face):
                valid_edge = edge_ids >= 0
                safe_edge = jnp.where(valid_edge, edge_ids, 0)

                fs_e = fs[safe_edge]
                fr_e = fr[safe_edge]

                fs_e = jnp.where(valid_edge & (fs_e == old_face), new_face, fs_e)
                fr_e = jnp.where(valid_edge & (fr_e == old_face), new_face, fr_e)

                fs = fs.at[safe_edge].set(fs_e)
                fr = fr.at[safe_edge].set(fr_e)
                return fs, fr

            new_fs = face_senders.at[safe_e].set(
                jnp.where(v, fL, face_senders[safe_e])
            )
            new_fr = face_receivers.at[safe_e].set(
                jnp.where(v, fR, face_receivers[safe_e])
            )

            new_fs, new_fr = replace_role(new_fs, new_fr, e_jk, fL, fL)
            new_fs, new_fr = replace_role(new_fs, new_fr, e_lj, fR, fL)
            new_fs, new_fr = replace_role(new_fs, new_fr, e_ki, fL, fR)
            new_fs, new_fr = replace_role(new_fs, new_fr, e_il, fR, fR)

            new_faces = frozendict({**faces, "edges": new_faces_edges})

            return (
                new_senders,
                new_receivers,
                new_faces,
                new_fs,
                new_fr,
                new_flip_edges,
                aggregate,
                node_neighbors,
            )

        def flip_edges_topology_no_conflicts(
                senders,
                receivers,
                faces,
                face_senders,
                face_receivers,
                flip_edges,  # (K,) padded with -1
                node_neighbors,
        ):
            """
            Identical to flip_edges_topology_with_conflicts,
            but WITHOUT conflict resolution.

            Every geometrically valid flip is committed.

            Returns:
              new_senders,
              new_receivers,
              new_faces,
              new_face_senders,
              new_face_receivers,
              new_flip_edges,   # (K,) (same as input, masked by validity)
              aggregate,        # (K,11)
              node_neighbors
            """

            faces_edges = faces["edges"]
            old_s = senders
            old_r = receivers
            num_faces = faces_edges.shape[0]

            # ============================================================
            # 1. PRECOMPUTE AGGREGATE (IDENTICAL LOGIC)
            # ============================================================

            valid = flip_edges >= 0
            e = jnp.where(valid, flip_edges, -1)

            fL = jnp.where(valid, face_senders[e], -1)
            fR = jnp.where(valid, face_receivers[e], -1)

            interior = (fL >= 0) & (fR >= 0)
            valid &= interior

            i = jnp.where(valid, old_s[e], -1)
            j = jnp.where(valid, old_r[e], -1)

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
            valid &= quad_ok

            # one flip per face (same as original)
            own_L = unique_owner(valid, fL, num_faces)
            own_R = unique_owner(valid, fR, num_faces)
            valid &= own_L & own_R

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
            valid &= neigh_ok

            def m(x): return jnp.where(valid, x, -1)

            aggregate = jnp.stack(
                [
                    m(i), m(j), m(k), m(l),
                    m(e), m(fL), m(fR),
                    m(e_jk), m(e_ki), m(e_il), m(e_lj),
                ],
                axis=1
            ).astype(jnp.int32)

            #final flip edges = all valid flips
            new_flip_edges = aggregate[:, 4]

            # ============================================================
            # 2. COMMIT FLIPS (IDENTICAL APPLY LOGIC)
            # ============================================================

            (
                i, j, k, l,
                e, fL, fR,
                e_jk, e_ki, e_il, e_lj
            ) = aggregate.T

            v = e >= 0
            vb = v[:, None]

            new_senders = senders.at[e].set(jnp.where(v, k, senders[e]))
            new_receivers = receivers.at[e].set(jnp.where(v, l, receivers[e]))

            # ------------------------------------------------------------
            # Update node_neighbors for edge flip (i,j) -> (k,l)
            # ------------------------------------------------------------

            def remove_neighbor(neigh, node, nbr):
                row = neigh[node]
                row = jnp.where(row == nbr, -1, row)
                return neigh.at[node].set(row)

            def add_neighbor(neigh, node, nbr):
                row = neigh[node]
                idx = jnp.argmax(row == -1)  # first free slot
                row = row.at[idx].set(nbr)
                return neigh.at[node].set(row)

            def update_neighbors(neigh, a, b, mask, fn):
                # applies fn(neigh, a[t], b[t]) if mask[t] is True
                def body(t, nbh):
                    return jax.lax.cond(
                        mask[t],
                        lambda x: fn(x, a[t], b[t]),
                        lambda x: x,
                        nbh
                    )

                return jax.lax.fori_loop(0, a.shape[0], body, neigh)

            # v = e >= 0 already defined
            node_neighbors = update_neighbors(node_neighbors, i, j, v, remove_neighbor)
            node_neighbors = update_neighbors(node_neighbors, j, i, v, remove_neighbor)

            node_neighbors = update_neighbors(node_neighbors, k, l, v, add_neighbor)
            node_neighbors = update_neighbors(node_neighbors, l, k, v, add_neighbor)

            new_faces_edges = faces_edges
            new_faces_edges = new_faces_edges.at[fL].set(
                jnp.where(vb, jnp.stack([e, e_lj, e_jk], axis=1), new_faces_edges[fL])
            )
            new_faces_edges = new_faces_edges.at[fR].set(
                jnp.where(vb, jnp.stack([e, e_ki, e_il], axis=1), new_faces_edges[fR])
            )

            def replace_role(fs, fr, edge_ids, old_face, new_face):
                valid_edge = edge_ids >= 0
                fs_edge = fs[edge_ids]
                fr_edge = fr[edge_ids]

                fs_edge = jnp.where(valid_edge & (fs_edge == old_face), new_face, fs_edge)
                fr_edge = jnp.where(valid_edge & (fr_edge == old_face), new_face, fr_edge)

                fs = fs.at[edge_ids].set(fs_edge)
                fr = fr.at[edge_ids].set(fr_edge)
                return fs, fr


            new_fs = face_senders.copy()
            new_fr = face_receivers.copy()
           # new_fs = face_senders.at[e].set(jnp.where(v, fL, face_senders[e]))
           # new_fr = face_receivers.at[e].set(jnp.where(v, fR, face_receivers[e]))

            #new_fs=face_senders.copy()
            #new_fr = face_receivers.copy()

           # new_fs, new_fr = replace_role(new_fs, new_fr, e_jk, fL, fL)
            new_fs, new_fr = replace_role(new_fs, new_fr, e_lj, fR, fL)
            new_fs, new_fr = replace_role(new_fs, new_fr, e_ki, fL, fR)
          #  new_fs, new_fr = replace_role(new_fs, new_fr, e_il, fR, fR)

            new_faces = frozendict({**faces, "edges": new_faces_edges})

            return (
                new_senders,
                new_receivers,
                new_faces,
                new_fs,
                new_fr,
                new_flip_edges,
                aggregate,
                node_neighbors,
            )







      #  (
      #      senders,
      #      receivers,
      #      faces,
      #      face_senders,
      #      face_receivers,
      #      flip_edges,
      #      aggregate,
      #      node_neighbors,
      #  ) = flip_edges_topology_no_conflicts(
      #      senders,
      #      receivers,
      #      faces,
      #      face_senders,
      #      face_receivers,
      #      flip_edges,  # (K,) padded with -1
      #      node_neighbors,

  #      )

      #  (
      #      senders,
      #      receivers,
      #      faces,
      #      face_senders,
      #      face_receivers,
      #      flip_edges,
      #      aggregate,
      #      node_neighbors,
      #      rng_key,
      #  ) = flip_edge_fn(
      #      senders,
      #      receivers,
      #      faces,
      #      face_senders,
      #      face_receivers,
      #      flip_edges,  # (K,) padded with -1
      #      node_neighbors,
      #      rng_key,
      #      n_node_max,
      #  )

      ##  faces = rebuild_faces_from_dual(
       #     senders,
       #     receivers,
       #     face_senders,
       #     face_receivers,
       #     faces
       # )



      #  senders, receivers, faces, face_senders, face_receivers, aggregate, node_neighbors= apply_flips_serial(
      #     senders,
      #  receivers,
      #      faces,
      #      face_senders,
      #      face_receivers,
      #      flip_edges,
      #      node_neighbors
      #  )

    #energy_after = per_flip_energy_sum(nodes["energy"], edges['energy'], faces['energy'], aggregate)


    globals_=frozendict({**globals_,"flips_per":jnp.sum(flip_mask)/n_edge_max,"flip_edge_per":jnp.sum(jnp.int32(flip_edges>=0))/n_edge_max,"flips_acc":accepted_fraction,"E_b":jnp.sum(energy_before),"E_a":jnp.sum(energy_after)})



    face_vertices=faces["vertices"]

    nodes=frozendict({**nodes,"neighbors":node_neighbors})




    #edges=frozendict({**edges,"flip_site":flip_mask})

       # print(face_vertices.shape, face_edges.shape,face_orient)

        #senders,receivers,face_senders, face_receivers,face_edges, face_vertices,faces, edges, nodes = flip_edge_fn(
        #    senders, receivers,  # [n_edge]
        #    face_senders, face_receivers,  # [n_edge] -> two adjacent faces per edge
        #    face_edges,  # [n_face,3] edge ids ordered to match face_vertices
        #    face_vertices,  # [n_face,3] vertex ids in CCW order
        #    faces["edge_orientation"],  # [n_face,3] +1/-1 orientation
        #    nodes, edges, faces,
        #    n_edge_max  # static integer: max number of flips to consider
        #)
    #j
    #nodes=frozendict(**nodes,"flip_site")


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
        receivers=receivers,
        senders=senders,
        face_receivers=face_receivers,
        face_senders=face_senders,
        face_vertices=face_vertices,
        globals=globals_,
        n_node=n_node,
        n_edge=n_edge,
        n_face=n_face,
        rng_key=rng_key,
        n_node_max=n_node_max,
        n_edge_max=n_edge_max,
        n_face_max=n_face_max)

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
