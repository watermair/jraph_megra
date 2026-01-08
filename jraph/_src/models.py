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
    nodes = frozendict({**nodes, "rn_min": jnp.ones(n_node_max)*jnp.inf})

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


    rn = jax.random.uniform(rng_sample, shape=(n_edge_max,))
    flip_candidates=edges["flip_site"].copy()
    edges = frozendict({**edges, "rn": jnp.where(flip_candidates,rn,jnp.ones(n_edge_max)*jnp.inf), "rn_min": jnp.where(flip_candidates,rn,jnp.ones(n_edge_max)*jnp.inf)})
    faces = frozendict({**faces, "rn_min": jnp.ones(n_face_max) * jnp.inf})
    nodes = frozendict({**nodes, "rn_min": jnp.ones(n_node_max) * jnp.inf})
    edges = frozendict({**edges, "flip_site": jnp.zeros((n_edge_max,), dtype=jnp.int32)})


    if update_node_selection_fn:
        sent_attributes = tree.tree_map(
            # lambda e: aggregate_edges_for_nodes_rflood_fn(e, senders, sum_n_node), edges)
            lambda e: aggregate_edges_for_nodes_rflood_fn(e, senders, sum_n_node), edges)
        received_attributes = tree.tree_map(
            lambda e: aggregate_edges_for_nodes_rflood_fn(e, receivers, sum_n_node),
            edges)
        # Here we scatter the global features to the corresponding nodes,
        # giving us tensors of shape [num_nodes, global_feat].
        global_attributes = tree.tree_map(lambda g: jnp.repeat(
            g, n_node, axis=0, total_repeat_length=sum_n_node), globals_)
        nodes = update_node_selection_fn(nodes, sent_attributes,
                                         received_attributes, global_attributes)

    if update_face_selection_fn:
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

        faces = update_face_selection_fn(faces, face_edges,nodes,senders,receivers, global_attributes)

    sent_face_attributes = tree.tree_map(lambda n: n[face_senders], faces)
    received_face_attributes = tree.tree_map(lambda n: n[face_receivers], faces)

    global_edge_attributes = tree.tree_map(lambda g: jnp.repeat(
        g, n_edge, axis=0, total_repeat_length=sum_n_edge), globals_)

    if update_edge_from_face_selection_fn:
      sent_face_attributes = tree.tree_map(lambda n: n[face_senders], faces)
      received_face_attributes = tree.tree_map(lambda n: n[face_receivers], faces)
      edges = update_edge_from_face_selection_fn(edges, sent_face_attributes, received_face_attributes,global_edge_attributes)


    if update_node_selection_fn:
        sent_attributes = tree.tree_map(
            # lambda e: aggregate_edges_for_nodes_rflood_fn(e, senders, sum_n_node), edges)
            lambda e: aggregate_edges_for_nodes_rflood_fn(e, senders, sum_n_node), edges)
        received_attributes = tree.tree_map(
            lambda e: aggregate_edges_for_nodes_rflood_fn(e, receivers, sum_n_node),
            edges)
        # Here we scatter the global features to the corresponding nodes,
        # giving us tensors of shape [num_nodes, global_feat].
        global_attributes = tree.tree_map(lambda g: jnp.repeat(
            g, n_node, axis=0, total_repeat_length=sum_n_node), globals_)
        nodes = update_node_selection_fn(nodes, sent_attributes,
                                         received_attributes, global_attributes)


    sent_attributes = tree.tree_map(lambda n: n[senders], nodes)
    received_attributes = tree.tree_map(lambda n: n[receivers], nodes)

 #   sent_face_attributes = tree.tree_map(lambda n: n[face_senders], faces)
 #   received_face_attributes = tree.tree_map(lambda n: n[face_receivers], faces)

    # Here we scatter the global features to the corresponding edges,
    # giving us tensors of shape [num_edges, global_feat].
    global_edge_attributes = tree.tree_map(lambda g: jnp.repeat(
        g, n_edge, axis=0, total_repeat_length=sum_n_edge), globals_)

    if update_edge_selection_fn:
        edges = update_edge_selection_fn(edges, sent_attributes, received_attributes,  global_edge_attributes)


    sent_attributes = tree.tree_map(
        # lambda e: aggregate_edges_for_nodes_rflood_fn(e, senders, sum_n_node), edges)
        lambda e: aggregate_edges_for_nodes_rflood_fn(e, senders, sum_n_node), edges)
    received_attributes = tree.tree_map(
        lambda e: aggregate_edges_for_nodes_rflood_fn(e, receivers, sum_n_node),
        edges)
    # Here we scatter the global features to the corresponding nodes,
    # giving us tensors of shape [num_nodes, global_feat].
    global_attributes = tree.tree_map(lambda g: jnp.repeat(
        g, n_node, axis=0, total_repeat_length=sum_n_node), globals_)
    nodes = update_node_selection_fn(nodes, sent_attributes,
                                     received_attributes, global_attributes)

    sent_attributes = tree.tree_map(lambda n: n[senders], nodes)
    received_attributes = tree.tree_map(lambda n: n[receivers], nodes)

    global_edge_attributes = tree.tree_map(lambda g: jnp.repeat(
        g, n_edge, axis=0, total_repeat_length=sum_n_edge), globals_)
    edges = update_edge_selection_fn(edges, sent_attributes, received_attributes, global_edge_attributes)


    flip_mask=edges["rn"]==edges["rn_min"]


   # if update_edge_flip_selection_fn:
   #     edges = update_edge_flip_selection_fn(edges, sent_attributes, received_attributes,sent_face_attributes,received_face_attributes,
   #                          global_edge_attributes)

   # flip_mask=flip_candidates

    from jax import lax



    def mis_kring_edges(
            senders,
            receivers,
            num_nodes,
            key,
            K=4,
            max_iters=8,
    ):

        import jax
        import jax.numpy as jnp
        from jax import lax
        """
        Luby-style MIS for edges with STRICT K-ring exclusion.
        Frontier-based expansion (non-idempotent).

        Parameters
        ----------
        senders, receivers : (E,) int
            Edge endpoints
        num_nodes : int
            Number of vertices
        key : PRNGKey
        K : int
            Ring depth (now meaningful)
        max_iters : int
            Luby iterations

        Returns
        -------
        selected : (E,) bool
            Maximal independent edge set
        """

        E = senders.shape[0]

        # -------------------------------------------------
        # E -> V -> E (1 expansion step)
        # -------------------------------------------------
        def ev_step(edge_mask):
            v = jnp.zeros(num_nodes, dtype=bool)
            v = v.at[senders].max(edge_mask)
            v = v.at[receivers].max(edge_mask)
            return v[senders] | v[receivers]

        # -------------------------------------------------
        # STRICT K-ring expansion using frontier tracking
        # -------------------------------------------------
        def block_kring_strict(seed_edges):
            blocked = seed_edges
            frontier = seed_edges

            def body(carry, _):
                blocked, frontier = carry
                expanded = ev_step(frontier)
                new = expanded & (~blocked)
                blocked = blocked | new
                frontier = new
                return (blocked, frontier), None

            (blocked, _), _ = lax.scan(
                body,
                (blocked, frontier),
                None,
                length=K,
            )
            return blocked

        # -------------------------------------------------
        # One Luby iteration
        # -------------------------------------------------
        def luby_step(carry, key):
            active, selected = carry

            key, subkey = jax.random.split(key)
            priority = jax.random.uniform(subkey, (E,))
            priority = jnp.where(active, priority, -jnp.inf)

            # local maxima (1-ring)
            v = jnp.full(num_nodes, -jnp.inf)
            v = v.at[senders].max(priority)
            v = v.at[receivers].max(priority)

            edge_max = jnp.maximum(v[senders], v[receivers])
            choose = active & (priority == edge_max)

            # strict K-ring block
            blocked = block_kring_strict(choose)

            active = active & (~blocked)
            selected = selected | choose

            return (active, selected), key

        # -------------------------------------------------
        # Run MIS iterations
        # -------------------------------------------------
        active0 = jnp.ones(E, dtype=bool)
        selected0 = jnp.zeros(E, dtype=bool)

        keys = jax.random.split(key, max_iters)

        (active, selected), _ = lax.scan(
            luby_step,
            (active0, selected0),
            keys,
        )

        return selected

    flip_mask = jnp.int32(mis_kring_edges(
        senders=senders,
        receivers=receivers,
        num_nodes=n_node_max,
        key=rng_key,
        K=2,  # 4-ring exclusion
        max_iters=20,  # usually enough
    ))














    K =  n_edge_max // 10

    def extract_flip_edges(flip_mask, K):
        # get candidate flip edges (possibly padded)
        flip_edges = jnp.nonzero(flip_mask, size=K, fill_value=-1)[0]

        # valid ones: >= 0
        valid = flip_edges >= 0

        # JAX-friendly "safe" set: keep invalid as -1 sentinel, not filtered away
        safe_edges = jnp.where(valid, flip_edges, -1)

        return flip_edges, safe_edges, valid



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
    flip_edges, safe_edges, valid = extract_flip_edges(flip_mask, K)

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

        senders, receivers, faces, face_senders, face_receivers, aggregate,node_neighbors = apply_flips_serial(
            senders,
            receivers,
            faces,
            face_senders,
            face_receivers,
            flip_edges,
            node_neighbors
        )

        def per_flip_energy_sum(node_energy, edge_energy, face_energy, aggregate):
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

    energy_before=per_flip_energy_sum(nodes["energy"],edges['energy'],faces['energy'],aggregate)






    ### RECOMPUTE
    node_mask,edge_mask,face_mask=build_local_mask_fn(aggregate,n_node_max,n_edge_max,n_face_max)

    if update_edge_local_fn:

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

        flip_edges,accepted_fraction,rng_key=metropolis_fn(flip_edges,
            energy_before,
            energy_after,
            10,
            rng_key,
        )


        senders, receivers, faces, face_senders, face_receivers, aggregate, node_neighbors= apply_flips_serial(
            senders,
            receivers,
            faces,
            face_senders,
            face_receivers,
            flip_edges,
            node_neighbors
        )


    globals_=frozendict({**globals_,"flips_per":jnp.sum(flip_mask)/n_edge_max,"flips_acc":accepted_fraction,"E_b":jnp.sum(energy_before),"E_a":jnp.sum(energy_after)})



    face_vertices=faces["vertices"]



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
