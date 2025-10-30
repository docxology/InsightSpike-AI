"""
PyTorch Geometric Compatible Metrics
====================================

Provides GED and IG calculations for PyTorch Geometric Data objects.
"""

import logging
import networkx as nx
import numpy as np
from typing import Any, Optional

logger = logging.getLogger(__name__)


def pyg_to_networkx(pyg_graph: Any) -> nx.Graph:
    """Convert a minimal PyG-like Data object to NetworkX safely.

    Robust to either torch.Tensor or numpy.ndarray (or array-like) for
    `x`, `edge_index`, and `edge_attr`.
    """
    try:
        G = nx.Graph()

        # Resolve node count
        num_nodes = None
        try:
            if hasattr(pyg_graph, 'num_nodes') and isinstance(pyg_graph.num_nodes, (int, np.integer)):
                num_nodes = int(pyg_graph.num_nodes)
            elif hasattr(pyg_graph, 'x') and pyg_graph.x is not None:
                x_obj = getattr(pyg_graph, 'x')
                if hasattr(x_obj, 'shape') and len(x_obj.shape) >= 1:
                    num_nodes = int(x_obj.shape[0])
        except Exception:
            num_nodes = None
        if not num_nodes:
            return G  # empty

        # Materialize node feature matrix as numpy if present
        feats_np: Optional[np.ndarray] = None
        if hasattr(pyg_graph, 'x') and pyg_graph.x is not None:
            x_obj = pyg_graph.x
            try:
                if hasattr(x_obj, 'detach'):
                    x_obj = x_obj.detach()
                if hasattr(x_obj, 'cpu'):
                    x_obj = x_obj.cpu()
                if hasattr(x_obj, 'numpy'):
                    feats_np = x_obj.numpy()
                else:
                    feats_np = np.asarray(x_obj)
            except Exception:
                feats_np = None

        # Add nodes
        for i in range(int(num_nodes)):
            node_attrs = {'idx': int(i)}
            if feats_np is not None and feats_np.ndim == 2 and i < feats_np.shape[0]:
                try:
                    node_attrs['features'] = feats_np[i]
                except Exception:
                    pass
            G.add_node(int(i), **node_attrs)

        # Edges
        if hasattr(pyg_graph, 'edge_index') and getattr(pyg_graph, 'edge_index') is not None:
            ei = getattr(pyg_graph, 'edge_index')
            try:
                if hasattr(ei, 'detach'):
                    ei = ei.detach()
                if hasattr(ei, 'cpu'):
                    ei = ei.cpu()
                if hasattr(ei, 'numpy'):
                    ei_np = ei.numpy()
                else:
                    ei_np = np.asarray(ei)
            except Exception:
                ei_np = np.asarray(ei)

            # Normalize to shape (2, E)
            if ei_np.ndim == 2 and ei_np.shape[0] == 2:
                edges = [(int(ei_np[0, j]), int(ei_np[1, j])) for j in range(ei_np.shape[1])]
            elif ei_np.ndim == 2 and ei_np.shape[1] == 2:
                edges = [(int(ei_np[j, 0]), int(ei_np[j, 1])) for j in range(ei_np.shape[0])]
            else:
                edges = []

            # Optional edge weights
            weights = None
            if hasattr(pyg_graph, 'edge_attr') and getattr(pyg_graph, 'edge_attr') is not None:
                try:
                    ea = getattr(pyg_graph, 'edge_attr')
                    if hasattr(ea, 'detach'):
                        ea = ea.detach()
                    if hasattr(ea, 'cpu'):
                        ea = ea.cpu()
                    if hasattr(ea, 'numpy'):
                        ea_np = ea.numpy()
                    else:
                        ea_np = np.asarray(ea)
                    weights = ea_np
                except Exception:
                    weights = None

            if weights is not None:
                w = np.asarray(weights)
                for idx, (u, v) in enumerate(edges):
                    try:
                        if w.ndim == 1:
                            G.add_edge(u, v, weight=float(w[idx]))
                        else:
                            G.add_edge(u, v, weight=float(w[idx, 0]))
                    except Exception:
                        G.add_edge(u, v)
            else:
                G.add_edges_from(edges)

        return G
    except Exception as e:
        logger.error(f"Failed to convert PyG to NetworkX: {e}")
        return nx.Graph()


def delta_ged_pyg(g_old: Any, g_new: Any) -> float:
    """
    Calculate ΔGED for PyTorch Geometric graphs.
    
    Returns negative value when graph simplifies (insight formation).
    """
    try:
        # Convert to NetworkX
        nx_old = pyg_to_networkx(g_old)
        nx_new = pyg_to_networkx(g_new)
        
        # Basic size comparison for quick check
        old_nodes = nx_old.number_of_nodes()
        new_nodes = nx_new.number_of_nodes()
        old_edges = nx_old.number_of_edges()
        new_edges = nx_new.number_of_edges()
        
        logger.debug(f"Graph comparison: ({old_nodes},{old_edges}) -> ({new_nodes},{new_edges})")
        
        # Simple heuristic GED calculation
        # More sophisticated calculation would use actual graph edit distance
        node_diff = abs(new_nodes - old_nodes)
        edge_diff = abs(new_edges - old_edges)
        
        # Weight node changes more than edge changes
        ged = node_diff * 1.0 + edge_diff * 0.5
        
        # Make negative if graph simplified
        if new_nodes < old_nodes or (new_nodes == old_nodes and new_edges < old_edges):
            ged = -ged
            
        logger.debug(f"Calculated GED: {ged}")
        return float(ged)
        
    except Exception as e:
        logger.error(f"GED calculation failed: {e}")
        return 0.0


def delta_ig_pyg(g_old: Any, g_new: Any) -> float:
    """Calculate ΔIG for PyG graphs using variance proxy.

    Returns positive when information increases (old_var - new_var, normalized).
    Robust to numpy or torch feature matrices.
    """
    try:
        def as_numpy(x: Any) -> Optional[np.ndarray]:
            try:
                if x is None:
                    return None
                obj = x
                if hasattr(obj, 'detach'):
                    obj = obj.detach()
                if hasattr(obj, 'cpu'):
                    obj = obj.cpu()
                if hasattr(obj, 'numpy'):
                    return obj.numpy()
                return np.asarray(obj)
            except Exception:
                return None

        old_features = as_numpy(getattr(g_old, 'x', None))
        new_features = as_numpy(getattr(g_new, 'x', None))
        if old_features is None or new_features is None:
            return 0.0

        old_var = float(np.var(old_features))
        new_var = float(np.var(new_features))
        ig = old_var - new_var
        if old_var > 0:
            ig = ig / old_var
        logger.debug(f"Calculated IG: {ig} (old_var={old_var:.4f}, new_var={new_var:.4f})")
        return float(ig)
    except Exception as e:
        logger.error(f"IG calculation failed: {e}")
        return 0.0
