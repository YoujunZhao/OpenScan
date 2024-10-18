import networkx as nx
from graph.node import Node
import torch

def cluster_into_new_nodes(iteration, old_nodes, graph):
    new_nodes = []
    for component in nx.connected_components(graph):
        node_info = (iteration, len(new_nodes))
        new_nodes.append(Node.create_node_from_list([old_nodes[node] for node in component], node_info))
    return new_nodes


def update_graph(nodes, observer_num_threshold, connect_threshold):
    '''
        update view consensus rates between nodes and return a new graph
    '''
    node_visible_frames = torch.stack([node.visible_frame for node in nodes], dim=0)
    node_contained_masks = torch.stack([node.contained_mask for node in nodes], dim=0)

    observer_nums = torch.matmul(node_visible_frames, node_visible_frames.transpose(0,1)) # M[i,j] stores the number of frames that node i and node j both appear
    supporter_nums = torch.matmul(node_contained_masks, node_contained_masks.transpose(0,1)) # M[i,j] stores the number of frames that supports the merging of node i and node j

    view_concensus_rate = supporter_nums / (observer_nums + 1e-7)

    disconnect = torch.eye(len(nodes), dtype=bool).cuda()
    disconnect = disconnect | (observer_nums < observer_num_threshold) # node pairs with less than observer_num_threshold observers are disconnected

    A = view_concensus_rate >= connect_threshold
    A = A & ~disconnect
    A = A.cpu().numpy()

    G = nx.from_numpy_array(A)
    return G


def iterative_clustering(nodes, observer_num_thresholds, connect_threshold, debug):
    if debug:
        print('====> Start iterative clustering')
    for iterate_id, observer_num_threshold in enumerate(observer_num_thresholds):
        if debug:
            print(f'Iterate {iterate_id}: observer_num', observer_num_threshold, ', number of nodes', len(nodes))
        graph = update_graph(nodes, observer_num_threshold, connect_threshold)
        nodes = cluster_into_new_nodes(iterate_id+1, nodes, graph)
    return nodes