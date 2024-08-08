import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx
from scipy.sparse import coo_matrix


'''
visualize_flow: A function to visualize the flow on the grid network
@ Input:
G: A grid network
paths: A dictionary where each key represents a trip (origin, destination) and the corresponding value is a list of nodes that represent
the assigned path going from origin to destination node
od_matrix: An OD matrix

@ Output:
None. (Save the plot as: flow_visualization.png)

'''

def visualize_flow(G, paths, od_matrix, N, pos, labels, filename = 'flow_visualization.png'):
    for (u, v, d) in G.edges(data=True):
        d['flow'] = 0

    # Aggregate flows for each path based on the OD matrix
    for (o, d), path in paths.items():
        # Get the demand from the OD matrix
        demand = od_matrix[o[0] * N + o[1], d[0] * N + d[1]]
        # Add demand to the flow on each edge in the path for the direction of travel
        for i in range(len(path) - 1):
            if G.has_edge(path[i], path[i + 1]):
                G[path[i]][path[i + 1]]['flow'] += demand

    # Calculate the flows for color mapping
    flows = [d['flow'] for u, v, d in G.edges(data=True)]

    # Visualize the network with arrows indicating the direction and color intensity based on flow
    plt.figure(figsize=(8,8))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color='lightblue', label=labels)
    
    # Draw edges with distinct colors and arrows for each direction
    for (u, v, d) in G.edges(data=True):
        flow = d['flow']
        edge_color = plt.cm.Blues(flow / max(flows) if max(flows) > 0 else 1)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=2, edge_color=edge_color, connectionstyle='arc3, rad=0.1')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12)
    
    plt.title('Network Flow Visualization with Direction')
    #fig, ax = plt.subplots()
    plt.axis('off')  # Turn off the axis
    sub_ax = plt.axes([0.9, 0.55, 0.02, 0.3]) 
    plt.subplots_adjust(left=0.05)

    plt.colorbar(plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=max(flows))),
                 orientation='vertical', cax=sub_ax)
    
    plt.savefig(filename)

    #plt.show()


'''
visualize_shortest_path: A function to visualize the shortest path from a pair of nodes(A to B)
@ Input:
G: A grid network
pos: Position of nodes for visualization
shortest_paths: A dictionary where each key represents a trip (origin, destination) and the corresponding value is a list of nodes that represent
the shortest path going from origin to destination node
A: Node ID where we start from
B: Node ID where we end at
labels: Node labels as IDs 

@ Output:
None. (Save the plot as: shortestpath_A_B.png)

'''
def visualize_shortest_path(G, pos, shortest_paths, A, B, labels):
    # Convert the node labels to grid positions
    A_pos = next((k for k, v in labels.items() if v == A), None)
    B_pos = next((k for k, v in labels.items() if v == B), None)
    
    # Find the shortest path between A and B
    path = shortest_paths.get((A_pos, B_pos), None)
    
    if not path:
        print(f"No path found between {A} and {B}.")
        return
    # Calculate total travel time
    total_time = sum(G[path[i]][path[i + 1]]['travel_time'] for i in range(len(path) - 1))
    plt.figure(figsize=(5,5))
    # Draw the full network
    nx.draw(G, pos, node_color='lightgrey', edge_color='lightgrey', with_labels=False)
    
    # Highlight the shortest path
    path_edges = list(zip(path, path[1:]))
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='lightblue', label=labels)
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='black', width=2)
    
    # Draw the labels for the nodes on the path
    path_labels = {n: labels[n] for n in path}
    nx.draw_networkx_labels(G, pos, labels=path_labels)
    
    # Display the start and end nodes more prominently
    nx.draw_networkx_nodes(G, pos, nodelist=[A_pos, B_pos], node_color='yellow', node_size=300)
    
    plt.title(f"Shortest Path from Node {A} to Node {B} with Total Travel Time: {total_time:.2f} min")
    plt.axis('off')
    fn = "shortestpath_" + str(A) + "_" + str(B) + ".png"
    plt.savefig(fn)
    #plt.show()