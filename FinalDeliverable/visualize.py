import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from scipy.sparse import coo_matrix
import itertools
from itertools import islice
import os 

def visualize_network(G, N, pos, labels, V_f, filename = None):
    
    # Visualize the grid network
    edgecolors = ['orange' if G[u][v]['speed_limit'] < V_f else 'green' for u, v in G.edges()]
    speed_limits = [G[u][v]['speed_limit'] for u, v in G.edges()]
    plt.figure(figsize=(6,6))
    plt.title(f'Grid Network ({N} by {N}) with CBD concept')
    nx.draw(G, pos, labels=labels, node_size=750, node_color='lightblue', edge_color = edgecolors,
            with_labels=True, font_weight=500, connectionstyle ='arc3, rad=0.1', width=2)
    
    plt.axis('off')  # Turn off the axis
    sub_ax = plt.axes([0.9, 0.55, 0.02, 0.3]) 
    plt.subplots_adjust(left=0.05)

    # Create a normalizer that will map travel times to the [0, 1] range for the colormap
    norm = mcolors.Normalize(vmin=min(speed_limits), vmax=max(speed_limits))
    # Define the color points for the custom colormap
    colors = ['orange', 'green']

    # Create a colormap using LinearSegmentedColormap
    cmap = mcolors.LinearSegmentedColormap.from_list('orange_to_green', colors, N=256)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    colorbar = plt.colorbar(sm, orientation='vertical', cax=sub_ax)
    colorbar.set_label("Speed Limit (km/hr)")

    # Saving the file dynamically
    if filename:
        # Ensure the directory exists, create it if it doesn't
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)

    plt.show(block=False)




def visualize_traveltime(G, pos, labels, total_time, variable='responsive_travel_time', filename = None):
    # Determine the range of travel times
    travel_times = [d[variable] for (u, v, d) in G.edges(data=True)]
    ff_travel_times = [d['travel_time'] for (u, v, d) in G.edges(data=True)]
    min_tt, max_tt = min(travel_times), max(ff_travel_times)*(1.15)

    # Create a normalizer that will map travel times to the [0, 1] range for the colormap
    norm = mcolors.Normalize(vmin=min_tt, vmax=max_tt)
    plt.figure(figsize=(6,6))
    nx.draw_networkx_nodes(G, pos, node_size=750, node_color='lightblue', label=labels)
    
    # Draw edges with distinct colors and arrows for each direction
    for (u, v, d) in G.edges(data=True):
        tt = d[variable]
        edge_color = plt.cm.rainbow(norm(tt))
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=2, edge_color=edge_color, connectionstyle='arc3, rad=0.1')

    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12)
    plt.title("Visualization of Travel Time (Total Travel time = %d mins)" % (total_time))

    plt.axis('off')  # Turn off the axis
    sub_ax = plt.axes([0.9, 0.55, 0.02, 0.3]) 
    plt.subplots_adjust(left=0.05)

    sm = plt.cm.ScalarMappable(cmap='rainbow', norm=norm)
    colorbar = plt.colorbar(sm, orientation='vertical', cax=sub_ax)
    colorbar.set_label('Travel Time (min)')

    # Saving the file dynamically
    if filename:
        # Ensure the directory exists, create it if it doesn't
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)

    plt.show(block=False)



def visualize_flow(G, pos, labels, filename = None):
    # Determine the range of travel times
    flows = [d['flow'] for (u, v, d) in G.edges(data=True)]
    capacities = [d['capacity'] for (u, v, d) in G.edges(data=True)]
    min_f, max_f = 0, max(capacities)

    # Create a normalizer that will map travel times to the [0, 1] range for the colormap
    norm = mcolors.Normalize(vmin=min_f, vmax=max_f)
    plt.figure(figsize=(6,6))
    nx.draw_networkx_nodes(G, pos, node_size=750, node_color='lightblue', label=labels)
    
    # Draw edges with distinct colors and arrows for each direction
    for (u, v, d) in G.edges(data=True):
        flow = d['flow']
        edge_color = plt.cm.Blues(norm(flow))
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=2, edge_color=edge_color, connectionstyle='arc3, rad=0.1')

    # Draw labels
    nx.draw_networkx_labels(G, pos, labels, font_size=12)
    plt.title("Visualization of Traffic Flow (%d Total Flows)" % (sum(flows)))
  
    plt.axis('off')  # Turn off the axis
    sub_ax = plt.axes([0.85, 0.55, 0.02, 0.3]) 
    plt.subplots_adjust(right=0.80, left=0.01)
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=norm)
    cb = plt.colorbar(sm, orientation='vertical', cax=sub_ax)
    cb.set_label("Number of Vehicles")
    # Saving the file dynamically
    if filename:
        # Ensure the directory exists, create it if it doesn't
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)

    plt.show(block=False)



def visualize_shortest_path(G, pos, shortest_paths, A_pos, B_pos, labels, filename = None):
    # Convert the node labels to grid positions
    #A_pos = next((k for k, v in labels.items() if v == A), None)
    #B_pos = next((k for k, v in labels.items() if v == B), None)
    
    # Find the shortest path between A and B
    path = shortest_paths.get((A_pos, B_pos), None)
    
    if not path:
        print(f"No path found between {A_pos} and {B_pos}.")
        return
    # Calculate total travel time
    total_time = 0.0
    for i in range(len(path) - 1):
        total_time += G[path[i]][path[i + 1]]['responsive_travel_time']
    plt.figure(figsize=(6,6))
    plt.title(f"Shortest Path from Node {A_pos} \n to Node {B_pos} with Total Travel Time: {total_time:.2f} min")

    # Draw the full network
    nx.draw(G, pos, node_color='lightgrey', edge_color='lightgrey', with_labels=False)
    
    # Highlight the shortest path
    path_edges = list(zip(path, path[1:]))
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='lightblue', label=labels, node_size=750)
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='black', width=2)
    
    # Draw the labels for the nodes on the path
    path_labels = {n: labels[n] for n in path}
    nx.draw_networkx_labels(G, pos, labels=path_labels)
    
    # Display the start and end nodes more prominently
    nx.draw_networkx_nodes(G, pos, nodelist=[A_pos, B_pos], node_color='yellow', node_size=750)
    

    # Saving the file dynamically
    if filename:
        # Ensure the directory exists, create it if it doesn't
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)

    plt.show(block=False)



def visualize_flood(G, pos, labels, flood_schedule, flood_impacts, iteration, filename = None):
    flood_info = flood_schedule.get(iteration, None)
    # Determine the range of travel times
    flows = [d['flow'] for (u, v, d) in G.edges(data=True)]
    if flood_info:
        capacities = []
        for (u, v, d) in G.edges(data=True):
            if (u,v) in flood_info['edges']:
                d['flood_capacity'] = d['capacity']*flood_impacts[flood_info['level']]
            else:
                d['flood_capacity'] = d['capacity']

            capacities.append(d['flood_capacity'])
    else:
        capacities = [d['capacity'] for (u, v, d) in G.edges(data=True)]
        for (u,v,d) in G.edges(data=True):
            d['flood_capacity'] = d['capacity']
    min_f, max_f = 0, max(capacities)

    # Create a normalizer that will map travel times to the [0, 1] range for the colormap
    norm = mcolors.Normalize(vmin=min_f, vmax=max_f)
    plt.figure(figsize=(6,6))
    nx.draw_networkx_nodes(G, pos, node_size=750, node_color='lightblue', label=labels)
    
    # Draw edges with distinct colors and arrows for each direction
    for (u, v, d) in G.edges(data=True):
        flow = d['flood_capacity']
        edge_color = plt.cm.rainbow(norm(flow))
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=2, edge_color=edge_color, connectionstyle='arc3, rad=0.1')

    # Draw labels
    nx.draw_networkx_labels(G, pos, labels, font_size=12)
    plt.title("Visualization of Link Capacities under Flash Flood")
  
    plt.axis('off')  # Turn off the axis
    sub_ax = plt.axes([0.85, 0.55, 0.02, 0.3]) 
    plt.subplots_adjust(right=0.80, left=0.01)
    sm = plt.cm.ScalarMappable(cmap='rainbow', norm=norm)
    cb = plt.colorbar(sm, orientation='vertical', cax=sub_ax)
    cb.set_label("Number of Vehicles")
    # Saving the file dynamically
    if filename:
        # Ensure the directory exists, create it if it doesn't
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)

    plt.show(block=False)
 
