import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from scipy.sparse import coo_matrix
import itertools
from itertools import islice
import os

# Function to create a grid network
def create_grid_network(N, L, V_f, veh_len, lane_per_dir, reaction_time):
    G = nx.grid_2d_graph(N, N, create_using=nx.DiGraph())
    pos = dict((n, n) for n in G.nodes())  # Position of nodes for visualization
    labels = dict(((i, j), '(%d, %d)' % (i,j) ) for i, j in G.nodes())  # Node labels as IDs

    # Assign attributes to all edges
    for (u, v, d) in G.edges(data=True):
        d['length'] = L
        d['flow'] = 0
        # If the edge is in the center of the grid, assign half the speed limit
        if (u[0] >= (N+1) // 4 and u[0] < 3 *(N+1) // 4) and (v[1] >= (N+1) // 4 and v[1] < 3 *(N+1) // 4) and (u[1] >= (N+1) // 4 and u[1] < 3 *(N+1) // 4) and (v[0] >= (N+1) // 4 and v[0] < 3 *(N+1) // 4):
            d['speed_limit'] = V_f/2
        else:
            d['speed_limit'] = V_f
        
        # Add attribute of free-flow travel time 
        d['travel_time'] = d['length'] / d['speed_limit'] * 60  # Convert hours to minutes
        
        # Add attribute of capacity 
        d['capacity'] = d['speed_limit'] * lane_per_dir / (d['speed_limit'] *reaction_time/3600 + veh_len)
        
        # Add attribute of responsive travel time
        alpha, beta = 0.15, 4  # Example values, adjust based on empirical data
        V = 0  # Current traffic volume on the road (to be updated during simulation)
        d['responsive_travel_time'] = d['travel_time'] * (1 + alpha * (V / d['capacity']) ** beta)
        
        # Since the graph is bidirectional, add the same attributes for the reverse direction
        G.add_edge(v, u, length=L, flow = 0, speed_limit=d['speed_limit'], travel_time = d['travel_time'], capacity=d['capacity'],responsive_travel_time = d['responsive_travel_time'])

    return G, pos, labels




def total_network_capacity(G):
    total_capacity = 0
    for (u, v, d) in G.edges(data=True):
        total_capacity += d['capacity']
    return total_capacity


# Function to generate a random OD matrix
def generate_random_od_matrix(G,N, density_level):
# Calculate the maximum possible number of trips considering the capacity
    max_trips = total_network_capacity(G)

    # Calculate the total number of trips based on the density level
    total_trips = density_level * max_trips

    # Initialize the OD matrix
    od_matrix = np.zeros((N**2, N**2))

    # Randomly distribute the total number of trips in the OD matrix
    for _ in range(int(total_trips)):
        origin, destination = np.random.randint(0, N**2, 2)
        while origin == destination:  # Ensure origin is not the same as destination
            origin, destination = np.random.randint(0, N**2, 2)
        od_matrix[origin, destination] += 1

    return coo_matrix(od_matrix)


# Visualize the OD matrix
# plt.figure(figsize=(5, 5))
# plt.matshow(od_matrix, cmap=plt.cm.Blues)
# plt.title('Random OD Matrix')
# plt.colorbar()
# plt.show()
def generate_CBD_od_matrix(G, N, density_level, seed, CBD_weight=5 ):
    np.random.seed(seed)
    def total_network_capacity(G):
        total_capacity = 0
        for (u, v, d) in G.edges(data=True):
            total_capacity += d['capacity']
        return total_capacity

    def is_boundary_node(node, N):
        x, y = divmod(node, N)
        return x < N/2 or x >= N - N/2 or y < N/2 or y >= N - N/2

    def is_CBD_node(node, N):
        x, y = divmod(node, N)
        CBD_start, CBD_end = (N+1) // 4, 3 * (N+1) // 4
        return CBD_start <= x < CBD_end and CBD_start <= y < CBD_end

   # Calculate the maximum possible number of trips considering the capacity
    max_trips = total_network_capacity(G)

    # Calculate the total number of trips based on the density level
    total_trips = density_level * max_trips

    # Initialize the OD matrix
    od_matrix = np.zeros((N**2, N**2))

    # Assign weights to each destination node
    destination_weights = np.array([CBD_weight if is_CBD_node(dest, N) else 1 for dest in range(N**2)])
    destination_weights = destination_weights / destination_weights.sum()  # Normalize weights

    # Distribute trips across the OD matrix
    allocated_trips = 0
    while allocated_trips < total_trips:
        origin = np.random.randint(0, N**2)
        if is_boundary_node(origin, N):
            destination = np.random.choice(N**2, p=destination_weights)
            if origin != destination:
                od_matrix[origin, destination] += 1
                allocated_trips += 1

    return coo_matrix(od_matrix).toarray()


def initialize_traffic_assignment(G, od_matrix, N, k=3):
    # Initialize all edge flows to zero
    nx.set_edge_attributes(G, 0, 'flow')


    # Function to find k-shortest paths
    def find_k_shortest_paths(G, source, target, k):
        return list(islice(nx.shortest_simple_paths(G, source, target, weight='responsive_travel_time'), k))

    # Function to distribute flows across paths
    def distribute_flows(G, paths, flow):
        for path in paths:
            # Check if there is enough capacity on the path
            path_capacity = min([G[u][v]['capacity'] - G[u][v]['flow'] for u, v in zip(path[:-1], path[1:])])
            if path_capacity <= 0:
                continue  # Skip this path if no capacity is left

            # Distribute flow considering the available capacity
            distributed_flow = min(flow, path_capacity)
            for i in range(len(path) - 1):
                G[path[i]][path[i+1]]['flow'] += distributed_flow
            flow -= distributed_flow

            if flow <= 0:
                break
    # Calculate the flow on each edge based on the OD matrix and update shortest paths
    shortest_paths = {}
    for origin, destination in itertools.product(G.nodes(), repeat=2):
        if origin != destination:
            origin_index = node_to_index(origin, N)
            destination_index = node_to_index(destination, N)
            flow = od_matrix[origin_index, destination_index]
            if flow > 0:
                paths = find_k_shortest_paths(G, origin, destination, k)
                distribute_flows(G,paths, flow)
                # Update the shortest path
                shortest_paths[(origin, destination)] = min(paths, key=lambda p: sum(G[p[i]][p[i+1]]['responsive_travel_time'] for i in range(len(p)-1)))

    # Update the responsive_travel_time for each edge
    for u, v, d in G.edges(data=True):
        V = d['flow']
        C = d['capacity']
        alpha, beta = 0.15, 4  # Example values
        d['responsive_travel_time'] = d['travel_time'] * (1 + alpha * (V / C) ** beta)

    return shortest_paths





def node_to_index(node, N):
    return node[0] * N + node[1]


def update_flow_and_TT(G, shortest_paths, od_matrix,N):
    # Convert node label (tuple) to index in the OD matrix
    nx.set_edge_attributes(G, 0, 'flow')

    # Update flow based on the current shortest paths
    for (origin_label, destination_label), path in shortest_paths.items():
        if len(path) >= 1:  
            origin_index = node_to_index(origin_label, N)
            destination_index = node_to_index(destination_label, N)
            trip_count = od_matrix[origin_index][destination_index]

            # Reduce flow on the first edge and add to subsequent edges
            # first_edge = (path[0], path[1])
            # G[first_edge[0]][first_edge[1]]['flow'] -= trip_count
            for i in range(0, len(path) - 1):
                G[path[i]][path[i + 1]]['flow'] += trip_count


    # Update responsive travel time for each edge
    for u, v, d in G.edges(data=True):
        V = d['flow']
        C = d['capacity']
        alpha, beta = 0.15, 4  # Example values
        d['responsive_travel_time'] = d['travel_time'] * (1 + alpha * (V / C) ** beta)
      

def update_traffic_assignment(G, od_matrix, N, k=3):
    nx.set_edge_attributes(G, 0, 'flow')  # Reset all edge flows to zero

    def find_k_shortest_paths(G, source, target, k):
        return list(islice(nx.shortest_simple_paths(G, source, target, weight='responsive_travel_time'), k))

    def distribute_flows(G, paths, flow):
        for path in paths:
            path_capacity = min([G[u][v]['capacity'] - G[u][v]['flow'] for u, v in zip(path[:-1], path[1:])])
            if path_capacity <= 0:
                continue
            distributed_flow = min(flow, path_capacity)
            for i in range(1, len(path) - 1):
                G[path[i]][path[i+1]]['flow'] += distributed_flow
            flow -= distributed_flow
            if flow <= 0:
                break

    # Update flows and shortest paths
    shortest_paths = {}
    for origin, destination in itertools.product(G.nodes(), repeat=2):
        if origin != destination:
            origin_index = node_to_index(origin, N)
            destination_index = node_to_index(destination, N)
            flow = od_matrix[origin_index, destination_index]
            if flow > 0:
                paths = find_k_shortest_paths(G, origin, destination, k)
                distribute_flows(G, paths, flow)
                shortest_paths[(origin, destination)] = min(paths, key=lambda p: sum(G[p[i]][p[i+1]]['responsive_travel_time'] for i in range(len(p)-1)))

    # Update travel times based on new flows
    for u, v, d in G.edges(data=True):
        V = d['flow']
        C = d['capacity']
        alpha, beta = 0.15, 4  # Example values
        d['responsive_travel_time'] = d['travel_time'] * (1 + alpha * (V / C) ** beta)

    return shortest_paths

def dynamic_traffic_assignment(G, od_matrix, N, pos,density_level, max_iterations=10, convergence_threshold = 0.01):
    shortest_paths = initialize_traffic_assignment(G, od_matrix, N)
    print("loaded initial traffic")
    for iteration in range(max_iterations):
        print(f"Iteration {iteration}: update OD, flow, shortest paths")
        # I don't think we need to update OD matrix. OD should be fixed
        # Update the flow based on the updated OD matrix
        # update_flow_and_TT(G, shortest_paths, od_matrix,N)
        shortest_paths = update_traffic_assignment(G, od_matrix, N, k=3)
    else:
        print("Max iterations reached before completing all trips.")
    return G, shortest_paths

def update_network_for_flood(G, iteration, flood_schedule, flood_impacts, original_capacities):
    flood_info = flood_schedule.get(iteration, None)

    if flood_info:
        flood_level = flood_info['level']
        affected_edges = flood_info['edges']
        capacity_factor = flood_impacts[flood_level]

        for u, v, d in G.edges(data=True):
            C = d['capacity']
            if affected_edges is None or (u, v) in affected_edges:
                C = original_capacities[u, v] * capacity_factor
            else:
                C = original_capacities[u, v]  # Reset to original capacity

            V = d['flow']
            alpha, beta = 0.15, 4  # Example values
            d['responsive_travel_time'] = C * (1 + alpha * (V / C) ** beta)
            
def dynamic_traffic_assignment_flood(G, od_matrix, N, flood_schedule, flood_impacts, max_iterations=10):
    shortest_paths = initialize_traffic_assignment(G, od_matrix, N)
    print("loaded initial traffic")

    original_capacities = {(u, v): d['capacity'] for u, v, d in G.edges(data=True)}
    
    for iteration in range(max_iterations):
        update_network_for_flood(G, iteration, flood_schedule, flood_impacts, original_capacities)
        print(f"Iteration {iteration}: update OD, flow, shortest paths")
        # Update OD matrix based on current shortest paths
        # # Update the flow based on the updated OD matrix
        # update_flow_and_TT(G, shortest_paths, od_matrix, N)
        shortest_paths = update_traffic_assignment(G, od_matrix, N, k=3)
    else:
        print("Max iterations reached before completing all trips.")
         
    return G, shortest_paths

def get_total_travel_time(G, shortest_paths, od_matrix, N):
    total_travel_time = 0.0

    # Update travel times based on new flows
    for u, v, d in G.edges(data=True):
        
        V = d['flow']
        TT = d['responsive_travel_time']
        total_travel_time += V * TT
        
    return total_travel_time
