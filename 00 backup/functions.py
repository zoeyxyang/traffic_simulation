import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx
from scipy.sparse import coo_matrix
import random
import math


'''
create_grid_network: A function to create a grid network with N by N nodes

@ Input: 
N: Grid size
L: Link length in km
V_f: Speed limit in km/h
veh_len: vehicle length in km 
lane_per_dir: lane per direction
reaction_time: driver reaction time in sec

@ Output:
G: a N by N graph with edge attributes such as: speed_limit, travel time, length, capacity
pos: Position of nodes for visualization
labels: Node labels as IDs 
'''
def create_grid_network(N, L, V_f, veh_len, lane_per_dir, reaction_time):
    G = nx.grid_2d_graph(N, N, create_using=nx.DiGraph())
    pos = dict((n, n) for n in G.nodes())  # Position of nodes for visualization
    labels = dict(((i, j), (j)*N + i + 1) for i, j in G.nodes())  # Node labels as IDs

    # Assign attributes to all edges
    for (u, v, d) in G.edges(data=True):
        d['length'] = L
        
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
        G.add_edge(v, u, length=L, speed_limit=d['speed_limit'], travel_time = d['travel_time'], capacity=d['capacity'],responsive_travel_time = d['responsive_travel_time'])

    return G, pos, labels

'''
update_grid_network: a function to update the grid network G based on the flood situation
@ Input: 
N: Grid size
G: grid network
flood_A: flood parameter to model the flood (affects the capacity of a link), larger flood_A decrease the capacity 
flood_B: flood parameter to model the flood (affects the capacity of a link), larger flood_B decrease the capacity
affected_links: a list of edges that are affected by the flood

@ Output:
G: a N by N graph with the travel time updated based on the flood parameters
pos: Position of nodes for visualization
labels: Node labels as IDs 
'''

def update_grid_network_flood(N, G, flood_A, flood_B, affected_links):
    pos = dict((n, n) for n in G.nodes())  # Position of nodes for visualization
    labels = dict(((i, j), (j)*N + i + 1) for i, j in G.nodes())  # Node labels as IDs
    for (u, v, d) in G.edges(data=True):
        if (u,v) in affected_links: 
            # update the capacity of the links that are affected by the flood
            update_cap = d['capacity']/flood_A - flood_B
            if update_cap < 0:
                update_cap = 1e-99
            d['capacity'] = update_cap
    return G, pos, labels


'''
akjdfl;akdj;lfakjdf;
'''
def total_network_capacity(G):
    total_capacity = 0
    for (u, v, d) in G.edges(data=True):
        total_capacity += d['capacity']
    return total_capacity



'''
generate_random_od_matrix: A function to generate a random OD matrix using poisson distribution
@ Input: 
N: Grid size of the grid network
scale: A parameter (expected number of events occuring during the time interval) in the poisson distribution 

@ Output:
coo_matrix: the generated OD matrix
'''

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





'''
generate_CBD_od_matrix: A function to generate a realistic OD matrix considering CBD area
@ Input: 
N: Grid size of the grid network
###########################Put details

@ Output:
coo_matrix: the generated OD matrix
'''

def generate_CBD_od_matrix(G, N, density_level, CBD_weight=5):
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

    return coo_matrix(od_matrix)






         
'''
find_shortest_paths: Compute the shortest path using Dijkstra's algorithm for all OD pairs
@ Input: 
G: A grid network
od_matrix: An OD matix we defined

@ Output:
shortest_paths: A dictionary where each key represents a trip (origin, destination) and the corresponding value is a list of nodes that represent
the shortest path going from origin to destination node
'''


# Compute the shortest path using Dijkstra's algorithm for all OD pairs
def find_shortest_paths(G, od_matrix):
    shortest_paths = {}
    for i, origin in enumerate(G.nodes()):
        for j, destination in enumerate(G.nodes()):
            if i != j:  # No path to itself
                # Find the shortest path
                path_length, path = nx.single_source_dijkstra(G, origin, destination, weight='travel_time')
                shortest_paths[(origin, destination)] = path
    return shortest_paths




'''
find_optimal_paths: Compute the optimal path based on an objective function
@ Input: 
G: A grid network
od_matrix: An OD matix we defined
N: Grid size
L: Link length in km
veh_len: vehicle length in km 
max_iter: maximum iterations in the simulated annealing


@ Output:
optimal_paths: A dictionary where each key represents a trip (origin, destination) and the corresponding value is a list of nodes that represent
the optimal path going from origin to destination node
'''
def find_optimal_paths(G, od_matrix, N, L, veh_len, max_iter = 100):
    optimal_paths = {}

    # Compute maximum flow (Capacity)
    #C = int(L / veh_len)

    # Compute shortest paths
    optimal_paths = find_shortest_paths(G, od_matrix)
    best_paths = optimal_paths
    
    best_total = 1e99
    ##current_total = total_travel_time(G, optimal_paths, od_matrix, N, L, C)
    current_total = total_travel_time(G, optimal_paths, od_matrix, N, L)
    thresh = 1000000.0

    # Simulated annealing parameters
    initial_temperature = 1e4
    final_temperature = 0.1
    cooling_rate = 0.99999
    current_temperature = initial_temperature
    current_prob = 0.0 

    iter = 0
    i = 0
    while best_total > thresh and iter < max_iter:
        # Calculate the new optimal paths based on demanded travel time 
        
        for (o,d), path in optimal_paths.items():
            # Choose two random indices from the list
            index1, index2 = random.sample(range(len(path)), 2)

            # Sort the indices from greatest to least
            if index1 > index2:
                index1, index2 = index2, index1

            # Compute the path before
            prev_path = path[:index1]
            post_path = path[index2+1:]

            # Compute the path after
            int_path = random_path(G, path[index1], path[index2])
            new_path = prev_path + int_path + post_path

            optimal_paths[(o,d)] = new_path
            # Find the new total travel time 
            ##tot_time = total_travel_time(G,optimal_paths, od_matrix, N, L, C)
            tot_time = total_travel_time(G, optimal_paths, od_matrix, N, L)
            
            delta_time = tot_time - current_total
            if delta_time < 0:
                if tot_time < best_total:
                    best_total = tot_time
                    best_paths = optimal_paths
                current_total = tot_time 
            elif random.random() < math.exp(-delta_time / current_temperature):
                current_total = tot_time
                
            else:
                optimal_paths[(o,d)] = path
                current_total = tot_time
                
            if delta_time >=0:
                current_prob = math.exp(-delta_time / current_temperature)

            i += 1
            print("Current: %.3f Best: %.3f Iteration %d (%.2f %% Finished) P=%.3f T=%.3f" % (current_total, best_total, iter, 100*i/len(optimal_paths) / max_iter, current_prob, current_temperature))
        
            current_temperature *= cooling_rate
        iter += 1

    return best_paths


'''
travel_time: Compute the travel time of a link (between two nodes:origin and destination)
@ Input: 
G: A grid network
origin: origin node
destination: destination node
V: flow
C: capacity
L: Link length

@ Output:
time: travel time
'''

def travel_time(G, origin, destination, V, C, L):
    # Find free-flow travel time 
    T0 = L / G[origin][destination]['speed_limit'] * 60
    # Calibration Parameters
    alpha = 0.5
    beta = 5
    # Use BPR formula
    time = T0*(1 + alpha*(V/C)**beta)
    return time


'''
total_travel_time: Compute the travel time of all the paths in the provided path 
@ Input: 
G: A grid network
optimal_paths: Paths provided to calculate the total travel time
od_matrix: od matrix
N: Grid size
L: Link length

@ Output:
time: total travel time of all the paths provided
'''

def total_travel_time(G, optimal_paths, od_matrix, N, L):
    # Set demanded flow to zero
    for (u, v, d) in G.edges(data=True):
        d['demanded_flow'] = 0

    # Compute demand for all links
    for (o,d), path in optimal_paths.items():
        demand = od_matrix[o[0] * N + o[1], d[0] * N + d[1]]
        for i in range(len(path) - 1):
            G[path[i]][path[i+1]]['demanded_flow'] += demand
            
    # Compute the demanded travel time 
    total_time = 0.0 
    for (u, v, d) in G.edges(data=True):
        ## Capcity can be modified with flood condition in the update_grid_network_flood function
        time = travel_time(G, u, v, d['demanded_flow'], d['capacity'], L)
        ## update the travel time of the link on G
        total_time += d['demanded_flow'] * time
        d['demanded_travel_time'] = time
    
    return total_time




def random_path(G, start, end):
    if start not in G or end not in G:
        raise ValueError("Start or end node not in graph")

    path = [start]
    current = start
    niter = 1
    iter = 0
    while iter <= niter and current != end:
        neighbors = list(G.neighbors(current))
        if len(neighbors) == 0:
            # Dead-end reached, restart or implement a backtracking strategy
            return random_path(G, start, end)

        # Exclude the node we're coming from to avoid immediate backtracking
        if len(path) > 1:
            neighbors = [n for n in neighbors if n != path[-2]]
        
        current = random.choice(neighbors)
        path.append(current)
    
        iter += 1
    
    if current != end:
        path_length, rest_of_path = nx.single_source_dijkstra(G, current, end, weight='demanded_travel_time')
    else:
        rest_of_path = [current]
    return path + rest_of_path[1:]
