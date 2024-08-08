
from functions import *
from plot import *
import argparse



if __name__=='__main__':
   #Define parameters from user inputs for the grid network
   N, L, V_f, veh_len, lane_per_dir, reaction_time = [10, 100, 25, 0.01, 1, 0.1]

   # Create a grid network
   G, pos, labels = create_grid_network(N, L, V_f,veh_len, lane_per_dir, reaction_time)

   # Generate an OD matrix
   od_matrix = generate_random_od_matrix(N, 20).toarray()

   # Decide paths to take (Here we are using shortest path) #TODO: Need to change it to optimize our objective function
   shortest_paths = find_shortest_paths(G, od_matrix)

   # Visualize the network flow based on the paths
   visualize_flow(G, shortest_paths, od_matrix,  N, pos, labels, filename = 'UnoptimizedFlow.png')

   # Optimize the paths
   optimal_paths = find_optimal_paths(G, od_matrix, N, L, veh_len, max_iter=15)

   # Visualize the network flow based on the paths
   visualize_flow(G, optimal_paths, od_matrix,  N, pos, labels, filename='OptimizedFlow.png')
   

   #Visualize one shortest path between node 22 and node 55
   #visualize_shortest_path(G, pos, shortest_paths, 22, 55, labels)

