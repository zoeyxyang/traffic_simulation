
from functions import *
from plot import *
import argparse



if __name__=='__main__':
    #Read the arguments from user's input
    parser = argparse.ArgumentParser()
    parser.add_argument('--gridsize', '-N', type= int, help='Grid size of the grid network')
    parser.add_argument('--link_length', '-L',  type= float, help='Link Length of the grid network')
    parser.add_argument('--speed_limit', '-V',  type= float, help='Speed limit of the grid network')
    parser.add_argument('--veh_length', '-VL',  type= float, help='Vehicle length of the grid network')
    parser.add_argument('--lane_per_dir', '-LPD',  type= int, help='Lane per direction of the grid network')
    parser.add_argument('--reactiontime', '-R',  type= float, help='Reaction time of the grid network')
    parser.add_argument('--ODmethod', '-OD',  type= str, help='Method to generate OD matrix: Type random or CBD')
    parser.add_argument('--density_level', '-D',  type= float, help='Coefficient for density control (0~1)')

    args = parser.parse_args()

    #Define parameters from user inputs for the grid network
    N, L, V_f, veh_len, lane_per_dir, reaction_time, density_level = args.gridsize, args.link_length, args.speed_limit, args.veh_length,args.lane_per_dir, args.reactiontime, args.density_level

    # Create a grid network
    G, pos, labels = create_grid_network(N, L, V_f,veh_len, lane_per_dir, reaction_time)
    
    print("Grid network visualized. Check network_visualization.png")
    visualize_network(G, N, pos, labels, V_f, filename = '00 Figures/network_visualization.png')
    
    # Generate an OD matrix
    if args.ODmethod =='random':
       od_matrix = generate_random_od_matrix(G, N, density_level).toarray()
    elif args.ODmethod =='CBD':
       od_matrix = generate_CBD_od_matrix(G,N, density_level, 2).toarray()
    else:
        raise NameError("OD method should be selected between random or CBD")


    print("calculating the unoptimized flow using the shortest path algorithm")
    # Decide paths to take (Here we are using shortest path) 
    shortest_paths = find_shortest_paths(G, od_matrix)
    # Visualize the network flow based on the paths
    visualize_flow(G, shortest_paths, od_matrix,  N, pos, labels, filename = '00 Figures/UnoptimizedFlow.png')
    
    
    print("calculating the optimized flow using the simulated annealing algorithm")
    # Optimize the paths
    optimal_paths = find_optimal_paths(G, od_matrix, N, L, veh_len, max_iter=1)
    # Visualize the network flow based on the paths
    visualize_flow(G, optimal_paths, od_matrix,  N, pos, labels, filename='00 Figures/OptimizedFlow.png')
    
    
    print("adding the flood event")
    print("recalculating the optimized flow using the simulated annealing algorithm")
    # Simulate the flooding event
    flooded_links = [((0, 0), (0, 1)), ((0, 0), (1, 0)), ((0, 1), (0, 0)), ((0, 1), (0, 2)), ((0, 1), (1, 1)), ((0, 2), (0, 1)), ((0, 2), (0, 3)), ((0, 2), (1, 2)), ((0, 3), (0, 2))]
    G, _, _ = update_grid_network_flood(N,G, 10, 13, flooded_links)
    optimal_paths = find_optimal_paths(G, od_matrix, N, L, veh_len, max_iter=1)
    # Visualize the network flow based on the paths
    visualize_flow(G, optimal_paths, od_matrix,  N, pos, labels, filename='00 Figures/Flood_OptimizedFlow.png')
    

    #Visualize one shortest path between node 22 and node 55
    #visualize_shortest_path(G, pos, shortest_paths, 22, 55, labels)

