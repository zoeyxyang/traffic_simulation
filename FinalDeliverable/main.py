from functions import *
from visualize import * 

if __name__ == '__main__':
    
    # System Parameter 
    seed = 42 # random seed
    dynamic_iterations = 12

    # Network Parameter
    N = 6 # Size of grid (N X N)
    L = 0.2 # Length of link (km)
    V_f = 50 # Speed limit (km/h)
    veh_len = 0.005 # Length of vehicle (km) 
    lane_per_dir = 1 # Lanes per direction
    reaction_time = 1.2 # Reaction time (s)
    OD = 'CBD'  # Select between 'random' and 'CBD'
    density_level = 0.3 # Total number of vehicles are controlled by density_level * total sum of capacities in the network
    
    # Flood-related Parameter
    flooding = False
    flood_impacts = {
    'light_rain': 0.95,   # 98% of original capacity
    'moderate_flood': 0.50,  # 50% of original capacity
    'severe_flood': 0.0000000001,    # 0% of original capacity (impassable)
    }
    
    flood_schedule = { # Modify flood schedule. key: the iteration number that certain level of flood will be applied
        4: {'level': 'light_rain', 'edges': []},  # Apply to all edges
        5: {'level': 'moderate_flood', 'edges': [((0,0), (0,1)), ((1,0), (2,0)), ((2,1), (2,2)), ((2,2), (3,2))]},
        6: {'level': 'moderate_flood', 'edges': [((4,1), (4,0)), ((4,0), (4,1)), ((4,1), (4,2)), ((4,2), (4,3))]},
        10: {'level': 'severe_flood', 'edges': [((2,1), (2,2)), ((2,2), (3,2)), ((0,2), (1,2)), ((0,0), (1,0)),((1,0), (0,0))]},
        11: {'level': 'severe_flood', 'edges': [ ((0,2), (1,2)), ((0,0), (1,0)),((1,0), (0,0))]},
    }


    print("Initializing Network...")
    # Creating the graph
    G, pos, labels = create_grid_network(N, L, V_f, veh_len, lane_per_dir, reaction_time)
    # Generating OD Matrix
    if OD =='CBD':
        od_matrix = generate_CBD_od_matrix(G, N, density_level, seed, CBD_weight=5)
    else:
        od_matrix = generate_random_od_matrix(G, N, density_level)
    # Initialize Traffic assignment
    shortest_paths = initialize_traffic_assignment(G, od_matrix, N, k=3)
    # Updating flow and travel time 
    update_flow_and_TT(G, shortest_paths, od_matrix, N)
    # Visualizing shortest paths
    total_time = get_total_travel_time(G, shortest_paths, od_matrix, N)
    # Visualizing travel time and flow
    visualize_traveltime(G, pos, labels, total_time, filename='visualizations/TT_initial.png')
    visualize_flow(G, pos, labels, filename = 'visualizations/Flow_initial.png')
    visualize_shortest_path(G, pos, shortest_paths, (0,0), (5,5), labels, filename = 'visualizations/shortest_path_initial.png')
    
    # Dynamically Updating traffic assignment
    for iteration in range(dynamic_iterations):
        print("Iteration %d" % (iteration + 1))
        if flooding:
            update_network_for_flood(G, iteration, flood_schedule, flood_impacts, {(u, v): d['capacity'] for u, v, d in G.edges(data=True)})
            shortest_paths = update_traffic_assignment(G, od_matrix, N, k=3)
            visualize_flood(G, pos, labels, flood_schedule, flood_impacts, iteration, filename='visualizations/Flood_Capacity_%d.png' % (iteration))
        else:
            shortest_paths = update_traffic_assignment(G, od_matrix, N, k=3)
        # Calculating total time 
        total_time = get_total_travel_time(G, shortest_paths, od_matrix, N)
        # Visualizing travel time 
        visualize_traveltime(G, pos, labels, total_time, filename = 'visualizations/TT_%d.png' % (iteration))
        visualize_flow(G, pos, labels, filename = 'visualizations/Flow_%d.png' % (iteration))
        # Visualizing shortest paths
        visualize_shortest_path(G, pos, shortest_paths, (0,0), (5,5), labels, filename = 'visualizations/shortest_path_%d.png' % (iteration))
        


