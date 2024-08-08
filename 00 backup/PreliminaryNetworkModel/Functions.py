import networkx as nx
import numpy as np
import random 
import matplotlib.pyplot as plt

def step(G, dt):
    tot_flow = 0.0
    for node1, node2 in G.edges():
        x = G[node1][node2]['vehicles']
        xmax = G[node1][node2]['max_vehicles']
        if x < xmax and x >= 0:
            tot_flow += abs(dXdt(G, node1, node2, dt))
            x = x + dXdt(G, node1, node2, dt) * dt
        
        G[node1][node2]['vehicles'] = x
        G.graph['time'] += dt
        G.graph['total_flow'].append(abs(tot_flow))

def dXdt(G, node1, node2, dt):
    V_in = Vin(G, node1, node2, dt)
    V_out = Vout(G, node1, node2, dt)
    G[node1][node2]['Vin'].append(V_in)
    G[node1][node2]['Vout'].append(V_out)
    return V_in - V_out

def Vin(G, node1, node2, dt):
    # Number of vehicles and max number of vehicles
    x = G[node1][node2]['vehicles']
    xmax = G[node1][node2]['max_vehicles']
    psi = G[node1][node2]['inflow']
    if x < xmax:
        V = psi
        for edge in G.edges():
            i = G[edge[0]][edge[1]]['index']
            j = G[node1][node2]['index']

            Vmax = G.graph['Vmax'][i,j]
            p = G.graph['P'][i,j]

            Vout_i = G[edge[0]][edge[1]]['Vout'][-1]
            V += min([Vmax, Vout_i*p])
    else:
        V = 0
    
    return V

def Vout(G, node1, node2, dt):
    V = 0
    for edge in G.edges():
        j = G[edge[0]][edge[1]]['index']
        i = G[node1][node2]['index']

        Vmax = G.graph['Vmax'][i,j]
        p = G.graph['P'][i,j]

        tt = travelTime(G, node1, node2, G[node1][node2]['vehicles'])

        # Converting travel time to the index of an array
        array_ind = round(tt / dt)
        if array_ind >= len(G[node1][node2]['Vin']):
            Vin = 0
        else:
            Vin = G[node1][node2]['Vin'][-array_ind]

        V += max([Vmax, Vin * p])
 
    
    return V

def travelTime(G, node1, node2, x):
    T0 = G[node1][node2]['length'] / G[node1][node2]['speed']
    Tv = T0*(1+G.graph['alpha'] * (1 + x / G[node1][node2]['max_vehicles']) ** G.graph['beta'])
    return Tv

def setProbabilityMatrix(G):
    P = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ])

    G.graph['P'] = P
    return G

def setMaxFlowMatrix(G):
    Vmax = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ]) * 15

    G.graph['Vmax'] = Vmax
    return G



def getSimpleNetwork():
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes with specific coordinates (x, y)
    node_coordinates = {
        'A': (0, 0),
        'B': (1, 1),
        'C': (2, 0),
        'D': (0, 2),
        'E': (2, 2)
    }
    G.add_nodes_from(node_coordinates.keys())

    G.graph['node_coordinates'] = node_coordinates

    # Add alpha and beta 
    G.graph['alpha'] = 0.5
    G.graph['beta'] = 1.1

    G.graph['time'] = 0.0
    G.graph['total_flow'] = []

    # Add Necessary Info 
    edges = [('A', 'B'), ('B', 'C'), ('C', 'A'), ('D', 'B'), ('E', 'D'), ('E', 'C')]
    speeds = [15, 25, 35, 25, 50, 35] # Speed in miles per hour
    lengths = [0.1, 0.5, 0.3, 0.4, 0.14, 0.25] # Lengths in miles
    inflows = [15, -3, 1 , 0, 1, -1 ] # Boundary Conditions of the node
    vehicles = [10, 15, 20, 21, 23, 5] # Number of vehicles on each link
    l_avg = 25.0 / 5280.0 # The length that the average car takes up (miles)
    max_vehicles = [round(length / l_avg) for length in lengths] # Maximum number of vehicles

    # Add edges to graph and assign random 'speed' attribute
    for i in range(len(edges)):
        G.add_edge(*edges[i], 
                   speed=speeds[i], 
                   length=lengths[i], 
                   inflow=inflows[i],
                   vehicles=vehicles[i],max_vehicles=max_vehicles[i],
                   Vin = [inflows[i] if inflows[i] > 0 else 0],
                   Vout = [inflows[i] if inflows[i] < 0 else 0],
                   index = i)

    return G

def drawNetwork(G, attribute, title_text=None, label_edges=False, display_flow=False):
    # Initialize plot
    if display_flow:
        fig, [ax, ax1] = plt.subplots(1, 2)
        plt.subplot(1,2,1)
    else:
        fig, ax = plt.subplots()
    
    

    # Draw graph
    pos = G.graph['node_coordinates']  # positions for all nodes

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', ax=ax)

    # Draw edges with color based on 'speed' attribute
    edges = G.edges()
    colors = [G[u][v][attribute] for u, v in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=colors, edge_cmap=plt.cm.RdYlGn_r, arrowstyle='->', arrowsize=25, width=2, ax=ax)

    # Add labels
    nx.draw_networkx_labels(G, pos, ax=ax)
    if label_edges:
        edge_labels = nx.get_edge_attributes(G, attribute)
        for key in edge_labels.keys():
            if type(edge_labels[key] == float):
                edge_labels[key] = round(edge_labels[key], 2)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

    # Add color bar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(vmin=min(nx.get_edge_attributes(G, attribute).values()), vmax=max(nx.get_edge_attributes(G, attribute).values())))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=attribute)
    
    # Display the title 
    if title_text:
        plt.title(title_text)
    # Show plot
    #plt.show()

    if display_flow:
        plt.subplot(1,2,2)
        plt.plot(np.linspace(0, G.graph['time'], len(G.graph['total_flow']), G.graph['total_flow']))
        plt.grid()
