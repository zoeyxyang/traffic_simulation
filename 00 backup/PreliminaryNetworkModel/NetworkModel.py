from Functions import *
import networkx as nx
import matplotlib.pyplot as plt
import random

if __name__=='__main__':
    print("Preliminary Network Model")

    # Load a simple network
    G = getSimpleNetwork()

    # Set probability States
    G = setProbabilityMatrix(G)

    # Set Maximum Flow Rates
    G = setMaxFlowMatrix(G)

    # Display the network with a specific attribute
    drawNetwork(G, 'max_vehicles', title_text = "Simple Network Graph", label_edges=True)
    for i in range(100):
        drawNetwork(G, 'vehicles', title_text='Number of Cars per Street', label_edges=True, display_flow=False)
        plt.draw()
        plt.pause(0.5)  # Pause briefly to show the updated plot
        plt.close()
        step(G, 0.05)