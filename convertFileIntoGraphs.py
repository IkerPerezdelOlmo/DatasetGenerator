import pandas as pd
import numpy as np
import torch
import random
import sys




# Define the function to parse the .gr file and extract graphs.
# The graphs will be stored into a list containig all teh edges.
def read_full_adj(file_path, problemPath,  skiprows, usecols):
    adj_lists = []
    for i in range(len(problemPath)):
        adj_list = np.loadtxt(file_path + problemPath[i], dtype=str, skiprows=skiprows, usecols=usecols)

        # Convert the strings in the array to integers
        adj_list = adj_list.astype(int)

        # Convert the resulting array to a list of tuples
        adj_list = list(map(tuple, adj_list))

        adj_list = sorted(adj_list, key=lambda x: x[0])
        adj_lists.append(adj_list)

    return adj_lists
   
# Samples subgraphs of a specific size from a larger graph's adjacency list.
# nodo_kop: The desired number of nodes in each sampled subgraph.
# lagina: The number of subgraphs to attempt to sample.
# adj_list: The adjacency list of the large graph (list of edge tuples).
# desfase: An offset used to vary the starting point for subgraph extraction. 
 

def grafoak_lagindu(desfase, nodo_kop, adj_list, lagina):
    # Normalize 'desfase' to be within the range [0, nodo_kop - 1].
    # This ensures the offset is relative to the subgraph size.
    if desfase // nodo_kop != 0:
        desfase = desfase % nodo_kop

    # 'oinarrizko_indizea' (base_index) determines the starting node ID for the current subgraph window.
    # Assuming 1-based indexing in the input graph file, adjusted by desfase.
    oinarrizko_indizea = 1 + desfase
    # Initialize the tensor where graph is going to be created.
    tensor1 = torch.zeros((nodo_kop, nodo_kop))
    tensor2 = tensor1 # tensor2 will be used as a working tensor for the current subgraph being built.

    grafoa_bukatu_da = False # Flag: True if a complete subgraph (with nodo_kop nodes) has been identified. "Graph is finished"
    hasiera = True # Flag: True if this is the first subgraph being processed in this call. "Beginning"
    adj_list_luzera = len(adj_list) # Total number of edges in the input graph.
    i = 0 # Iterator for the edges in adj_list.

    # Loop through the edges of the large graph as long as there are edges and we still need to sample more subgraphs.
    while i < adj_list_luzera and lagina > 0:
        # When we have used nodo_kop nodes, it means we have created a graph
        if (adj_list[i][0]) % nodo_kop == 0+desfase and adj_list[i][0] > oinarrizko_indizea:
            grafoa_bukatu_da = True

        # If we have completed the first graph, save it in tensor1 and update the neccesary values to continue getting graphs
        if grafoa_bukatu_da and hasiera:
            hasiera = False
            tensor1 = tensor2
            oinarrizko_indizea = adj_list[i][0]
            lagina -= 1

        # if it is not the first graph add it to tensor1, as here we will keep all the graphs
        elif grafoa_bukatu_da:
            if torch.sum(tensor2).item() != 0:
                tensor1 = torch.cat((tensor1, tensor2), dim=0)
                lagina -= 1
            grafoa_bukatu_da = False
            oinarrizko_indizea = adj_list[i][0]

        # Check if teh connection we are adding is suitable for the graph length we want
        if adj_list[i][0] - oinarrizko_indizea> 0 and adj_list[i][1] - oinarrizko_indizea < nodo_kop and  adj_list[i][1] - oinarrizko_indizea > 0:
            # Save the relative graph we are seeing using realtive node names, so as the node names are between [0, nodo_kop-1]
            tensor2[adj_list[i][1]-oinarrizko_indizea, adj_list[i][0]-oinarrizko_indizea] = 1
            tensor2[adj_list[i][0]-oinarrizko_indizea, adj_list[i][1]-oinarrizko_indizea] = 1

        i += 1


    return tensor1, lagina





def main(pathLoad, problemPath, nodo_kop, pathSave, desfase, laginak = [50],  skiprows = 0, usecols = (1,2), problemName = None):

    if problemName is None:
        print("problemaren izena zehaztu behar duzu0")
        raise RuntimeError("problemName adierazi")


    # get the original matrix and keep it in a list where each element are pair of connected nodes.
    adj = read_full_adj(pathLoad, problemPath,  skiprows, usecols)

    for lagina in laginak:
        for i in range(len(nodo_kop)):
            print(nodo_kop[i], lagina)
            desfaseTrain = 0
            desfaseTest = desfase[i] % nodo_kop[i]
            listOfDesfases = [i for i in range(nodo_kop[i])]
            listOfDesfases.remove(desfaseTrain)
            listOfDesfases.remove(desfaseTest)


            laginaTest = lagina
            laginaTrain = lagina


            while laginaTest != 0 or laginaTrain != 0:
                tensorsTrain, laginaTrain = grafoak_lagindu(desfaseTrain, nodo_kop[i], adj[0], laginaTrain)
                tensorsTest, laginaTest = grafoak_lagindu(desfaseTest, nodo_kop[i], adj[0], laginaTest)

                if len(adj) != 1:
                    for j in range(len(adj)-1):
                        tensorsTrain2, laginaTrain = grafoak_lagindu(desfaseTrain, nodo_kop[i], adj[j+1], laginaTrain)
                        tensorsTrain = torch.cat((tensorsTrain, tensorsTrain2), dim=0)
                        tensorsTest2, laginaTest = grafoak_lagindu(desfaseTest, nodo_kop[i], adj[j+1], laginaTest)
                        tensorsTest = torch.cat((tensorsTest, tensorsTest2), dim=0)


                if len(listOfDesfases) > 2:
                    desfaseTrain, desfaseTest =  random.sample(listOfDesfases, 2)
                    listOfDesfases.remove(desfaseTrain)
                    listOfDesfases.remove(desfaseTest)

                print("laginaTrain", laginaTrain)
                print("laginaTest", laginaTest)

            if lagina == 1:
                lagina = 5
                tensorsTest = torch.cat((tensorsTest, tensorsTest, tensorsTest, tensorsTest, tensorsTest), dim=0)
                tensorsTrain = torch.cat((tensorsTrain, tensorsTrain, tensorsTrain, tensorsTrain, tensorsTrain), dim=0)


            np.savetxt(pathSave + '-'.join(problemName.split("-")[:-1])+"_"+str(lagina)+"_N"+str(nodo_kop[i])+"_"+problemName.split("-")[-1], tensorsTrain, delimiter=',')
            np.savetxt(pathSave + '-'.join(problemName.split("-")[:-1])+"_"+str(lagina)+"_N"+str(nodo_kop[i])+"_"+problemName.split("-")[-1]+"_test", tensorsTest, delimiter=',')

    




# Paarmetros:
#   road: carpeta = ... /LOPsortzailea/ROAD/" ;  skiprows= 1 ; usecols= (1,2); problemName = "USA-road_d-NY"
#   DIMACS: carpeta = ... /LOPsortzailea/DIMACS/" ;  skiprows= 0  ; usecols= (0,1) ; problemName = "C-DSJC-p-san-DIMACS"
#   InternetTopology = ... /LOPsortzailea /InternetTopology; skiprows = 5; usecols = (0,1); problemName = "Internet_Topology-skitter"
if __name__ == "__main__":
    #main(pathLoad ='C:/TRABAJO/Iker/MisCosas/trabajo/Practicas/PracticasUniversidad/codigo/LOPfuntzioak/LOPsortzailea/ROAD/', problemPath= ['USA-road-d-NY.gr'], nodo_kop= [2, 10, 40, 100, 200], pathSave = 'C:/TRABAJO/Iker/MisCosas/trabajo/Practicas/PracticasUniversidad/codigo/GraphRNN_Maximum_Independent_Set/csvToGraph_Iker/data_in_csv_2/', desfase = [37,59,67,127,7], laginak = [1], skiprows= 1, usecols= (1,2), problemName = "USA-road_d-NY")
    main(pathLoad ='C:/TRABAJO/Iker/MisCosas/trabajo/Practicas/PracticasUniversidad/codigo/LOPfuntzioak/LOPsortzailea/InternetTopology/', problemPath= ['as-skitter.txt'], nodo_kop= [10, 40, 100], pathSave = 'C:/TRABAJO/Iker/MisCosas/trabajo/Practicas/PracticasUniversidad/codigo/GraphRNN_Maximum_Independent_Set/csvToGraph_Iker/data_in_csv2/', desfase = [37,59,67,127,7], laginak = [15, 50, 150, 500], skiprows= 5, usecols= (0,1), problemName = "Internet_Topology-skitter")
