import torch
import numpy as np
import networkx as nx
import random
import torch.nn.functional as F


# MIS (maximum independent set) optimizazio problemarako laginak sortzen ditu, laginetan agertzen diren balioak banaketa uniforme batetatik sortuak izan dira.
def createInstancesMIS(dimension, probability):
    tensorea = torch.zeros(dimension, dimension)
    for i in range(dimension):
        LOPerrenkada = np.random.rand(dimension - i-1)
        LOPerrenkada = np.where(LOPerrenkada < probability, 0, 1)
        LOPerrenkada = torch.tensor(LOPerrenkada)
        tensorea[i,i+1:dimension]=LOPerrenkada
        tensorea [i+1:dimension, i] = LOPerrenkada
    return tensorea

# distribuzio geometriko bat erabiliko da matrizeko zenbakiak sortzeko. 
# Banaketa geometrikoak funztio exponentzialarekin zerikusia dauka eta 0 inguruko balioek gainontzekoak baino probabilitate handiagoa dute. 
# Pareto hau parametrizatzeko probabilitate bat erabili izan da, zeinak 0 inguruko balioei zenbateko indarra ematen zaien adierazten duen.def createInstancesLOP(dimension, probability):
def createInstancesLOP(dimension, probability):
    tensorea = torch.zeros(dimension, dimension)
    for i in range(dimension):
        LOPerrenkada = np.random.rand(p=probability, size=dimension - i-1) - np.ones((dimension - i-1))
        LOPerrenkada = torch.tensor(LOPerrenkada)
        tensorea[i,i+1:dimension]=LOPerrenkada
        LOPerrenkada = np.random.geometric(p=probability, size=dimension - i-1) - np.ones((dimension - i-1))
        LOPerrenkada = torch.tensor(LOPerrenkada)
        tensorea [i+1:dimension, i] = LOPerrenkada
    return tensorea



# Erdos renyi teknikaren bidez grafoak sortu 
def createInstancesErdosRenyi(dimension, probability):
    G = nx.erdos_renyi_graph(dimension, probability)
    adj = nx.to_numpy_array(G)
    adjacency_tensor = torch.tensor(adj, dtype=torch.float32)

    return adjacency_tensor



# Erdos renyi teknikaren bidez GraphRNN-ko paperrean agertzen den Grid dataseta sortu. 

def generate_community_graph(dimension = 100, probability = 0.3, inter_prob=0.05):
    """
    Generates a single graph with two Erdos-Renyi communities and inter-community edges.
    """
    # Randomly choose total number of nodes
    V = dimension
    V1 = V // 2  # Size of each community

    # Generate two ER graphs
    G1 = nx.erdos_renyi_graph(V1, p=probability)
    G2 = nx.erdos_renyi_graph(V - V1, p=probability)

    # Relabel nodes of the second community to avoid overlap
    mapping = {node: node + V1 for node in G2.nodes()}
    G2 = nx.relabel_nodes(G2, mapping)

    # Combine both communities
    G = nx.compose(G1, G2)

    # Add inter-community edges
    num_inter_edges = int(inter_prob * V)
    possible_inter_edges = [(u, v) for u in G1.nodes() for v in G2.nodes()]
    inter_edges = random.sample(possible_inter_edges, min(num_inter_edges, len(possible_inter_edges)))

    G.add_edges_from(inter_edges)

    adj = nx.to_numpy_array(G)
    adjacency_tensor = torch.tensor(adj, dtype=torch.float32)

    return adjacency_tensor



def createDataset(dimension, quantity, probability, problema, filepath, test):
    if problema == "LOP":
        createInstances = createInstancesLOP
    elif problema == "MIS":
        createInstances = createInstancesMIS
    elif problema == "erdos-renyi":
        createInstances = createInstancesErdosRenyi
    elif problema == "Community":
        createInstances = generate_community_graph

    if isinstance(dimension, str):
        low_bound = int(dimension.rsplit("-",1)[0])
        upper_bound = int(dimension.rsplit("-",1)[1])
        rand = random.randint(low_bound, upper_bound)
        tensor1 = createInstances(rand, probability)

        # hacer que todos los tensores sean de 160x160
        paddings = (0, 160-rand, 0, 160-rand)
        tensor1 = F.pad(tensor1, paddings, "constant", 0)
    else:
        tensor1 = createInstances(dimension, probability)
    # If we only want an instance, due to how the model is coded we will create 5 equal instances.
    if quantity == 1:
        tensor1 = torch.cat((tensor1, tensor1, tensor1, tensor1, tensor1), dim=0)

    for i in range(quantity-1):
        if isinstance(dimension, str):
            low_bound = int(dimension.rsplit("-",1)[0])
            upper_bound = int(dimension.rsplit("-",1)[1])
            rand = random.randint(low_bound, upper_bound)
            tensor2 = createInstances(rand, probability)

            # hacer que todos los tensores sean de 160x160
            paddings = (0, 160-rand, 0, 160-rand)
            tensor2 = F.pad(tensor2, paddings, "constant", 0)
        else:
            tensor2 = createInstances(dimension, probability)

        tensor1 = torch.cat((tensor1, tensor2), dim=0)

    if quantity == 1:
        quantity = 5

    csv_file_path = "random_" + problema +"_"+ str(quantity)+'_M'+str(dimension)+'_P'+str(int(10*probability))
    if test:
        csv_file_path +="_test"
    np.savetxt(filepath + csv_file_path, tensor1, delimiter=',')
    return csv_file_path


def main(dimensions, quantities, probabilities, problem, filepath):
    izenak = []
    for dim in dimensions:
        for quantity in quantities:
            for probability in probabilities:
                name = createDataset(dim, quantity, probability, problem, filepath, test = False)
                izenak.append(name)
                name_test = createDataset(dim, quantity, probability, problem, filepath, test = True)
                izenak.append(name_test)
    print(izenak)

if __name__ == "__main__":
    main(dimensions =[2,10,40,100], quantities = [1] , probabilities = [0.3], problem= "erdos-renyi", filepath = 'C:/TRABAJO/Iker/MisCosas/trabajo/Practicas/PracticasUniversidad/codigo/GraphRNN_Maximum_Independent_Set/csvToGraph_Iker/data_in_csv_2/')