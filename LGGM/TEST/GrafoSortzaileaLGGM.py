import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_undirected
import os

def arrange_data(adj_matrix):
    n_nodes = adj_matrix.shape[0]

    edge_index = adj_matrix.nonzero().t()
    edge_attr = torch.tensor([[0, 1] for _ in range(edge_index.shape[1])])

    edge_index, edge_attr = to_undirected(edge_index, edge_attr, n_nodes, reduce = 'mean')
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

    x = torch.ones((n_nodes, 1))
    y = torch.empty(1, 0)


    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)



def load_dataset(dataname, batch_size, hydra_path, sample, num_train):
    domains = ['asn', 'bio', 'chem', 'col', 'eco', 'econ', 'email', 'power', 'road', 'rt', 'socfb', 'web', 'citation', 'soc', 'qm9topo', 'InternetTopology_500_N100_skitter', 'random_erdos-renyi_500_M100_P3','USA-road_d_500_N100_NY']
    
    for domain in domains:
        if not os.path.exists(f'{hydra_path}\\..\\graphs{sample}\\{domain}\\train.pt'):
            print(111, domain)
            data = torch.load(f'{hydra_path}\\..\\graphs{sample}\\{domain}\\{domain}.pt', weights_only=True)

            #fix seed
            torch.manual_seed(0)

            #random permute and split
            n = len(data)
            indices = torch.randperm(n)

            if domain == 'eco':
                train_indices = indices[:4].repeat(50)
                val_indices = indices[4:5].repeat(50)
                test_indices = indices[5:]
            else:
                train_indices = indices[:int(0.8 * n)]
                val_indices = indices[int(0.8 * n):int(0.9 * n)]
                test_indices = indices[int(0.9 * n):]

            train_data = [data[_] for _ in train_indices]
            val_data = [data[_] for _ in val_indices]
            test_data = [data[_] for _ in test_indices]

            torch.save(train_indices, f'{hydra_path}\\..\\graphs{sample}\\{domain}\\train_indices.pt')
            torch.save(val_indices, f'{hydra_path}\\..\\graphs{sample}\\{domain}\\val_indices.pt')
            torch.save(test_indices, f'{hydra_path}\\..\\graphs{sample}\\{domain}\\test_indices.pt')
            
            torch.save(train_data, f'{hydra_path}\\..\\graphs{sample}\\{domain}\\train.pt')
            torch.save(val_data, f'{hydra_path}\\..\\graphs{sample}\\{domain}\\val.pt')
            torch.save(test_data, f'{hydra_path}\\..\\graphs{sample}\\{domain}\\test.pt')

    if dataname in domains:
        print(222, dataname)

        if num_train == -1:
            train_data = [arrange_data(_) for _ in torch.load(f'{hydra_path}\\..\\graphs{sample}\\{dataname}\\train.pt', weights_only=True)]
        else:
            train_data = [arrange_data(_) for _ in torch.load(f'{hydra_path}\\..\\graphs{sample}\\{dataname}\\train.pt', weights_only=True)][:num_train]
        val_data = [arrange_data(_) for _ in torch.load(f'{hydra_path}\\..\\graphs{sample}\\{dataname}\\val.pt', weights_only=True)]

        if dataname != 'eco':
            test_data = [arrange_data(_) for _ in torch.load(f'{hydra_path}\\..\\graphs{sample}\\{dataname}\\test.pt', weights_only=True)]
        else:
            test_data = [arrange_data(_) for _ in torch.load(f'{hydra_path}\\..\\graphs{sample}\\{dataname}\\train.pt', weights_only=True)] + [arrange_data(_) for _ in torch.load(f'{hydra_path}\\..\\graphs{sample}\\{dataname}\\val.pt', weights_only=True)] + [arrange_data(_) for _ in torch.load(f'{hydra_path}\\..\\graphs{sample}\\{dataname}\\test.pt', weights_only=True)]
            
    elif 'wo' in dataname:
        held_out = dataname.split('wo')[-1].strip(' ')
        train_data, val_data, test_data = [], [], []

        for domain in domains:
            if domain == held_out or domain == 'qm9topo':
                continue
            
            train_data.extend([arrange_data(_) for _ in torch.load(f'{hydra_path}\\..\\graphs{sample}\\{domain}\\train.pt', weights_only=True)])
            val_data.extend([arrange_data(_) for _ in torch.load(f'{hydra_path}\\..\\graphs{sample}\\{domain}\\val.pt')], weights_only=True)
            test_data.extend([arrange_data(_) for _ in torch.load(f'{hydra_path}\\..\\graphs{sample}\\{domain}\\test.pt')], weights_only=True)
    elif dataname == 'all':
        train_data, val_data, test_data = [], [], []

        for domain in domains:
            if domain == 'qm9topo':
                continue
            
        train_data.extend([arrange_data(_) for _ in torch.load(f'{hydra_path}\\..\\graphs{sample}\\{domain}\\train.pt')], weights_only=True)
        val_data.extend([arrange_data(_) for _ in torch.load(f'{hydra_path}\\..\\graphs{sample}\\{domain}\\val.pt')], weights_only=True)
        test_data.extend([arrange_data(_) for _ in torch.load(f'{hydra_path}\\..\\graphs{sample}\\{domain}\\test.pt')], weights_only=True)
        

    print('Size of dataset', len(train_data), len(val_data), len(test_data))

    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size = batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


    return train_loader, val_loader, test_loader




def main(dataname, batch_size, hydra_path, sample, num_train):
    train, val, test = load_dataset(dataname, batch_size, hydra_path, sample, num_train)

    train_batch = next(iter(train))
    val_batch = next(iter(val))
    test_batch = next(iter(test))

    # Print shapes and sample data
    print("\n=== Train DataLoader ===")
    print(f"Batch structure: {type(train_batch)}")  # Could be tuple, dict, etc.
    # print(train_batch)
    # print("length", len(train_batch))
    # print("X", train_batch.x)
    # print("batch", train_batch.batch)
    # print("edge index", train_batch.edge_index)
    # print("edge_attr", train_batch.edge_attr)
    # print("ptr", train_batch.ptr)
    
    item = train_batch

    # Extract the ptr values
    ptr = item.ptr.tolist()

    # Reconstruct the graphs
    graphs = []
    for i in range(len(ptr) - 1):
        start = ptr[i]
        end = ptr[i + 1]

        # Extract the graph-specific data
        graph_x = item.x[start:end]
        graph_batch_indices = torch.where((item.batch >= i) & (item.batch < i + 1))[0]
        graph_edge_index_mask = torch.isin(item.edge_index[0], torch.arange(start, end)) & torch.isin(item.edge_index[1], torch.arange(start, end))
        graph_edge_index = item.edge_index[:, graph_edge_index_mask] - start
        graph_edge_attr = item.edge_attr[graph_edge_index_mask]
        graph_y = item.y[i]

        # Create a Data object for the graph
        graph = Data(x=graph_x, edge_index=graph_edge_index, edge_attr=graph_edge_attr, y=graph_y)
        graphs.append(graph)

    # 'graphs' now contains the reconstructed graphs
    print(graphs)
    print(graphs[0])
    print(graphs[1])
    print(graphs[2])
    print(train_batch)



    # know what type of data we are working with
    raw_data = torch.load(f'{hydra_path}\\..\\graphs{sample}\\{dataname}\\train.pt', weights_only=True)

    print(f"Type of loaded data: {type(raw_data)}")  
    print(f"List length: {len(raw_data)}")
    print(f"Shape of the first element: {raw_data[0].shape}")
    print("Arrange 1", arrange_data(raw_data[0]))
    print(f"Shape of the second element: {raw_data[1].shape}")
    print(f"Shape of the third element: {raw_data[2].shape}")
    

if __name__ == "__main__":
    main('USA-road_d_500_N100_NY', 32, 'C:\\TRABAJO\\Iker\\MisCosas\\trabajo\\Practicas\\PracticasUniversidad\\codigo\\LOPfuntzioak\\LOPsortzailea\\LGGM\\TEST', 'seed', 3)
