import torch
import os


OLD_DATA_DIR = "./OLD_DATA/"
NEW_DATA_DIR = "./NEW_DATA/"


# This function uses previously generated files for GraphRNN and generates files suited for LGGM.
# Concretely this function takes two files _A.txt and _graph_indicator.txt, which indicate the edge connections among graphs and generated train.pt, val.pt and test.pt
def createDataForLGGM(filename, filename_base, validation = False):
    # Step 1: Load node-to-graph mapping
    with open(".\\OLD_DATA\\" + filename + "_graph_indicator.txt", "r") as f:
        node_to_graph = [int(line.strip()) for line in f.readlines()]

    num_nodes = len(node_to_graph)
    num_graphs = max(node_to_graph)

    # Step 2: Load edges
    edges = []
    with open(".\\OLD_DATA\\" + filename + "_A.txt", "r") as f:
        for line in f:
            src, dst = map(int, line.strip().split(","))
            edges.append((src - 1, dst - 1))  # Convert to zero-based index

    # Step 3: Organize edges into graphs
    graphs = {i: [] for i in range(1, num_graphs + 1)}
    for src, dst in edges:
        graph_id = node_to_graph[src]  # Determine the graph this node belongs to
        if node_to_graph[dst] == graph_id:  # Ensure both nodes are in the same graph
            graphs[graph_id].append((src, dst))

    # Step 4: Convert graphs to adjacency matrices
    adj_matrices = []
    for graph_id, edge_list in graphs.items():
        nodes_in_graph = [i for i, g in enumerate(node_to_graph) if g == graph_id]
        n = len(nodes_in_graph)
        adj_matrix = torch.zeros((n, n), dtype=torch.float32)

        node_idx_map = {node: idx for idx, node in enumerate(nodes_in_graph)}
        for src, dst in edge_list:
            if src in node_idx_map and dst in node_idx_map:
                adj_matrix[node_idx_map[src], node_idx_map[dst]] = 1

        adj_matrices.append(adj_matrix)

 


    output_dataset_dir = os.path.join(NEW_DATA_DIR, filename_base)
    os.makedirs(output_dataset_dir, exist_ok=True)

    if validation:
        split_index = len(adj_matrices) // 2
        test = adj_matrices[:split_index]  # First half
        val = adj_matrices[split_index:]  # Second half

        # Save each half separately
        torch.save(test, os.path.join(output_dataset_dir, "test.pt"))
        torch.save(val, os.path.join(output_dataset_dir, "val.pt"))

    else:
        # Save the file
        torch.save(adj_matrices, os.path.join(output_dataset_dir, "train.pt"))




# if __name__ == "__main__":
#     createDataForLGGM("USA-road_d_500_N100_NY")
#     createDataForLGGM("USA-road_d_500_N100_NY_test", validation = True)

if __name__ == "__main__":
    processed_datasets = set() # To keep track of datasets already processed

    for filename in os.listdir(OLD_DATA_DIR):
        if filename.endswith("_graph_indicator.txt"):
            # Extract the base filename (e.g., "USA-road_d_500_N100_NY")
            base_filename = filename.replace("_graph_indicator.txt", "")

            if base_filename.endswith("_test"):
                base_filename = base_filename.replace("_test", "")

            if base_filename in processed_datasets:
                continue # Skip if this dataset has already been processed

            # Check if the corresponding _A.txt file exists
            if os.path.exists(os.path.join(OLD_DATA_DIR, base_filename + "_A.txt")):
                print(f"Processing: {base_filename}")
                createDataForLGGM(base_filename, base_filename)
                createDataForLGGM(base_filename +"_test", base_filename, validation = True)
                processed_datasets.add(base_filename)
            else:
                print(f"Skipping {base_filename}: Corresponding _A.txt not found.")
