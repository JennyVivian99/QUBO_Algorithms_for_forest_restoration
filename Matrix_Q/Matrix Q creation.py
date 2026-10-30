# Matrix Q creation with mock data
# region import libraries
import numpy as np
import csv
#endregion

#region helper functions
def export_results_to_csv(filename, Q, species_list, tree_list, K, params):
    """
    Exports the QUBO matrix to CSV
    """
    I_size = len(species_list)
    G_size = len(tree_list)
    
    # Create variable labels: Species_Tree or Slack_k
    labels = []
    for i in species_list:
        for g in tree_list:
            labels.append(f"{i}_Tree{g}")
    for k in range(K):
        labels.append(f"Slack_{k}")

    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        
        # Write Metadata Header
        writer.writerow(["# Forestry QUBO Export Details"])
        writer.writerow(["# Species", ", ".join(species_list)])
        writer.writerow(["# Tree Count", G_size])
        for key, val in params.items():
            writer.writerow([f"# Coefficient {key}", val])
        writer.writerow([])

        # Write Matrix Header Row
        writer.writerow([""] + labels)

        # Write Matrix Rows
        for i, row_label in enumerate(labels):
            writer.writerow([row_label] + list(Q[i]))

    print(f"Matrix successfully exported to {filename}")

def build_forestry_qubo(species_list, tree_list, tree_neighbors, R, E, C_cap, N_max, N_target, P_unique, B, D):
    I_size = len(species_list)
    G_size = len(tree_list)
    
    K = int(np.ceil(np.log2(N_max))) if N_max > 0 else 0
    num_vars = (I_size * G_size) + K
    Q = np.zeros((num_vars, num_vars))

    def idx(i, g): return i * G_size + g
    def idx_s(k): return (I_size * G_size) + k

    # Terms logic (Diversity, Capacity, Cost, Interaction, Unique)
    # Diversity
    for i in range(I_size):
        for g in range(G_size):
            Q[idx(i, g), idx(i, g)] += 1.0 / (N_target**2)
            for gp in range(g + 1, G_size):
                Q[idx(i, g), idx(i, gp)] += 2.0 / (N_target**2)
    # Environmental capacity
    all_vars, coeffs = [], []
    for i in range(I_size):
        for g in range(G_size):
            all_vars.append(idx(i, g)); coeffs.append(1)
    for k in range(K):
        all_vars.append(idx_s(k)); coeffs.append(2**k)

    for a in range(len(all_vars)):
        v_a, w_a = all_vars[a], coeffs[a]
        Q[v_a, v_a] += C_cap * (w_a**2 - 2 * N_max * w_a)
        for b in range(a + 1, len(all_vars)):
            v_b, w_b = all_vars[b], coeffs[b]
            Q[v_a, v_b] += C_cap * (2 * w_a * w_b)
    # Effect on the environment for each species
    for i in range(I_size):
        for g in range(G_size):
            Q[idx(i, g), idx(i, g)] += (R - E[i])
    # Clustering penalty 
    for (g_a, g_b) in tree_neighbors:
        for i in range(I_size):
            u, v = sorted([idx(i, g_a), idx(i, g_b)])
            Q[u, v] += B
            for i_other in range(I_size):
                if i != i_other:
                    u_diff, v_diff = sorted([idx(i, g_a), idx(i_other, g_b)])
                    Q[u_diff, v_diff] -= D
    # Constraint on the presence of one tree per location
    for t in range(G_size):
        for i1 in range(I_size):
            for i2 in range(i1 + 1, I_size):
                Q[idx(i1, t), idx(i2, t)] += P_unique
    
    # Export parameters
    params = {
        "R": R, "C_cap": C_cap, "N_max": N_max, 
        "N_target": N_target, "P_unique": P_unique, 
        "B": B, "D": D
    }
    export_results_to_csv(f"Quantum_simple_simulation_results_{len(species_list)}_Species.csv", Q, species_list, tree_list, K, params)
    return Q, print (Q)
#endregion

# region examples
# List of Species
species = ["S.macrophylla", "A.mangium", "P.indicus"]

# List of Tree IDs (1 to 10)
trees = list(range(10)) 

# Neighbors (which trees are physically close to each other)
# Example: Tree 0 is near Tree 1, Tree 1 is near Tree 2
tree_neighbors = [(0, 1), (1, 2), (2, 3), (0, 3), (2, 5), (4,5)]

# Effect on the environment for each species
E_species = [1.5, 2.0, 1.2] 

# Run the builder
Q_matrix = build_forestry_qubo(
    species, trees, tree_neighbors, 
    R=2, E=E_species, C_cap=5, N_max=5, 
    N_target=2, P_unique=100, B=5, D=3
)
#endregion