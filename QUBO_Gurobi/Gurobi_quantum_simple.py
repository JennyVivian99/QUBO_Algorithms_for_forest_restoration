# Gurobi simple simulation
# region libraries
import numpy as np
import matplotlib.pyplot as plt
import csv
from adjustText import adjust_text

# Gurobi
import gurobipy as gp
from gurobipy import GRB
# endregion

# region helper functions definition
def decode_planting_plan(sample, locations, species):
    print("\n FINAL PLANTING PLAN ")
    trees_planted = 0
    breakdown = {s: 0 for s in species}
    
    for g_idx, loc_name in enumerate(locations):
        found_tree = False
        for s_idx, spec_name in enumerate(species):
            var_name = f'y[{g_idx}][{s_idx}]'
            if sample.get(var_name, 0) == 1:
                print(f"{loc_name}: Plant {spec_name}")
                trees_planted += 1
                breakdown[spec_name] += 1
                found_tree = True
        if not found_tree:
            print(f"⚪ {loc_name}: -- Empty --")

    print("\n SUMMARY ")
    print(f"Total Trees Planted: {trees_planted}")
    print("Diversity Breakdown:")
    for spec, count in breakdown.items():
        print(f"  - {spec}: {count}")

def plot_forest(sample, locations, species, coordinates, num_species="try"):
    """
    Visualizes the result. 
    """
    if len(coordinates) != len(locations):
        print("Error: Coordinate mismatch.")
        return

    plt.figure(figsize=(8, 8))
    colors = ['blue', 'red','green', 'orange', 'purple']
    
    # Plot holes
    x_coords = [c[0] for c in coordinates]
    y_coords = [c[1] for c in coordinates]
    plt.scatter(x_coords, y_coords, c='lightgray', s=300, label='Empty Hole')
    texts = []

    # Plot trees
    for g_idx, (x, y) in enumerate(coordinates):
        for s_idx, spec_name in enumerate(species):
            var_name = f'y[{g_idx}][{s_idx}]'
            if sample.get(var_name, 0) == 1:
                color = colors[s_idx % len(colors)]
                plt.scatter(x, y, c=color, s=200, edgecolors='black', label=spec_name)

                t = plt.text(
                    x, y, spec_name, 
                    fontsize=9, weight='bold', zorder=11,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray')
                )
                texts.append(t)

    if texts:
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))

    plt.title(f"Optimized Reforestation Map (Gurobi Simulation) {num_species}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle='--', alpha=0.5)

    filename = f"Quantum_Gurobi_simple_simulation_output_{num_species.replace(' ', '_')}.png"
    plt.savefig(filename, bbox_inches='tight')
    print(f"Plot saved as '{filename}'. Check your project folder.")
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    # plt.show()

def print_diversity_metrics(sample, locations, species, N_target):
    print("\n DIVERSITY METRICS ")
    
    counts = {s: 0 for s in species}
    total_planted = 0
    
    for s_idx, spec_name in enumerate(species):
        c = 0
        for g_idx in range(len(locations)):
            var_name = f'y[{g_idx}][{s_idx}]'
            if sample.get(var_name, 0) == 1:
                c += 1
        
        counts[spec_name] = c
        total_planted += c

    sum_sq_counts = sum(c**2 for c in counts.values())
    diversity_penalty = sum_sq_counts / (N_target**2)
    
    print(f"Species Counts: {counts}")
    print(f"Sum of Squares (Concentration): {sum_sq_counts}")
    print(f"Diversity Penalty Value (H1): {diversity_penalty:.4f}")
    
    if total_planted > 0:
        simpson_D = sum((c/total_planted)**2 for c in counts.values())
        print(f"Simpson Diversity Index (D): {simpson_D:.4f}")

        shannon_H = 0
        for count in counts.values():
            if count > 0: 
                pi = count / total_planted
                shannon_H += -1 * pi * np.log(pi)
        print(f"Shannon Diversity Index (H'): {shannon_H:.4f}")
    else:
        print("Diversity Indices: N/A (No trees planted)")

def export_results_to_csv(filename, sample, locations, species, coords):
    print(f"\n Saving results to {filename}...")
    plan_rows = []
    counts = {s: 0 for s in species}
    total_planted = 0

    for g_idx, loc_name in enumerate(locations):
        x, y = coords[g_idx]
        planted_spec = "Empty"
        for s_idx, spec_name in enumerate(species):
            var_name = f'y[{g_idx}][{s_idx}]'
            if sample.get(var_name, 0) == 1:
                planted_spec = spec_name
                counts[spec_name] += 1
                total_planted += 1
        plan_rows.append([loc_name, x, y, planted_spec])

    try:
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([" PLANTING MAP "])
            writer.writerow(["Location_ID", "X_Coord", "Y_Coord", "Species_Planted"])
            writer.writerows(plan_rows)
            writer.writerow([])
            writer.writerow([" SPECIES BREAKDOWN "])
            for spec, count in counts.items():
                writer.writerow([spec, count])
        print("File saved.")
    except IOError as e:
        print(f"Error saving CSV: {e}")
# endregion


# region Gurobi model definition
def solve_with_gurobi(locations, species, neighbors):
    """
    Constructs and solves the Constrained Binary Quadratic Program (CBQP) using Gurobi.
    """
    # Hyperparameters
    N_target = 8.0     # Target for diversity scaling
    N_max = 10         # Max carrying capacity 
    
    # Ecological Constants
    R = 0.5            
    if len(species) == 2:
        E = {'Species_A': 0.8, 'Species_B': 0.6}
    else:
        E = {'Species_A': 0.8, 'Species_B': 0.6, "Species_C": 0.7}
    
    # Weights 
    B = 1.0   
    D = 1.0   
    
    print("\n[1] Building Gurobi Model...")
    # Initialize Gurobi model instance
    m = gp.Model("Forest_Restoration")
    # Suppress verbose solver output for cleaner console
    m.setParam('OutputFlag', 0)

    # Define variables
    # y[g][s] = 1 if species s is planted at location g
    y = {}
    for g_idx in range(len(locations)):
        for s_idx in range(len(species)):
            # Name format strictly matches previous logic
            y[g_idx, s_idx] = m.addVar(vtype=GRB.BINARY, name=f"y[{g_idx}][{s_idx}]")
            
    m.update()

    # Build objective expression
    obj = gp.QuadExpr()

    # 1. Diversity Term: Sum_s (Count_s)^2 / N_target^2
    for s_idx in range(len(species)):
        species_sum = gp.quicksum(y[g_idx, s_idx] for g_idx in range(len(locations)))
        obj += (species_sum * species_sum) / (N_target**2)

    # 2. Ecological Suitability (Linear Cost): Cost = R - E[species]
    for g_idx in range(len(locations)):
        for s_idx in range(len(species)):
            cost = R - E[species[s_idx]]
            obj += cost * y[g_idx, s_idx]

    # 3. Clustering / Neighbor Interactions
    for (loc_a, loc_b) in neighbors:
        # Reward any neighbors: -D * Sum(y_a) * Sum(y_b)
        sum_a = gp.quicksum(y[loc_a, s_i] for s_i in range(len(species)))
        sum_b = gp.quicksum(y[loc_b, s_j] for s_j in range(len(species)))
        obj -= D * sum_a * sum_b
        
        # Penalty for proximity of trees of the same species: (B + D) * Sum(y_a_i * y_b_i)
        for s_idx in range(len(species)):
            obj += (B + D) * y[loc_a, s_idx] * y[loc_b, s_idx]

    # Set objective
    m.setObjective(obj, GRB.MINIMIZE)

    # Add constraints
    # Global Capacity Constraint: Sum(all trees) <= N_max
    m.addConstr(
        gp.quicksum(y[g, s] for g in range(len(locations)) for s in range(len(species))) <= N_max, 
        "Global_Capacity"
    )

    # One tree per location: Sum_species(y_g) <= 1
    for g_idx in range(len(locations)):
        m.addConstr(
            gp.quicksum(y[g_idx, s] for s in range(len(species))) <= 1, 
            f"OneTree_Loc_{g_idx}"
        )

    # Solve the model
    print("\n[2] Solving via Gurobi Optimizer...")
    m.optimize()

    # Extract results
    best_sample = {}
    fval = None
    
    if m.Status == GRB.OPTIMAL:
        fval = m.ObjVal
        for v in m.getVars():
            # Round to ensure pure binary output (0 or 1) rather than floats like 0.9999
            best_sample[v.VarName] = int(round(v.X))
    else:
        print("No optimal solution found!")

    return best_sample, fval, N_target
# endregion

if __name__ == "__main__":
    
    # region 2 species, 4 locations
    print("\n" + "="*50)
    print(" RUNNING 2 SPECIES / 4 LOCATIONS ")
    print("="*50)   
    
    locations = ['Loc_0', 'Loc_1', 'Loc_2', 'Loc_3'] 
    species = ['Species_A', 'Species_B'] 
    neighbors = [(0, 1), (1, 2), (2,3), (3,0)] 
    coords = [(0,0), (0,1), (1,1), (1,0)]  

    # Run Gurobi Solver
    best_sample, fval, N_target = solve_with_gurobi(locations, species, neighbors)

    print("\n" + "="*30)
    print("Gurobi Simulation Results")
    print("="*30)
    print(f"Final Energy (Objective): {fval:.4f}")
    print("\n Active Variables (set to 1):")
    
    for var_label in sorted(best_sample.keys()):
        if best_sample[var_label] == 1:
            print(f"  • {var_label} = 1")
            
    # Decode and visualize
    decode_planting_plan(best_sample, locations, species)
    
    try:
        plot_forest(best_sample, locations, species, coords, num_species="2 species")
    except Exception as e:
        print(f"Could not plot: {e}")
    
    print_diversity_metrics(best_sample, locations, species, N_target)
    csv_filename = "Quantum_Gurobi_simple_simulation_results_2Species.csv"
    export_results_to_csv(csv_filename, best_sample, locations, species, coords)
    # endregion

    # region 3 species, 5 locations
    print("\n" + "="*50)
    print(" RUNNING 3 SPECIES / 5 LOCATIONS ")
    print("="*50)
    
    locations_3 = ['Loc_0', 'Loc_1', 'Loc_2', 'Loc_3', 'Loc_4']
    species_3 = ['Species_A', 'Species_B', 'Species_C']
    neighbors_3 = [(0, 1), (1, 2), (2,3), (3,4), (4,0)]
    coords_3 = [(0,0), (0,1), (1,1), (1,0), (0.5, 1.5)]

    best_sample_3, fval_3, N_target_3 = solve_with_gurobi(locations_3, species_3, neighbors_3)

    print("\n" + "="*30)
    print("Gurobi Simulation Results")
    print("="*30)
    print(f"Final Energy (Objective): {fval_3:.4f}")
    print("\n Active Variables (set to 1):")
    
    for var_label in sorted(best_sample_3.keys()):
        if best_sample_3[var_label] == 1:
            print(f"  • {var_label} = 1")
            
    decode_planting_plan(best_sample_3, locations_3, species_3)
    
    try:
        plot_forest(best_sample_3, locations_3, species_3, coords_3, num_species="3 species")
    except Exception as e:
        print(f"Could not plot: {e}")
    
    print_diversity_metrics(best_sample_3, locations_3, species_3, N_target_3)
    csv_filename_3 = "Quantum_Gurobi_simple_simulation_results_3Species_5Loc.csv"
    export_results_to_csv(csv_filename_3, best_sample_3, locations_3, species_3, coords_3)
    # endregion

    # region 3 species, 8 locations
    print("\n" + "="*50)
    print(" RUNNING 3 SPECIES / 8 LOCATIONS ")
    print("="*50)
    
    locations_8 = ['Loc_0', 'Loc_1', 'Loc_2', 'Loc_3', "Loc_4",'Loc_5', 'Loc_6', 'Loc_7']
    species_8 = ['Species_A', 'Species_B', 'Species_C'] 
    neighbors_8 = [(0, 1), (1, 2), (2,3), (3,0), (3,4), (4,5), (5,2),(5,6), (6,7), (7,4)] 
    coords_8 = [(0,0), (0,1), (1,1), (1,0), (1,2), (0,2), (1,3), (0,3)]  

    best_sample_8, fval_8, N_target_8 = solve_with_gurobi(locations_8, species_8, neighbors_8)

    print("Gurobi Simulation Results")
    print(f"Final Energy (Objective): {fval_8:.4f}")
    print("\n Active Variables (set to 1):")
    
    for var_label in sorted(best_sample_8.keys()):
        if best_sample_8[var_label] == 1:
            print(f"  • {var_label} = 1")
            
    decode_planting_plan(best_sample_8, locations_8, species_8)
    
    try:
        plot_forest(best_sample_8, locations_8, species_8, coords_8, num_species="3 species 8 loc")
    except Exception as e:
        print(f"Could not plot: {e}")
    
    print_diversity_metrics(best_sample_8, locations_8, species_8, N_target_8)
    csv_filename_8 = "Quantum_Gurobi_simple_simulation_results_3species_8loc.csv"
    export_results_to_csv(csv_filename_8, best_sample_8, locations_8, species_8, coords_8)
    # endregion