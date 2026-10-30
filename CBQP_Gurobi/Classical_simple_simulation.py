# Classic simple simulation
#region libraries
# import libraries
import numpy as np 
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import math
import csv
from adjustText import adjust_text
#endregion

# region helper functions definition
def decode_planting_plan(sample, locations, species):
    print("\n FINAL PLANTING PLAN ")
    trees_planted = 0
    breakdown = {s: 0 for s in species}
    
    for g_idx, loc_name in enumerate(locations):
        found_tree = False
        for s_idx, spec_name in enumerate(species):
            # The keys must match the format 'y[g][s]'
            var_name = f'y[{g_idx}][{s_idx}]'
            if sample.get(var_name) == 1:
                print(f"{loc_name}: Plant {spec_name}")
                trees_planted += 1
                breakdown[spec_name] += 1
                found_tree = True
        if not found_tree:
            print(f"{loc_name}: Empty ")

    print("\n SUMMARY ")
    print(f"Total Trees Planted: {trees_planted}")
    print("Diversity Breakdown:")
    for spec, count in breakdown.items():
        print(f"  - {spec}: {count}")

def plot_forest(sample, locations, species, coordinates, filename="try", num_species="try"):
    """
    Visualizes the result. 
    """
    if len(coordinates) != len(locations):
        print("Error: Coordinate mismatch.")
        return

    plt.figure(figsize=(8, 8))
    colors = ['blue', 'red','green',  'orange', 'purple']
    
    # Plot holes
    x_coords = [c[0] for c in coordinates]
    y_coords = [c[1] for c in coordinates]
    plt.scatter(x_coords, y_coords, c='lightgray', s=300, label='Empty Hole')

    # List to hold text objects for adjustText
    texts = []

    # Plot trees
    for g_idx, (x, y) in enumerate(coordinates):
        for s_idx, spec_name in enumerate(species):
            var_name = f'y[{g_idx}][{s_idx}]'
            if sample.get(var_name) == 1:
                color = colors[s_idx % len(colors)]
                plt.scatter(x, y, c=color, s=200, edgecolors='black', label=spec_name)
                # Append text objects to the list instead of drawing immediately
                t = plt.text(
                    x, y, spec_name, 
                    fontsize=9, weight='bold', 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray')
                )
                texts.append(t)

    # Automatically shift labels so they never overlap with each other or points
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))

    plt.title(f"Optimized Reforestation Map (Classical computation) {num_species}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save plot
    plt.savefig(filename, bbox_inches='tight')  # bbox_inches='tight' prevents cut-off labels
    print(f"Plot saved as '{filename}'. Check your project folder.")
    plt.close()

    # Legend fix
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    

def print_diversity_metrics(sample, locations, species, N_target):
    print("\n DIVERSITY METRICS ")
    
    counts = {s: 0 for s in species}
    total_planted = 0
    
    for s_idx, spec_name in enumerate(species):
        c = 0
        for g_idx in range(len(locations)):
            var_name = f'y[{g_idx}][{s_idx}]'
            if sample.get(var_name) == 1:
                c += 1
        
        counts[spec_name] = c
        total_planted += c

    sum_sq_counts = sum(c**2 for c in counts.values())
    diversity_penalty = sum_sq_counts / (N_target**2)
    
    print(f"Species Counts: {counts}")
    print(f"Sum of Squares (Concentration): {sum_sq_counts}")
    print(f"Diversity Penalty Value (H1): {diversity_penalty:.4f}")
    
    if total_planted > 0:
        # Simpson Index
        simpson_D = sum((c/total_planted)**2 for c in counts.values())
        diversity_index = simpson_D
        print(f"Actual Simpson Diversity Index (D; the lower, the better): {diversity_index:.4f}")

        # Shannon Index
        shannon_H = 0
        for count in counts.values():
            if count > 0: 
                # p_i is the proportion of species i
                pi = count / total_planted
                # H' = -sum(pi * ln(pi))
                shannon_H += -1 * pi * math.log(pi)
        print(f"Shannon Diversity Index (H'): {shannon_H:.4f}")

    else:
        print("Simpson Index: N/A (No trees planted)")
        print("Shannon Index: N/A (No trees planted)")

def export_results_to_csv(filename, sample, locations, species, coords, N_target):
    """
    Saves the planting plan and diversity metrics to a CSV file.
    """
    print(f"\n Saving results to {filename}...")
    
    # Data preparation
    plan_rows = []
    counts = {s: 0 for s in species}
    total_planted = 0
    
    # Generate Map Data
    for g_idx, loc_name in enumerate(locations):
        x, y = coords[g_idx]
        planted_spec = "Empty"
        
        for s_idx, spec_name in enumerate(species):
            var_name = f'y[{g_idx}][{s_idx}]'
            if sample.get(var_name) == 1:
                planted_spec = spec_name
                counts[spec_name] += 1
                total_planted += 1
        
        plan_rows.append([loc_name, x, y, planted_spec])

    # Calculate Metrics
    sum_sq_counts = sum(c**2 for c in counts.values())
    diversity_penalty = sum_sq_counts / (N_target**2)
    
    simpson_D = 0
    shannon_H = 0
    if total_planted > 0:
        simpson_D = sum((c/total_planted)**2 for c in counts.values())
        for count in counts.values():
            if count > 0:
                pi = count / total_planted
                shannon_H += -1 * pi * np.log(pi)

    # Export to .csv
    try:
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            
            # Section 1: The Map
            writer.writerow([" PLANTING MAP "])
            writer.writerow(["Location_ID", "X_Coord", "Y_Coord", "Species_Planted"])
            writer.writerows(plan_rows)
            
            writer.writerow([]) # Empty row for spacing
            
            # Section 2: The Metrics
            writer.writerow([" DIVERSITY METRICS "])
            writer.writerow(["Metric", "Value", "Notes"])
            writer.writerow(["Total Trees", total_planted, f"Max Capacity: {len(locations)}"])
            writer.writerow(["Shannon Index (H')", f"{shannon_H:.4f}", "Higher is better"])
            writer.writerow(["Simpson Index (D)", f"{simpson_D:.4f}", "Lower is better"])
            writer.writerow(["Quadratic Penalty Term", f"{diversity_penalty:.4f}", f"Target N={N_target}"])
            
            # Section 3: Species Breakdown
            writer.writerow([])
            writer.writerow([" SPECIES BREAKDOWN "])
            for spec, count in counts.items():
                writer.writerow([spec, count])
                
        print("File saved")
        
    except IOError as e:
        print(f"Error saving CSV: {e}")
#endregion

# region model definition
def solve_classical(T_species, G_locations, Neighbors, N_target, N_max, R, E, B, D):
    """
    Returns: (sample_dict, energy_value)
    """
    
    # Initialize Model
    model = gp.Model("Forest_restoration") # Model name for clarity in logs
    model.setParam('OutputFlag', 0)

    # Indices
    T_idxs = range(len(T_species))   # i
    G_idxs = range(len(G_locations)) # g

    # Variables: y[i, g] 
    # (Species i at Location g)
    y = model.addVars(T_idxs, G_idxs, vtype=GRB.BINARY, name="y")

    # Objectives
    # Diversity (Smoothing per species across the grid)
    # Sum over locations g for each species i
    obj_diversity = 0
    for i in T_idxs:
        count_i = gp.quicksum(y[i, g] for g in G_idxs)
        obj_diversity += (count_i * count_i) / (N_target**2)

    # Ecology Costs
    obj_ecology = 0
    for i in T_idxs:
        for g in G_idxs:
            cost = R - E[T_species[i]]
            obj_ecology += cost * y[i, g]

    # Neighbor Clustering (Graph interactions)
    obj_clustering = 0
    for (loc_u, loc_v) in Neighbors:
        sum_u = gp.quicksum(y[i, loc_u] for i in T_idxs)
        sum_v = gp.quicksum(y[i, loc_v] for i in T_idxs)
        
        # Term D: Reward having neighbors
        term_adjacency = -D * (sum_u * sum_v)

        # Term B: Penalty for the same species if nearby
        term_similarity = 0
        for i in T_idxs:
            term_similarity += y[i, loc_u] * y[i, loc_v]
            
        obj_clustering += term_adjacency + ((B + D) * term_similarity)

    model.setObjective(obj_diversity + obj_ecology + obj_clustering, GRB.MINIMIZE)

    # Constraints
    # Global Capacity Constraint
    model.addConstr(
        gp.quicksum(y[i, g] for i in T_idxs for g in G_idxs) <= N_max,
        name="Global_Capacity"
    )

    # Uniqueness Constraint
    # For each location g, sum of species <= 1
    for g in G_idxs:
        model.addConstr(
            gp.quicksum(y[i, g] for i in T_idxs) <= 1,
            name=f"OneTree_at_Loc_{g}"
        )

    # Solve
    model.optimize()

    # Format Output
    if model.status == GRB.OPTIMAL:
        sample = {}
        # Convert Gurobi variables to simple dictionary { 'y[g][i]': 0/1 }, from y[i,g] to y[g][i]
        for g in G_idxs:
            for i in T_idxs:
                # Check if effectively 1
                val = 1 if y[i, g].X > 0.5 else 0
                key_name = f"y[{g}][{i}]"
                sample[key_name] = val
        
        return sample, model.objVal
    else:
        print("No solution found")
        return {}, 0
#endregion

# region 2 species 4 locations 
# main execution, here define the variables value
# Set here the hyperparameters
if __name__ == "__main__":
    
    # Setup Data
    locations = ['Loc_0', 'Loc_1', 'Loc_2', 'Loc_3'] 
    species = ['Species_A', 'Species_B']
    neighbors = [(0, 1), (1, 2), (2,3), (3,0)] 
    coords = [(0,0), (0,1), (1,1), (1,0)] 

    # Hyperparameters
    N_target = 8 
    N_max = 10         
    R = 0.5           
    E_vals = {'Species_A': 0.8, 'Species_B': 0.6}
    B_coeff = 1.0   
    D_coeff = 1.0   

    # Run Classical Solver
    print("Solving with Classical Exact Solver...")
    best_sample, best_energy = solve_classical(
        species, locations, neighbors, 
        N_target, N_max, R, E_vals, B_coeff, D_coeff
    )

    # Output Inspection
    print(" Raw Classical Solver Output")
    print(f"Final Energy: {best_energy:.4f}")

    print("\n Active Variables (set to 1):")
    # Sort keys for cleaner reading
    for var_label in sorted(best_sample.keys()):
        if best_sample[var_label] == 1:
            print(f"  • {var_label} = 1")

    print("\n Slack Variable Status:")
    print("  • N/A (Exact Solvers do not use slack bits)")

    # Decode & Visualize
    decode_planting_plan(best_sample, locations, species)
    filename = "Classic_optimization_simple_simulation_output2Species.png"
    # Calculate Metrics
    print_diversity_metrics(best_sample, locations, species, N_target)

    # Plot
    try:
        plot_forest(best_sample, locations, species, coords, filename=filename,num_species="2 species")
    except Exception as e:
        print(f"Could not plot: {e}")
    # Export to CSV
    export_results_to_csv("Quantum_simple_simulation_results2Species.csv", best_sample, locations, species, coords, N_target)
#endregion

# region 3 species 8 locations 
    print("\n" + "="*50)
    print(" RUNNING 3 SPECIES / 5 LOCATIONS ")
    print("="*50)
    
    # Setup Data
    locations = ['Loc_0', 'Loc_1', 'Loc_2', 'Loc_3', "Loc_4"] # consider n locations
    species = ['Species_A', 'Species_B', 'Species_C'] # consider n species
    neighbors = [(0, 1), (1, 2), (2,3), (3,4), (4,0)] # disposition of the location (which is near to the others)
    coords = [(0,0), (0,1), (1,1), (1,0), (0.5, 1.5)]  # coordinates of the locations

    # Hyperparameters
    N_target = 8 
    N_max = 10         
    R = 0.5           
    E_vals = {'Species_A': 0.8, 'Species_B': 0.6, "Species_C": 0.7}
    B_coeff = 1 
    D_coeff = 1 

    # Run Classical Solver
    print("Solving with Classical Exact Solver...")
    best_sample, best_energy = solve_classical(
        species, locations, neighbors, 
        N_target, N_max, R, E_vals, B_coeff, D_coeff
    )

    # Output Inspection
    print("\n" + "="*30)
    print(" Raw Classical Solver Output")
    print("="*30)
    print(f"Final Energy: {best_energy:.4f}")

    print("\n Active Variables (set to 1):")
    # Sort keys for cleaner reading
    for var_label in sorted(best_sample.keys()):
        if best_sample[var_label] == 1:
            print(f"  • {var_label} = 1")

    print("\n Slack Variable Status:")
    print("  • N/A (Exact Solvers do not use slack bits)")

    # Decode & Visualize
    decode_planting_plan(best_sample, locations, species)
    
    # Calculate Metrics
    print_diversity_metrics(best_sample, locations, species, N_target)
    filename = "Classic_optimization_simple_simulation_output3Species5Loc.png"

    # Plot
    try:
        plot_forest(best_sample, locations, species, coords,filename=filename,num_species="3 species")
    except Exception as e:
        print(f"Could not plot: {e}")

    # Export to CSV
    export_results_to_csv("Quantum_simple_simulation_results3Species5Loc.csv", best_sample, locations, species, coords, N_target)
#endregion

# region 3 species 8 locations 
    print("\n" + "="*50)
    print(" RUNNING 3 SPECIES / 8 LOCATIONS ")
    print("="*50)
    
    # Setup Data
    locations = ['Loc_0', 'Loc_1', 'Loc_2', 'Loc_3', "Loc_4",'Loc_5', 'Loc_6', 'Loc_7'] # consider n locations
    species = ['Species_A', 'Species_B', 'Species_C'] # consider n species
    neighbors = [(0, 1), (1, 2), (2,3), (3,0), (3,4), (4,5), (5,2),(5,6), (6,7), (7,4)] # disposition of the location (which is near to the others)
    coords = [(0,0), (0,1), (1,1), (1,0), (1,2), (0,2), (1,3), (0,3)]  # coordinates of the locations

    # Hyperparameters
    N_target = 8 
    N_max = 10         
    R = 0.5           
    E_vals = {'Species_A': 0.8, 'Species_B': 0.6, "Species_C": 0.7}
    B_coeff = 1 
    D_coeff = 1 

    # Run Classical Solver
    print("Solving with Classical Exact Solver...")
    best_sample, best_energy = solve_classical(
        species, locations, neighbors, 
        N_target, N_max, R, E_vals, B_coeff, D_coeff
    )

    # Output Inspection
    print("\n" + "="*30)
    print(" Raw Classical Solver Output")
    print("="*30)
    print(f"Final Energy: {best_energy:.4f}")

    print("\n Active Variables (set to 1):")
    # Sort keys for cleaner reading
    for var_label in sorted(best_sample.keys()):
        if best_sample[var_label] == 1:
            print(f"  • {var_label} = 1")

    print("\n Slack Variable Status:")
    print("  • N/A (Exact Solvers do not use slack bits)")

    # Decode & Visualize
    decode_planting_plan(best_sample, locations, species)
    
    # Calculate Metrics
    print_diversity_metrics(best_sample, locations, species, N_target)
    filename = "Classic_optimization_simple_simulation_output3Species8Locpng"

    # Plot
    try:
        plot_forest(best_sample, locations, species, coords,filename=filename,num_species="3 species")
    except Exception as e:
        print(f"Could not plot: {e}")

    # Export to CSV
    export_results_to_csv("Quantum_simple_simulation_results3Species8Loc.csv", best_sample, locations, species, coords, N_target)
#endregion