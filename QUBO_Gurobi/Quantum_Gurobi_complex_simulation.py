# Gurobi complex simulation
# region libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
import math
from adjustText import adjust_text

# Gurobi
import gurobipy as gp
from gurobipy import GRB
# endregion

# region helper functions
def get_var_name(i, m, g):
    return f"y_{i}_{m}_{g}"

def decode_planting_plan(result_dict, num_species, num_subareas, num_spots, species_list):
    print("\n FINAL PLANTING PLAN ", flush=True)
    trees_planted = 0
    breakdown = {spec: 0 for spec in species_list}
    
    for m in range(num_subareas):
        for g in range(num_spots):
            loc_name = f"Subarea_{m} Spot_{g}"
            found = False
            for i, spec_name in enumerate(species_list):
                var_name = get_var_name(i, m, g)
                if result_dict.get(var_name) == 1:
                    print(f"{loc_name}: Plant {spec_name}")
                    trees_planted += 1
                    breakdown[spec_name] += 1
                    found = True
            if not found:
                print(f"⚪ {loc_name}: -- Empty --")

    print("\n SUMMARY ", flush=True)
    print(f"Total Trees Planted: {trees_planted} of {num_species * num_subareas * num_spots}")
    for spec, count in breakdown.items():
        print(f"  - {spec}: {count}")

def print_diversity_metrics(result_dict, num_subareas, num_spots, species_list, N_target):
    print("\n DIVERSITY METRICS ", flush=True)
    counts = {s: 0 for s in species_list}
    total_planted = 0
    
    for i, spec_name in enumerate(species_list):
        c = 0
        for m in range(num_subareas):
            for g in range(num_spots):
                if result_dict.get(get_var_name(i, m, g)) == 1:
                    c += 1
        counts[spec_name] = c
        total_planted += c

    sum_sq_counts = sum(c**2 for c in counts.values())
    diversity_penalty = sum_sq_counts / (N_target**2)
    
    print(f"Species Counts: {counts}")
    print(f"Diversity Term Value (H1): {diversity_penalty:.4f}")
    
    if total_planted > 0:
        simpson_D = sum((c/total_planted)**2 for c in counts.values())
        shannon_H = sum([- (c/total_planted) * math.log(c/total_planted) for c in counts.values() if c > 0])
        print(f"Simpson Index (D): {simpson_D:.4f}")
        print(f"Shannon Index (H'): {shannon_H:.4f}")

def plot_forest(result_dict, coords, bounds, species_list, patch_w, patch_h, filename):
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    colors_sub = ['#FFDDC1', '#C1FFD7', '#C1E1FF', '#E1C1FF']
    for m, (min_x, max_x, min_y, max_y) in bounds.items():
        rect = patches.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y, 
                                 linewidth=2, edgecolor='black', facecolor=colors_sub[m % 4], alpha=0.4)
        ax.add_patch(rect)

    spec_colors = ['blue', 'red', 'green', 'purple']
    texts = []  
    
    for (m, g), (x, y) in coords.items():
        plt.scatter(x, y, c='lightgray', s=300, alpha=0.5, label='Empty Hole' if m==0 and g==0 else "", zorder=5)
        for i, spec_name in enumerate(species_list):
            if result_dict.get(get_var_name(i, m, g)) == 1:
                plt.scatter(x, y, s=200, c=spec_colors[i % len(spec_colors)], edgecolors='black', zorder=10)
                
                t = plt.text(
                    x, y, spec_name, 
                    fontsize=8, weight='bold', zorder=11,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.8, edgecolor='gray')
                )
                texts.append(t)

    if texts:
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black', lw=0.4))

    plt.title(f"Quantum Complex Gurobi Simulation Output ({len(species_list)} Species)")
    plt.xlim(0, patch_w); plt.ylim(0, patch_h)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.savefig(filename, bbox_inches='tight')
    plt.close() 
    print(f"Plot saved to {filename}")

def export_results_to_csv(filename, result_dict, coords, num_subareas, num_spots, species_list, energy):
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([" OPTIMIZATION SUMMARY "])
        writer.writerow(["Final Energy", f"{energy:.6f}"])
        writer.writerow([])
        writer.writerow(["Location_ID", "X_Coord", "Y_Coord", "Species_Planted"])
        for m in range(num_subareas):
            for g in range(num_spots):
                x, y = coords[(m, g)]
                planted = "Empty"
                for i, spec_name in enumerate(species_list):
                    if result_dict.get(get_var_name(i, m, g)) == 1:
                        planted = spec_name
                writer.writerow([f"Sub_{m}_Spot_{g}", x, y, planted])
# endregion

# region model definition
def solve_gurobi_complex(num_species, num_subareas, num_spots, e_vals, j_matrix, r_val, n_target, n_min, n_max, coords, d_sep, c_penalty, j_decay_rate):
    """
    Constructs and solves the Complex CBQP reflecting Eq. 1 through Eq. 5 using Gurobi
    """
    print("\n[1] Building Complex Gurobi Model...")
    m_model = gp.Model("Reforestation_Complex_Gurobi")
    
    # Solver Limits to prevent infinite hanging
    m_model.setParam('OutputFlag', 1)  # Set to 1 so you can see Gurobi's progress bar
    m_model.setParam('TimeLimit', 60)  # Stop after 60 seconds maximum
    m_model.setParam('MIPGap', 0.05)   # Stop if it mathematically proves the solution is within 5% of perfect
    m_model.setParam('NonConvex', 2)   # Ensure Gurobi translates quadratic terms correctly
    
    # Define Binary Variables: y_{i, m, g}
    y = {}
    for i in range(num_species):
        for m in range(num_subareas):
            for g in range(num_spots):
                var_name = get_var_name(i, m, g)
                y[i, m, g] = m_model.addVar(vtype=GRB.BINARY, name=var_name)
                
    m_model.update()

    obj = gp.QuadExpr()

    # 1. Diversity Term: Sum_i (Count_i)^2 / N_target^2
    for i in range(num_species):
        species_sum = gp.quicksum(y[i, m, g] for m in range(num_subareas) for g in range(num_spots))
        obj += (species_sum * species_sum) / (n_target**2)

    # 2. Ecological Suitability (Linear Cost): (R - E)
    for i in range(num_species):
        for m in range(num_subareas):
            for g in range(num_spots):
                e_val = e_vals[i][m] if isinstance(e_vals[i], (list, np.ndarray)) else e_vals[i]
                obj += (r_val - e_val) * y[i, m, g]

    # 3. Spatial Interactions & Clustering Penalty
    for i in range(num_species):
        for m1 in range(num_subareas):
            for g1 in range(num_spots):
                for j in range(num_species):
                    for m2 in range(num_subareas):
                        for g2 in range(num_spots):
                            v1_name = get_var_name(i, m1, g1)
                            v2_name = get_var_name(j, m2, g2)
                            
                            # Match Qiskit's upper-triangular logic to avoid duplicate pairs
                            if v1_name >= v2_name: 
                                continue
                                
                            x1, y1 = coords[(m1, g1)]
                            x2, y2 = coords[(m2, g2)]
                            dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                            
                            term_weight = 0.0
                            
                            # [Eq 4] Dynamic J Distance Decay
                            dynamic_j = j_matrix[i][j] * math.exp(-j_decay_rate * dist)
                            term_weight += dynamic_j
                                
                            # [Eq 5] Clustering Penalty (Seed Dispersal Constraint)
                            if dist < d_sep:
                                term_weight += c_penalty
                                
                            if term_weight != 0:
                                obj += term_weight * y[i, m1, g1] * y[j, m2, g2]

    # Set Objective
    m_model.setObjective(obj, GRB.MINIMIZE)

    # Constraints
    total_trees = gp.quicksum(y[i, m, g] for i in range(num_species) for m in range(num_subareas) for g in range(num_spots))
    
    # Minimum Quota Constraint
    m_model.addConstr(total_trees >= n_min, name="min_capacity_constraint")
    
    # Global Maximum Capacity Constraint 
    m_model.addConstr(total_trees <= n_max, name="max_capacity_constraint")

    # One tree per location constraint
    for m in range(num_subareas):
        for g in range(num_spots):
            m_model.addConstr(
                gp.quicksum(y[i, m, g] for i in range(num_species)) <= 1, 
                name=f"one_tree_loc_{m}_{g}"
            )

    # Solve
    print("\n[2] Solving via Gurobi Optimizer...")
    m_model.optimize()
    
    best_sample = {}
    fval = None
    
    # Check if we found an optimal solution or hit the time limit with a feasible solution
    if m_model.Status == GRB.OPTIMAL or m_model.Status == GRB.TIME_LIMIT:
        if m_model.SolCount > 0:
            fval = m_model.ObjVal
            for v in m_model.getVars():
                best_sample[v.VarName] = int(round(v.X))
        else:
            print("Time limit reached, but no feasible solution was found.")
    else:
        print("No optimal solution found!")

    return best_sample, fval, n_target
# endregion

# region Modular Execution Wrapper
def run_scenario(scenario_name, num_species, num_subareas, grid_rows, grid_cols, n_target, n_min, n_max, e_vals, j_matrix, r_val=8.0, d_sep=8.0, c_penalty=10.0, j_decay_rate=0.2):
    print("\n" + "="*50)
    print(f" RUNNING SCENARIO: {scenario_name.upper()} ")
    print("="*50)
    
    num_spots = grid_rows * grid_cols
    species_names = [f"Species {chr(65+i)}" for i in range(num_species)]

    # Dynamic Grid Generation
    locations_coords = {}
    subarea_bounds = {}
    quadrants = [
        (0, 20, 0, 20),   # Subarea 0: Bottom-Left
        (20, 40, 0, 20),  # Subarea 1: Bottom-Right
        (0, 20, 20, 40),  # Subarea 2: Top-Left
        (20, 40, 20, 40)  # Subarea 3: Top-Right
    ]
    
    for m in range(num_subareas):
        min_x, max_x, min_y, max_y = quadrants[m % 4]
        subarea_bounds[m] = (min_x, max_x, min_y, max_y)
        x_step = (max_x - min_x) / (grid_cols + 1)
        y_step = (max_y - min_y) / (grid_rows + 1)
        
        spot_id = 0
        for r in range(1, grid_rows + 1):
            for c in range(1, grid_cols + 1):
                locations_coords[(m, spot_id)] = (min_x + c * x_step, min_y + r * y_step)
                spot_id += 1

    # Run Solver 
    sample, energy, _ = solve_gurobi_complex(
        num_species, num_subareas, num_spots, 
        e_vals, j_matrix, r_val, n_target, n_min, n_max,
        locations_coords, d_sep, c_penalty, j_decay_rate
    )

    # Finalize Outputs
    if energy is not None:
        print("\n" + "="*30)
        print("Gurobi Simulation Results")
        print("="*30)
        print(f"Final Energy (Objective): {energy:.4f}")       
        
        decode_planting_plan(sample, num_species, num_subareas, num_spots, species_names)
        print_diversity_metrics(sample, num_subareas, num_spots, species_names, n_target)
        
        safe_name = scenario_name.replace(' ', '_')
        plot_forest(sample, locations_coords, subarea_bounds, species_names, 40, 40, f"Quantum_Gurobi_Complex_{safe_name}.png")
        export_results_to_csv(f"Quantum_Gurobi_Complex_{safe_name}.csv", sample, locations_coords, num_subareas, num_spots, species_names, energy)
# endregion


if __name__ == "__main__":
    # 2 Species Simulation (4 Subareas, 2x2 Grid = 16 locations)
    # Note that N target and Nmin correspond both to the number of trees to be planted.
    run_scenario(
        scenario_name="2_Species_16_Locations",
        num_species=2, num_subareas=4, grid_rows=2, grid_cols=2,
        n_target=12.0, n_max=16.0, n_min=12.0,j_decay_rate=0.2,
        e_vals=np.array([
            [0.3, 0.3, 0.3, 0.3],       
            [0.35, 0.25, 0.33, 0.29]    
        ]),
        j_matrix=np.array([[0.1, -0.2], 
                           [-0.2, 0.1]]),
        d_sep=8,       
        c_penalty=10.0   
    )
            
    # 3 Species Simulation (5 locations for each subarea)
    run_scenario(
        scenario_name="3_Species_20_Locations",
        num_species=3, num_subareas=4, grid_rows=1, grid_cols=5,
        n_target=16.0, n_max=20.0, n_min=16.0,j_decay_rate=0.2,
        e_vals=np.array([
                [0.3, 0.3, 0.3, 0.3],        
                [0.35, 0.3, 0.33, 0.29],    
                [0.33, 0.31, 0.32, 0.29]     
            ]),
        j_matrix=np.array([
                [0.1,  -0.2,  -0.2], 
                [-0.2,  0.1,  0.15], 
                [-0.2, 0.15, 0.1]
            ]),
        d_sep=8.0,      
        c_penalty=10.0   
    )
        
    # 3 Species Simulation (4 Subareas, 8 locations-2x4 Grid = 32 locations)
    run_scenario(
        scenario_name="3_Species_32_Locations",
        num_species=3, num_subareas=4, grid_rows=2, grid_cols=4,
        n_target=26.0, n_max=32.0, n_min=26.0,j_decay_rate=0.2, 
        e_vals=np.array([
            [0.3, 0.3, 0.3, 0.3],        
            [0.35, 0.3, 0.33, 0.29],    
            [0.33, 0.31, 0.32, 0.29]     
        ]),
        j_matrix=np.array([
            [0.1,  -0.2,  -0.2], 
            [-0.2,  0.1,  0.15], 
            [-0.2, 0.15, 0.1]
        ]),
        d_sep=8.0,      
        c_penalty=10.0   
    )