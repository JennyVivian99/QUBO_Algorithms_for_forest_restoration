# Gurobi Exact Solver for Complex Simulation
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
def decode_planting_plan(sample, num_subareas, num_spots, species_list):
    print("\n FINAL PLANTING PLAN ", flush=True)
    trees_planted = 0
    breakdown = {s: 0 for s in species_list}
    
    for m in range(num_subareas):
        for g in range(num_spots):
            loc_name = f"Subarea_{m} Spot_{g}"
            found = False
            for s_idx, spec_name in enumerate(species_list):
                var_name = f'y[{s_idx}][{m}][{g}]'
                if sample.get(var_name) == 1:
                    print(f"{loc_name}: Plant {spec_name}")
                    trees_planted += 1
                    breakdown[spec_name] += 1
                    found = True
            if not found:
                print(f"⚪ {loc_name}: -- Empty --")

    print("\n SUMMARY ", flush=True)
    print(f"Total Trees Planted: {trees_planted} of {num_subareas * num_spots}")
    print("Diversity Breakdown:")
    for spec, count in breakdown.items():
        print(f"  - {spec}: {count}")

def print_diversity_metrics(sample, num_subareas, num_spots, species_list, n_target):
    print("\n DIVERSITY METRICS ", flush=True)
    counts = {s: 0 for s in species_list}
    total_planted = 0
    
    for s_idx, spec_name in enumerate(species_list):
        c = 0
        for m in range(num_subareas):
            for g in range(num_spots):
                var_name = f'y[{s_idx}][{m}][{g}]'
                if sample.get(var_name) == 1:
                    c += 1
        counts[spec_name] = c
        total_planted += c

    sum_sq_counts = sum(c**2 for c in counts.values())
    diversity_penalty = sum_sq_counts / (n_target**2)
    
    print(f"Species Counts: {counts}")
    print(f"Diversity Term Value (H1): {diversity_penalty:.4f}")
    
    if total_planted > 0:
        simpson_D = sum((c/total_planted)**2 for c in counts.values())
        shannon_H = sum([- (c/total_planted) * math.log(c/total_planted) for c in counts.values() if c > 0])
        print(f"Simpson Diversity Index (D): {simpson_D:.4f}")
        print(f"Shannon Diversity Index (H'): {shannon_H:.4f}")

def plot_forest(sample, coords, bounds, species_list, patch_w, patch_h, filename):
    print("\n[INFO] Generating Plot...", flush=True)
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    colors_sub = ['#FFDDC1', '#C1FFD7', '#C1E1FF', '#E1C1FF']
    for m, (min_x, max_x, min_y, max_y) in bounds.items():
        rect = patches.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y, 
                                 linewidth=2, edgecolor='black', facecolor=colors_sub[m % 4], alpha=0.4)
        ax.add_patch(rect)

    spec_colors = ['blue', 'red', 'green', 'purple']
    texts = []  # List to hold text objects for collision avoidance
    
    for (m, g), (x, y) in coords.items():
        plt.scatter(x, y, c='lightgray', s=300, alpha=0.5, label='Empty Hole' if m==0 and g==0 else "", zorder=5)
        for s_idx, spec_name in enumerate(species_list):
            var_name = f'y[{s_idx}][{m}][{g}]'
            if sample.get(var_name) == 1:
                plt.scatter(x, y, s=200, c=spec_colors[s_idx % len(spec_colors)], edgecolors='black', zorder=10)
                
                # Create text object with a white background box for legibility
                t = plt.text(
                    x, y, spec_name, 
                    fontsize=8, weight='bold', zorder=11,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.8, edgecolor='gray')
                )
                texts.append(t)

    # Automatically shift overlapping labels and draw pointers to their trees
    if texts:
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black', lw=0.4))

    plt.title(f"Gurobi Complex Simulation Output ({len(species_list)} Species)")
    plt.xlim(0, patch_w); plt.ylim(0, patch_h)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    # bbox_inches='tight' prevents labels on the edges from getting cut off
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {filename}")

def export_results_to_csv(filename, sample, coords, num_subareas, num_spots, species_list, n_target, energy):
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([" OPTIMIZATION SUMMARY "])
        writer.writerow(["Final Energy", f"{energy:.6f}"])
        writer.writerow([])
        writer.writerow(["Location_ID", "X_Coord", "Y_Coord", "Species_Planted"])
        for m in range(num_subareas):
            for g in range(num_spots):
                x, y_coord = coords[(m, g)]
                planted_spec = "Empty"
                for s_idx, spec_name in enumerate(species_list):
                    var_name = f'y[{s_idx}][{m}][{g}]'
                    if sample.get(var_name) == 1:
                        planted_spec = spec_name
                writer.writerow([f"Sub_{m}_Spot_{g}", x, y_coord, planted_spec])
# endregion

# region model definition
def solve_gurobi_complex_with_synergy(num_species, num_subareas, num_spots, e_vals, j_matrix, r_val, n_target, n_min, n_max, coords, d_sep, c_sep, j_decay_rate):
    """
    Constructs and solves the model exactly using Gurobi MIQP (Mixed Integer Quadratic Programming).
    """
    model = gp.Model("Gurobi_Complex_Reforestation")
    # model.setParam('OutputFlag', 0)  # Suppresses the massive Gurobi log stream

    model.setParam('NonConvex', 2) # Forces Gurobi to solve non-convex matrices

    # Turn the console output on
    model.setParam('OutputFlag', 1) 
    
    model.setParam('NonConvex', 2) 
    
    # Add a Time Limit (here 60 seconds)
    model.setParam('TimeLimit', 60)
    
    # Accept an answer if it is within 5% of the theoretical perfection
    model.setParam('MIPGap', 0.05)


    # 1. Define Variables: y[i, m, g]
    y = {}
    for i in range(num_species):
        for m in range(num_subareas):
            for g in range(num_spots):
                y[i, m, g] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{m}_{g}")

    # Initialize quadratic objective
    obj = gp.QuadExpr()

    # 2. Diversity Term: (Sum of y_i)^2 / N_target^2
    div_coeff = 1.0 / (n_target**2)
    for i in range(num_species):
        species_sum = gp.quicksum(y[i, m, g] for m in range(num_subareas) for g in range(num_spots))
        obj += div_coeff * (species_sum * species_sum)
    
    # 3. Suitability Reward: (R - E_i) * y_i
    for i in range(num_species):
        for m in range(num_subareas):
            # Handle if e_vals is a flat list or an array of lists
            e_val = e_vals[i][m] if isinstance(e_vals[i], (list, np.ndarray)) else e_vals[i]
            for g in range(num_spots):
                obj += (r_val - e_val) * y[i, m, g]
    
    # 4. Interaction terms: Dynamic J Matrix & D_sep Clustering
    spots = [(m, g) for m in range(num_subareas) for g in range(num_spots)]
    for idx1, (m1, g1) in enumerate(spots):
        for idx2, (m2, g2) in enumerate(spots):
            if idx1 < idx2:  # Avoid double counting
                dist = math.dist(coords[(m1, g1)], coords[(m2, g2)])
                
                for i in range(num_species):
                    for j in range(num_species):
                        term_weight = 0.0
                        
                        # Dynamic J Distance Decay (Applied when in different subareas)
                        term_weight += j_matrix[i][j] * math.exp(-j_decay_rate * dist)
                            
                        # Separation Penalty C_sep if distance is less than D_sep
                        if dist < d_sep:
                            term_weight += c_sep
                            
                        if term_weight != 0.0:
                            obj += term_weight * y[i, m1, g1] * y[j, m2, g2]

    # Set the objective function to MINIMIZE energy
    model.setObjective(obj, GRB.MINIMIZE)

    # 5. Exact Constraints (being in Gurobi, no slack variables are set)
    
    # Min trees to plant
    model.addConstr(
        gp.quicksum(y[i, m, g] for i in range(num_species) for m in range(num_subareas) for g in range(num_spots)) >= n_min,
        name="Global_Minimum_Quota"
    )

    # Capacity Constraint: Total trees <= N_max
    model.addConstr(
        gp.quicksum(y[i, m, g] for i in range(num_species) for m in range(num_subareas) for g in range(num_spots)) <= n_max,
        name="Global_Capacity"
    )
    
    # Uniqueness Constraint: Max ONE tree per spot
    for m in range(num_subareas):
        for g in range(num_spots):
            model.addConstr(
                gp.quicksum(y[i, m, g] for i in range(num_species)) <= 1,
                name=f"Unique_Spot_{m}_{g}"
            )
    
    # 6. Optimize
    model.optimize()
    
    # Extract results
    sample = {}
    energy = 0
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED] and model.SolCount > 0:
        energy = model.ObjVal # Note the capital 'O' in ObjVal for Gurobi
        for i in range(num_species):
            for m in range(num_subareas):
                for g in range(num_spots):
                    # Use .X to get the value of the variable in the best solution found
                    val = 1 if y[i, m, g].X > 0.5 else 0
                    sample[f"y[{i}][{m}][{g}]"] = val
    else:
        print("Gurobi could not find a valid solution!")

    return sample, energy
# endregion

# region Modular Execution Wrapper
def run_scenario(scenario_name, num_species, num_subareas, grid_rows, grid_cols, n_target,n_min, n_max, e_vals, j_matrix, r_val=8.0, d_sep=2.0, c_penalty=10.0, j_decay_rate=0.2):
    print("\n" + "="*50)
    print(f" RUNNING GUROBI SCENARIO: {scenario_name.upper()} ")
    print("="*50)
    
    num_spots = grid_rows * grid_cols
    species_names = [f"Species {chr(65+i)}" for i in range(num_species)]

    # Dynamic Grid Generation
    locations_coords = {}
    subarea_bounds = {}
    # 20x20 Subareas for a 40x40 total map
    quadrants = [
        (0, 20, 0, 20),   
        (20, 40, 0, 20),  
        (0, 20, 20, 40),  
        (20, 40, 20, 40)  
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
    sample, energy = solve_gurobi_complex_with_synergy(
        num_species, num_subareas, num_spots, 
        e_vals, j_matrix, r_val, n_target, n_min, n_max,
        locations_coords, d_sep, c_penalty, j_decay_rate
    )

    print(f"\n FINAL MINIMUM ENERGY (Objective Value): {energy:.4f}")
    print("-" * 50)

    decode_planting_plan(sample, num_subareas, num_spots, species_names)
    print_diversity_metrics(sample, num_subareas, num_spots, species_names, n_target)
    
    safe_name = scenario_name.replace(' ', '_')
    plot_forest(sample, locations_coords, subarea_bounds, species_names, 40, 40, f"Gurobi_Complex_{safe_name}.png")
    export_results_to_csv(f"Gurobi_Complex_{safe_name}.csv", sample, locations_coords, num_subareas, num_spots, species_names, n_target, energy)
# endregion

if __name__ == "__main__":
    # 2 Species Simulation (4 Subareas, 2x2 Grid = 16 locations)
    run_scenario(
        scenario_name="2_Species_16_Locations",
        num_species=2, num_subareas=4, grid_rows=2, grid_cols=2,
        n_target=12.0, n_min=12.0, n_max=16.0,
        r_val=6.0,
        e_vals=np.array([
            [0.3, 0.3, 0.3, 0.3],       
            [0.35, 0.25, 0.33, 0.29]    
        ]),
        j_matrix=np.array([
            [0.1, -0.2], 
            [-0.2, 0.1]
        ]),
        d_sep=8.0,       
        c_penalty=10.0,
        j_decay_rate=0.2 
    )
        
    # 3 Species Simulation (4 subareas, 5 locations for each subarea = 20 locations)
    run_scenario(
        scenario_name="3_Species_20_Locations",
        num_species=3, num_subareas=4, grid_rows=1, grid_cols=5,
        n_target=16.0, n_min=16.0, n_max=20.0,
        r_val=8.0,
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
    
    # 3 Species Simulation (4 Subareas, 2x4 Grid = 32 locations)
    run_scenario(
        scenario_name="3_Species_32_Locations",
        num_species=3, num_subareas=4, grid_rows=2, grid_cols=4,
        n_target=26.0, n_min=26.0, n_max=32.0,
        r_val=8.0,
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