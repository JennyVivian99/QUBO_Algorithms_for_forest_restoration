# Quantum complex simulation
# region libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
import math
from adjustText import adjust_text

# Qiskit
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_algorithms import NumPyMinimumEigensolver
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
    texts = []  # List to hold text objects for collision avoidance
    
    for (m, g), (x, y) in coords.items():
        plt.scatter(x, y, c='lightgray', s=300, alpha=0.5, label='Empty Hole' if m==0 and g==0 else "", zorder=5)
        for i, spec_name in enumerate(species_list):
            if result_dict.get(get_var_name(i, m, g)) == 1:
                plt.scatter(x, y, s=200, c=spec_colors[i % len(spec_colors)], edgecolors='black', zorder=10)
                
                # Create text object with a background box for legibility
                t = plt.text(
                    x, y, spec_name, 
                    fontsize=8, weight='bold', zorder=11,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.8, edgecolor='gray')
                )
                texts.append(t)

    # Automatically shift overlapping labels and draw pointers to their trees
    if texts:
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black', lw=0.4))

    plt.title(f"Complex Quantum Simulation Output ({len(species_list)} Species)")
    plt.xlim(0, patch_w); plt.ylim(0, patch_h)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    # bbox_inches='tight' prevents labels on the edges from getting cut off
    plt.savefig(filename, bbox_inches='tight')
    plt.close() # Free memory after saving
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
def solve_quantum_complex(num_species, num_subareas, num_spots, e_vals, j_matrix, r_val, n_target, n_min, n_max, coords, d_sep, c_penalty, j_decay_rate):
    """
    Constructs the Logical Quadratic Program reflecting Eq. 1 through Eq. 5
    """
    qp = QuadraticProgram("Reforestation_Complex_Quantum_Simulation")
    
    # Define Binary Variables: y_{i, m, g}
    for i in range(num_species):
        for m in range(num_subareas):
            for g in range(num_spots):
                qp.binary_var(name=get_var_name(i, m, g))

    linear = {}
    quadratic = {}
    div_coeff = 1.0 / (n_target**2)

    # Build Objective Terms
    for i in range(num_species):
        for m in range(num_subareas):
            for g in range(num_spots):
                v1 = get_var_name(i, m, g)
                
                # Linear Terms: Diversity base + Environmental Suitability (R - E)
                e_val = e_vals[i][m] if isinstance(e_vals[i], (list, np.ndarray)) else e_vals[i]
                linear[v1] = linear.get(v1, 0) + div_coeff + (r_val - e_val)
                
                for j in range(num_species):
                    for n in range(num_subareas):
                        for g2 in range(num_spots):
                            v2 = get_var_name(j, n, g2)
                            
                            # Skip duplicate pair processing in undirected matrix
                            if v1 >= v2: 
                                continue
                                
                            # Calculate Euclidean Distance for Spatial terms
                            x1, y1 = coords[(m, g)]
                            x2, y2 = coords[(n, g2)]
                            dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                            
                            term_weight = 0.0
                            
                            # [Eq 1] Quadratic Diversity Penalty (Interaction within same species)
                            if i == j:
                                term_weight += (2.0 * div_coeff)
                            
                            # [Eq 4] Dynamic J Distance Decay
                            # Applied for all trees
                            # J_dynamic = J_base * e^(-decay_rate * distance)
                            dynamic_j = j_matrix[i][j] * math.exp(-j_decay_rate * dist)
                            term_weight += dynamic_j
                                
                            # [Eq 5] Clustering Penalty (Seed Dispersal Constraint)
                            # If trees are planted closer than Dsep, apply penalty C
                            if dist < d_sep:
                                term_weight += c_penalty
                                
                            # Add accumulated weight to QUBO
                            if term_weight != 0:
                                quadratic[(v1, v2)] = quadratic.get((v1, v2), 0) + term_weight

    qp.minimize(linear=linear, quadratic=quadratic)


    # Minimum Quota Constraint (Forces the solver to plant trees)
    qp.linear_constraint(
        linear={get_var_name(i, m, g): 1 for i in range(num_species) for m in range(num_subareas) for g in range(num_spots)},
        sense=">=", rhs=n_min, name="min_capacity_constraint")

    # Global Maximum Capacity Constraint (automatically creates Slack variables S_k)
    qp.linear_constraint(
        linear={get_var_name(i, m, g): 1 for i in range(num_species) for m in range(num_subareas) for g in range(num_spots)},
        sense="<=", rhs=n_max, name="capacity_constraint")

    # One tree per location constraint: Sum of species for a specific location <= 1
    for m in range(num_subareas):
        for g in range(num_spots):
            loc_vars = {get_var_name(i, m, g): 1 for i in range(num_species)}
            qp.linear_constraint(linear=loc_vars, sense="<=", rhs=1, name=f"one_tree_loc_{m}_{g}")

    # Convert to Hamiltonian Form
    converter = QuadraticProgramToQubo()
    qubo = converter.convert(qp)
    observable, offset = qubo.to_ising()
    
    print(f"\n ISING HAMILTONIAN ")
    print(f"Total Qubits: {observable.num_qubits}")
    print(f"Pauli Terms: {len(observable)}")

    # Solve
    print("\n Solving via Exact Classical Solver...")
    solver = NumPyMinimumEigensolver()
    optimizer = MinimumEigenOptimizer(solver)
    result = optimizer.solve(qp)
    
    return result, n_target, qp
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
    # Each subarea is now 20x20. Total area is 40x40.
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

    # Calculate Required Qubits
    base_qubits = num_species * num_subareas * num_spots
    slack_qubits = math.ceil(math.log2(n_max)) if n_max > 0 else 0
    total_estimated_qubits = base_qubits + slack_qubits

    print("Qubits estimation report:")
    print(f"Base Variables: {base_qubits} | Slack Qubits: ~{slack_qubits} | Total: {total_estimated_qubits}")
    if total_estimated_qubits > 24:
        print("WARNING: High qubit count! The local NumPy exact solver will likely be slow or run out of memory.")

    # Run Solver (Passing coords, d_sep, and c_penalty now)
    result, _, qp = solve_quantum_complex(
        num_species, num_subareas, num_spots, 
        e_vals, j_matrix, r_val, n_target, n_min, n_max,
        locations_coords, d_sep, c_penalty, j_decay_rate
    )

    # Finalize Outputs
    sample = {k: int(v) for k, v in result.variables_dict.items()}
    energy = result.fval

    decode_planting_plan(sample, num_species, num_subareas, num_spots, species_names)
    print_diversity_metrics(sample, num_subareas, num_spots, species_names, n_target)
    
    safe_name = scenario_name.replace(' ', '_')
    plot_forest(sample, locations_coords, subarea_bounds, species_names, 40, 40, f"Quantum_Complex_{safe_name}.png")
    export_results_to_csv(f"Quantum_Complex_{safe_name}.csv", sample, locations_coords, num_subareas, num_spots, species_names, energy)
# endregion


if __name__ == "__main__":
    # 2 Species Simulation (4 Subareas, 2x2 Grid = 16 locations)
        run_scenario(
                   scenario_name="2_Species_16_Locations",
                   num_species=2, num_subareas=4, grid_rows=2, grid_cols=2,
                   n_target=12.0, n_max=16.0, n_min=12.0,j_decay_rate=0.2,
                   e_vals=np.array([
                       [0.3, 0.3, 0.3, 0.3],       
                       [0.35, 0.25, 0.33, 0.29]    
                   ]),
                   j_matrix=np.array([[0.1, -0.2], # Negative for rewarding
                                       [-0.2, 0.1]]),
                   d_sep=8,       # Seed dispersion minimum distance, increase
                   c_penalty=10.0   # Weight penalty C for breaking d_sep
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