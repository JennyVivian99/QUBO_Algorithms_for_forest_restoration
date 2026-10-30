# Quantum simple simulation
# region libraries
# import libraries
import numpy as np
import matplotlib.pyplot as plt
import csv
from adjustText import adjust_text

# Qiskit
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.optimizers import SPSA
#endregion

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
    # List to hold text objects for collision prevention
    texts = []

    # Plot trees
    for g_idx, (x, y) in enumerate(coordinates):
        for s_idx, spec_name in enumerate(species):
            var_name = f'y[{g_idx}][{s_idx}]'
            if sample.get(var_name, 0) == 1:
                color = colors[s_idx % len(colors)]
                plt.scatter(x, y, c=color, s=200, edgecolors='black', label=spec_name)

                # Append text with a white background box for readability
                t = plt.text(
                    x, y, spec_name, 
                    fontsize=9, weight='bold', zorder=11,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray')
                )
                texts.append(t)

    # Automatically shift labels to avoid overlapping
    if texts:
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))

    plt.title(f"Optimized Reforestation Map (Qiskit Simulation) {num_species}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save plot with bbox_inches='tight' to prevent labels from cutting off at borders
    filename = f"Quantum_simple_simulation_output_{num_species.replace(' ', '_')}.png"
    plt.savefig(filename, bbox_inches='tight')
    print(f"Plot saved as '{filename}'. Check your project folder.")
    
    # Legend fix
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    # plt.show() # Uncomment if running locally with a display

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
            if sample.get(var_name, 0) == 1:
                planted_spec = spec_name
                counts[spec_name] += 1
                total_planted += 1
        plan_rows.append([loc_name, x, y, planted_spec])


    # Export to .csv
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

def visualize_quantum_circuit(qp, reps=1, num_species="try"):
    """
    Generates and draws the QAOA Quantum Circuit for the given problem.
    """
    print("\n Generating Quantum Circuit Diagram...")

    # Convert the problem to an Ising Hamiltonian (Pauli Operator)
    # Add the necessary slack variables and penalties
    converter = QuadraticProgramToQubo()
    qubo = converter.convert(qp)
    cost_operator, _ = qubo.to_ising()
    
    print(f" Number of Qubits: {cost_operator.num_qubits}")
    print(f" Circuit Depth (Repetitions): {reps}")

    # Create the QAOA Ansatz
    # QAOA creates a circuit that alternates between the Cost Operator and a Mixer
    ansatz = QAOAAnsatz(cost_operator, reps=reps)

    # Decompose to see actual gates (CNOT, RZ, H)
    # Without decomposition, it would be seen just big blocks labeled "QAOA"
    real_circuit = ansatz.decompose(reps=3) 

    # Draw the circuit
    # 'mpl' uses Matplotlib to make it look nice (requires pylatexenc)
    # 'fold=-1' prevents the circuit from wrapping, making it a long horizontal scroll
    diagram = real_circuit.draw(output='mpl', fold=-1, style='iqp')
    
    # Save
    filename = f"IBM_Quantum_Circuit_Gates_{num_species.replace(' ', '_')}.png"
    diagram.savefig(filename)
    print(f"[SUCCESS] Circuit diagram saved as '{filename}'")
    
    # Return the circuit object if you want to inspect it further
    return real_circuit
#endregion


# region qiskit model definition, and exact solver. Here set the hyperparameters
def solve_with_qiskit(locations, species, neighbors):
    """
    Constructs the Quadratic Program and solves it using an Exact Solver 
    (Quantum-Ready Formulation).
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
    
    print("\n[1] Building Qiskit Quadratic Program...")
    qp = QuadraticProgram("Forest_Restoration")

    # Define variables
    # y[g][i] = 1 if species i is planted at location g
    for g_idx in range(len(locations)):
        for s_idx in range(len(species)):
            qp.binary_var(name=f"y[{g_idx}][{s_idx}]")

    # Indices helper
    G_range = range(len(locations))
    S_range = range(len(species))

    # Build objectives (Terms)
        # Collect linear and quadratic coefficients to pass to qp.minimize()
    linear_terms = {}
    quadratic_terms = {}

    # Diversity Term: Sum_i (Count_i)^2 / N_target^2
    # (y_i1 + y_i2 + ...)^2 = sum(y^2) + sum(y_a * y_b)
    # Since y is binary, y^2 = y.
    inv_N2 = 1.0 / (N_target**2)
    
    for s_idx in S_range:
        for g1 in G_range:
            v1 = f"y[{g1}][{s_idx}]"
            # Diagonal term (from squaring the sum)
            quadratic_terms[(v1, v1)] = quadratic_terms.get((v1, v1), 0) + inv_N2
            
            for g2 in G_range:
                if g1 < g2:
                    v2 = f"y[{g2}][{s_idx}]"
                    # Cross terms (2 * coeff)
                    quadratic_terms[(v1, v2)] = quadratic_terms.get((v1, v2), 0) + (2 * inv_N2)

    # Ecological Suitability (Linear Cost)
    # Cost = R - E[species]
    for g_idx in G_range:
        for s_idx in S_range:
            var_name = f"y[{g_idx}][{s_idx}]"
            cost = R - E[species[s_idx]]
            linear_terms[var_name] = linear_terms.get(var_name, 0) + cost

    # Clustering / Neighbor Interactions
    for (loc_a, loc_b) in neighbors:
        # Reward any neighbors: -D * Sum(y_a) * Sum(y_b)
        for s_i in S_range:
            for s_j in S_range:
                va = f"y[{loc_a}][{s_i}]"
                vb = f"y[{loc_b}][{s_j}]"
                quadratic_terms[(va, vb)] = quadratic_terms.get((va, vb), 0) - D
        
        # Penalty for proximity of trees of the same species: (B + D) * Sum(y_a_i * y_b_i)
        for s_idx in S_range:
            va = f"y[{loc_a}][{s_idx}]"
            vb = f"y[{loc_b}][{s_idx}]"
            quadratic_terms[(va, vb)] = quadratic_terms.get((va, vb), 0) + (B + D)

    # Add objective to the model
    qp.minimize(linear=linear_terms, quadratic=quadratic_terms)

    # Add constraints
    # Global Capacity Constraint: Sum(all trees) <= N_max
    # Qiskit handles the creation of slack variables automatically here.
    all_vars = {f"y[{g}][{s}]": 1 for g in G_range for s in S_range}
    qp.linear_constraint(linear=all_vars, sense='LE', rhs=N_max, name="Global_Capacity")

    # One tree per location: Sum_species(y_g) <= 1
    for g_idx in G_range:
        loc_vars = {f"y[{g_idx}][{s}]": 1 for s in S_range}
        qp.linear_constraint(linear=loc_vars, sense='LE', rhs=1, name=f"OneTree_Loc_{g_idx}")

    # Convert to Hamiltonian, Pauli form
    print("\n Converting simple quantum to Ising Hamiltonian...")
    # This converter turns constraints into penalty terms (Lagrange multipliers)
    converter = QuadraticProgramToQubo()
    qubo = converter.convert(qp)
    
    # Get Pauli Operator (Ising Hamiltonian)
    H_op, offset = qubo.to_ising()
    
    print(f"Total Qubits: {H_op.num_qubits}")
    print(f"Hamiltonian Offset: {offset:.4f}")
    print("Sample Pauli Terms (first 5):")
    print(H_op[:5])

    # Solve
    print("\n Solving via Exact Classical Solver...")
    solver = NumPyMinimumEigensolver()
    
    # Run the optimization
    optimizer = MinimumEigenOptimizer(solver)
    result = optimizer.solve(qp)
    return result, N_target, qp


if __name__ == "__main__":
    
    # region 2 species, 4 locations
    print("\n" + "="*50)
    print(" RUNNING 2 SPECIES / 4 LOCATIONS ")
    print("="*50)   
    # here define the variables value
    # Setup data
    locations = ['Loc_0', 'Loc_1', 'Loc_2', 'Loc_3'] # consider n locations
    species = ['Species_A', 'Species_B'] # consider n species
    neighbors = [(0, 1), (1, 2), (2,3), (3,0)] # disposition of the location (which is near to the others)
    coords = [(0,0), (0,1), (1,1), (1,0)]  # coordinates of the locations

    # Run Qiskit Solver
    result, N_target, qp = solve_with_qiskit(locations, species, neighbors)

    # Process Results
    # Convert result dictionary to standard python dict
    best_sample = {k: int(v) for k, v in result.variables_dict.items()}

    print("\n" + "="*30)
    print("Qiskit Simulation Results")
    print("="*30)
    print(f"Final Energy (Objective): {result.fval:.4f}")
    print("\n Active Variables (set to 1):")
    
    # Filter for logic variables (starting with 'y')
    for var_label in sorted(best_sample.keys()):
        if var_label.startswith('y') and best_sample[var_label] == 1:
            print(f"  • {var_label} = 1")
    # Qiskit generates internal slack variables for inequalities, which are here hided for clarity.

    # Decode and visualize
    decode_planting_plan(best_sample, locations, species)
    # Pass the seed to the visualization functions
    visualize_quantum_circuit(qp,num_species="2 species")
    
    # Plot
    try:
        plot_forest(best_sample, locations, species, coords, num_species="2 species")
    except Exception as e:
        print(f"Could not plot: {e}")
    
    # Metrics & Export
    print_diversity_metrics(best_sample, locations, species, N_target)
    csv_filename = "IBM_Quantum_simple_simulation_results_2Species.csv"
    export_results_to_csv(csv_filename, best_sample, locations, species, coords)
#endregion

# region 3 species, 5 locations (Optimized scale to prevent exact solver memory crash)
    print("\n" + "="*50)
    print(" RUNNING 3 SPECIES / 5 LOCATIONS ")
    print("="*50)
    
    locations_3 = ['Loc_0', 'Loc_1', 'Loc_2', 'Loc_3', 'Loc_4']
    species_3 = ['Species_A', 'Species_B', 'Species_C']
    neighbors_3 = [(0, 1), (1, 2), (2,3), (3,4), (4,0)]
    coords_3 = [(0,0), (0,1), (1,1), (1,0), (0.5, 1.5)]

    result_3, N_target_3, qp_3 = solve_with_qiskit(locations_3, species_3, neighbors_3)
    best_sample_3 = {k: int(v) for k, v in result_3.variables_dict.items()}

    print("\n" + "="*30)
    print("Qiskit Simulation Results")
    print("="*30)
    print(f"Final Energy (Objective): {result_3.fval:.4f}")
    print("\n Active Variables (set to 1):")
    
    for var_label in sorted(best_sample_3.keys()):
        if var_label.startswith('y') and best_sample_3[var_label] == 1:
            print(f"  • {var_label} = 1")
            
    decode_planting_plan(best_sample_3, locations_3, species_3)
    visualize_quantum_circuit(qp_3, num_species="3 species")
    
    try:
        plot_forest(best_sample_3, locations_3, species_3, coords_3, num_species="3 species")
    except Exception as e:
        print(f"Could not plot: {e}")
    
    print_diversity_metrics(best_sample_3, locations_3, species_3, N_target_3)
    csv_filename_3 = "IBM_Quantum_simple_simulation_results_3Species.csv"
    export_results_to_csv(csv_filename_3, best_sample_3, locations_3, species_3, coords_3)
    #endregion


# region 3 species, 8 locations
    print("\n" + "="*50)
    print(" RUNNING 3 SPECIES / 8 LOCATIONS ")
    print("="*50)
    # here define the variables value    
    # Setup data
    locations = ['Loc_0', 'Loc_1', 'Loc_2', 'Loc_3', "Loc_4",'Loc_5', 'Loc_6', 'Loc_7'] # consider n locations
    species = ['Species_A', 'Species_B', 'Species_C'] # consider n species
    neighbors = [(0, 1), (1, 2), (2,3), (3,0), (3,4), (4,5), (5,2),(5,6), (6,7), (7,4)] # disposition of the location (which is near to the others)
    coords = [(0,0), (0,1), (1,1), (1,0), (1,2), (0,2), (1,3), (0,3)]  # coordinates of the locations

    # Run Qiskit Solver
    result, N_target, qp = solve_with_qiskit(locations, species, neighbors)

    # Process Results
    # Convert result dictionary to standard python dict
    best_sample = {k: int(v) for k, v in result.variables_dict.items()}

    print("Qiskit Simulation Results")
    print(f"Final Energy (Objective): {result.fval:.4f}")
    print("\n Active Variables (set to 1):")
    
    # Filter for logic variables (starting with 'y')
    for var_label in sorted(best_sample.keys()):
        if var_label.startswith('y') and best_sample[var_label] == 1:
            print(f"  • {var_label} = 1")
            
    # Qiskit generates internal slack variables for inequalities, which are here hided for clarity.

    # Decode and visualize
    decode_planting_plan(best_sample, locations, species)
    visualize_quantum_circuit(qp, num_species="3 species")
    
    # Plot
    try:
        plot_forest(best_sample, locations, species, coords, num_species="3 species")
    except Exception as e:
        print(f"Could not plot: {e}")
    
    # Metrics & Export
    print_diversity_metrics(best_sample, locations, species, N_target)
    csv_filename = "IBM_Quantum_simple_simulation_results3species.csv"
    export_results_to_csv(csv_filename, best_sample, locations, species, coords)
#endregion