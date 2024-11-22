from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
import matplotlib.pyplot as plt

# Função para criar um circuito de somador completo (full adder)
def create_full_adder_circuit():
    qc = QuantumCircuit(5, 2)
    # Qubits: A, B, Cin, Sum, Cout
    # Portas para somador completo
    qc.cx(0, 3)
    qc.cx(1, 3)
    qc.ccx(0, 1, 4)
    qc.cx(2, 3)
    qc.ccx(2, 3, 4)
    # Medições
    qc.measure(3, 0)  # Sum
    qc.measure(4, 1)  # Cout
    return qc

# Função para simular o circuito com e sem ruído
def simulate_circuit(qc, noise_model=None):
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator, noise_model=noise_model, shots=1024).result()
    counts = result.get_counts()
    return counts

# Criar o circuito
qc = create_full_adder_circuit()

# Simular sem ruído
ideal_counts = simulate_circuit(qc)

# Definir modelo de ruído
noise_model = NoiseModel()
# Erro de despolarização
error = depolarizing_error(0.01, 1)
# Adicionar erro a todas as portas
noise_model.add_all_qubit_quantum_error(error, ['cx', 'ccx'])

# Simular com ruído
noisy_counts = simulate_circuit(qc, noise_model)

# Plotar resultados
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
plot_histogram(ideal_counts, ax=ax[0], title='Sem Ruído')
plot_histogram(noisy_counts, ax=ax[1], title='Com Ruído')
plt.show()
