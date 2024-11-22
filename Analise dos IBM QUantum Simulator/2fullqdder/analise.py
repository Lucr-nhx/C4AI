import json
import os
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Diretório onde os arquivos JSON estão localizados
directory = "C:/Users/lucca/OneDrive/Área de Trabalho/C4AI/Analise dos IBM QUantum Simulator/2fullqdder/"

# Lista de arquivos JSON com o caminho completo
json_files_double_adder = [
    os.path.join(directory, 'job-cwd20c1bhxtg008kb9wg-result.json'),
    os.path.join(directory, 'job-cwdn4dsmptp0008250sg-result.json'),
    os.path.join(directory, 'job-cwdq0hj9r49g0085g79g-result.json'),
    os.path.join(directory, 'job-cwdqkmymptp000825hvg-result.json'),
    os.path.join(directory, 'job-cwdrqny40e00008873w0-result.json'),
    os.path.join(directory, 'job-cvd2yntkmd100082b02g-result.json'),
    os.path.join(directory, 'job-cvj1wqjw5350008xdfbg-result.json'),
    os.path.join(directory, 'job-cvzj1yqf100g008z75ng-result.json'),
    os.path.join(directory, 'job-cwbqvqe9ezk0008923tg-result.json'),
    os.path.join(directory, 'job-cwbtnpyggr6g00890c00-result.json')
]

# Função para processar arquivos JSON e extrair amostras
def process_json_files(files):
    all_samples = []
    for file_path in files:
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                hex_samples = data['results'][0]['data']['c']['samples']
                decimal_samples = [int(sample, 16) for sample in hex_samples]
                all_samples.extend(decimal_samples)
        except FileNotFoundError:
            print(f"Arquivo não encontrado: {file_path}")
        except json.JSONDecodeError:
            print(f"Erro ao decodificar JSON: {file_path}")
    return all_samples

# Processamento das amostras dos arquivos JSON
samples_double_adder = process_json_files(json_files_double_adder)

# Converte as amostras para strings binárias de 6 bits
binary_samples_double_adder = [format(sample, '06b') for sample in samples_double_adder]

# Define o resultado esperado
expected_result_double_adder = "111111"  # Exemplo para (soma1, carry1, após_rx1, soma2, carry2, após_rx2)

# Calcula a frequência de cada saída
binary_sample_counts_double_adder = Counter(binary_samples_double_adder)

# Cria um DataFrame com as frequências
binary_results_double_df = pd.DataFrame(
    binary_sample_counts_double_adder.items(),
    columns=['Binary Output', 'Frequency']
).sort_values(by='Frequency', ascending=False)

# Marca o resultado correto
binary_results_double_df['IsCorrect'] = binary_results_double_df['Binary Output'] == expected_result_double_adder

# Cria o gráfico de frequência com cores ajustadas
colors_double_adder = ['green' if is_correct else 'red' for is_correct in binary_results_double_df['IsCorrect']]

plt.figure(figsize=(14, 8))
plt.bar(binary_results_double_df['Binary Output'], binary_results_double_df['Frequency'], color=colors_double_adder, edgecolor="black")
plt.title("Frequência das Saídas Binárias dos Dois Somadores Completos Acoplados", fontsize=16)
plt.xlabel("Saída Binária", fontsize=14)
plt.ylabel("Frequência", fontsize=14)
plt.xticks(rotation=45, fontsize=10)
plt.legend(["Resultado Correto (Verde)", "Ruído (Vermelho)"], loc="upper right", fontsize=12)
plt.tight_layout()
plt.show()
