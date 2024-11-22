import json
import os
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Defina o diretório de trabalho onde seus arquivos JSON estão localizados
directory = "C:/Users/lucca/OneDrive/Área de Trabalho/C4AI/Analise dos IBM QUantum Simulator/3fulladder/"

# Lista de arquivos JSON com o caminho completo
json_files_triple_adder = [
    os.path.join(directory, 'job-cvd3wt3p7drg008kt5yg-result.json'),
    os.path.join(directory, 'job-cvj1mwvp7drg008m8bp0-result.json'),
    os.path.join(directory, 'job-cvzx02gf100g008z8sgg-result.json'),
    os.path.join(directory, 'job-cwb5078bhxtg008sdpjg-result.json'),
    os.path.join(directory, 'job-cwbqvjejzdhg0089wmq0-result.json'),
    os.path.join(directory, 'job-cwd2040ggr6g008ak770-result.json'),
    os.path.join(directory, 'job-cwdn4hj31we000879mh0-result.json'),
    os.path.join(directory, 'job-cwdq0400r6b0008nzjkg-result.json'),
    os.path.join(directory, 'job-cwdqkrz40e0000886xyg-result.json'),
    os.path.join(directory, 'job-cwdrqjp9r49g0085gha0-result.json')
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
samples_triple_adder = process_json_files(json_files_triple_adder)

# Converte as amostras para strings binárias de 9 bits para extrair os resultados dos somadores
binary_samples_triple_adder = [format(sample, '09b') for sample in samples_triple_adder]

# Define o resultado esperado para os três somadores acoplados
expected_result_triple_adder = "111111111"

# Calcula a frequência de cada saída
binary_sample_counts_triple_adder = Counter(binary_samples_triple_adder)

# Cria um DataFrame com as frequências
binary_results_triple_df = pd.DataFrame(
    binary_sample_counts_triple_adder.items(),
    columns=['Binary Output', 'Frequency']
).sort_values(by='Frequency', ascending=False)

# Filtra os top N resultados mais frequentes, agrupando o restante como "Outros"
N = 10  # Número de resultados mais frequentes a exibir
top_results_df = binary_results_triple_df.head(N)
other_results_frequency = binary_results_triple_df['Frequency'][N:].sum()

# Adiciona uma linha para "Outros" usando pd.concat
top_results_df = pd.concat([
    top_results_df,
    pd.DataFrame([{"Binary Output": "Outros", "Frequency": other_results_frequency}])
], ignore_index=True)

# Marca o resultado correto para diferenciação de cor - apenas "111111111" como correto
top_results_df['Is_Correct'] = top_results_df['Binary Output'].apply(lambda x: x == expected_result_triple_adder)

# Cria a lista de cores com base no critério atualizado
colors = ['green' if is_correct else 'blue' if output == "Outros" else 'red'
          for output, is_correct in zip(top_results_df['Binary Output'], top_results_df['Is_Correct'])]

# Plota o gráfico de barras
plt.figure(figsize=(12, 7))
plt.bar(top_results_df['Binary Output'], top_results_df['Frequency'], color=colors, edgecolor="black")
plt.title("Frequência dos Principais Resultados Binários dos Três Somadores Completos Acoplados")
plt.xlabel("Saída Binária")
plt.ylabel("Frequência")
plt.xticks(rotation=45)
plt.legend(["Resultado Correto", "Outros", "Ruído"], loc="upper right")
plt.show()
