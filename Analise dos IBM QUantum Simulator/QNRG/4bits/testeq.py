import json
import matplotlib.pyplot as plt
from collections import Counter

# Caminho para o arquivo JSON do experimento de 4 bits
json_file_path = "C:/Users/lucca/Downloads/job-cwk4smemptp00085yrz0-result.json"

# Carregar os dados do JSON
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Extrair amostras hexadecimais dos dados JSON
hex_samples = data["results"][0]["data"]["c"]["samples"]

# Converter amostras de hexadecimal para binário de 4 bits e contar frequências
binary_samples = [bin(int(sample, 16))[2:].zfill(4) for sample in hex_samples]
sample_counts = Counter(binary_samples)

# Preparar dados para o gráfico
labels, frequencies = zip(*sample_counts.items())

# Plotar histograma
plt.figure(figsize=(10, 6))
plt.bar(labels, frequencies, color='royalblue')
plt.title("Distribution of 4-Bit Quantum Random Numbers")
plt.xlabel("4-Bit Binary Numbers")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
