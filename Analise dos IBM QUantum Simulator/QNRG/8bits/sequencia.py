import json
import matplotlib.pyplot as plt

# Caminho do arquivo JSON
#caminho_arquivo = "C:/Users/lucca/Downloads/job-cwka1ed0r6b0008q0a80-result.json"
#job-cwa0n4mggr6g0087trq0-result

caminho_arquivo = "C:/Users/lucca/Downloads/job-cwk4smemptp00085yrz0-result.json"

# Carregar dados do arquivo JSON
with open(caminho_arquivo, 'r') as f:
    dados = json.load(f)

# Extrair amostras e converter de hexadecimal para inteiros
amostras_hex = dados['results'][0]['data']['c']['samples']
amostras_int = [int(x, 16) for x in amostras_hex]  # Converte hexadecimal para inteiros

# Converter amostras para valores binários de 8 bits
amostras_binarias = [format(x, '08b') for x in amostras_int]

# Contagem das frequências de cada valor binário único
valores_unicos = sorted(set(amostras_binarias))
frequencias = [amostras_binarias.count(valor) for valor in valores_unicos]

# Plotar o histograma com o eixo X em valores binários de 8 bits
plt.figure(figsize=(12, 6))
plt.bar(valores_unicos, frequencias, edgecolor='black')
plt.title('Frequency Distribution of 8-bit Samples')
plt.xlabel('8-bit value (in binary)')
plt.ylabel('Frequency')
plt.xticks(rotation=90)  # Rotaciona os valores no eixo X para facilitar a leitura
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('histograma_amostras_8bits_binario.png')  # Salva o histograma localmente
plt.show()

# Concatenar sequência em uma única string binária para análise no NIST
sequencia_binaria = ''.join(amostras_binarias)

# Salvar sequência como um arquivo de texto
with open("sequencia_binaria_8bits20000runs teste.txt", "w") as f:
    f.write(sequencia_binaria)

print("Histograma salvo como 'histograma_amostras_8bits_binario.png' e sequência binária salva como 'sequencia_binaria_8bits20000runs.txt'.")
