import json
import matplotlib.pyplot as plt

# Caminho do arquivo JSON
caminho_arquivo = "C:/Users/lucca/Downloads/job-cwp7m4r60bqg008nyyc0-result.json"  # Insira o caminho correto do seu arquivo JSON aqui

# Função para processar o JSON e gerar o histograma
def gerar_histograma_e_concat(caminho_arquivo, num_bits):
    # Carregar dados do arquivo JSON
    with open(caminho_arquivo, 'r') as f:
        dados = json.load(f)

    # Extrair amostras e converter de hexadecimal para inteiros
    amostras_hex = dados['results'][0]['data']['c']['samples']
    amostras_int = [int(x, 16) for x in amostras_hex]  # Converte hexadecimal para inteiros

    # Converter amostras para valores binários com o número de bits especificado
    formato_binario = '{:0' + str(num_bits) + 'b}'
    amostras_binarias = [formato_binario.format(x) for x in amostras_int]

    # Contagem das frequências de cada valor binário único
    valores_unicos = sorted(set(amostras_binarias))
    frequencias = [amostras_binarias.count(valor) for valor in valores_unicos]

    # Plotar o histograma com o eixo X em valores binários
    plt.figure(figsize=(12, 6))
    plt.bar(valores_unicos, frequencias, edgecolor='black')
    plt.title(f'Distribuição de Frequência das Amostras de {num_bits} bits')
    plt.xlabel(f'Valor de {num_bits} bits (em binário)')
    plt.ylabel('Frequência')
    plt.xticks(rotation=90)  # Rotaciona os valores no eixo X para facilitar a leitura
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'histograma_amostras_{num_bits}bits_binario.png')  # Salva o histograma localmente
    plt.show()

    # Concatenar sequência em uma única string binária para análise no NIST
    sequencia_binaria = ''.join(amostras_binarias)

    # Salvar sequência como um arquivo de texto
    with open(f"sequencia_binaria_{num_bits}bits.txt", "w") as f:
        f.write(sequencia_binaria)

    print(f"Histograma salvo como 'histograma_amostras_{num_bits}bits_binario.png' e sequência binária salva como 'sequencia_binaria_{num_bits}bits.txt'.")

# Exemplo de uso
num_bits = 3  # Insira aqui o número de bits desejado
gerar_histograma_e_concat(caminho_arquivo, num_bits)
