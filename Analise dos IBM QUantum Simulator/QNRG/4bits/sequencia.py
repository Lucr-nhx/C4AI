import json

# Caminho do arquivo JSON com os dados
json_file_path = "C:/Users/lucca/Downloads/job-cwnwzy75v39g008gwnr0-result.json"
#job-cwk4smemptp00085yrz0-resul
#job-cwk4smemptp00085yrz0-result
# Carregar os dados do JSON
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Extrair amostras hexadecimais dos dados JSON e converter para binário de 4 bits
hex_samples = data["results"][0]["data"]["c"]["samples"]
binary_samples = [bin(int(sample, 16))[2:].zfill(4) for sample in hex_samples]

# Concatenar todas as amostras em uma única sequência de bits
bit_sequence = ''.join(binary_samples)

# Salvar a sequência em um arquivo de texto para uso com o NIST Test Suite
with open("sequencia_4bits_nist_20000.txt", "w") as output_file:
    output_file.write(bit_sequence)

print("Sequência de bits salva em 'sequencia_bits_nist.txt'")
