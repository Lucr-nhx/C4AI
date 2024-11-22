file_path = 'C:/Users/lucca/Downloads/efras48500(2).dat'

# Abrir o arquivo em modo binário e ler o conteúdo
with open(file_path, 'rb') as file:
    binary_data = file.read()

# Converter o conteúdo binário para uma sequência de bits
bit_sequence = ''.join(format(byte, '08b') for byte in binary_data)

# Salvar a sequência de bits em um arquivo de texto
output_file_path = 'C:/Users/lucca/Downloads/4850quantis(2).txt'
with open(output_file_path, 'w') as output_file:
    output_file.write(bit_sequence)

print("Sequência de bits completa salva em:", output_file_path)
