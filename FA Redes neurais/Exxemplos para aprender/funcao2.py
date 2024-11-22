import random
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


input_size = 3  # A, B, Carry-in
hidden_size = 2
output_size = 2  # Soma, Carry-out


weights_input_hidden = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
weights_hidden_output = [[random.uniform(-1, 1) for _ in range(output_size)] for _ in range(hidden_size)]

# Treinamento
def train(inputs, outputs, epochs, learning_rate):
    global weights_input_hidden, weights_hidden_output
    for epoch in range(epochs):
        for input_vector, expected_output in zip(inputs, outputs):
            # Forward pass
            hidden_layer_input = [sum(input_vector[j] * weights_input_hidden[j][i] for j in range(input_size)) for i in range(hidden_size)]
            hidden_layer_output = [sigmoid(x) for x in hidden_layer_input]
            
            output_layer_input = [sum(hidden_layer_output[i] * weights_hidden_output[i][o] for i in range(hidden_size)) for o in range(output_size)]
            output_layer_output = [sigmoid(x) for x in output_layer_input]

            # erro
            errors = [expected_output[o] - output_layer_output[o] for o in range(output_size)]
            
            # Backpropagation 
            d_output = [errors[o] * sigmoid_derivative(output_layer_output[o]) for o in range(output_size)]
            
            # Atualiza pesos 
            for o in range(output_size):
                for i in range(hidden_size):
                    weights_hidden_output[i][o] += learning_rate * d_output[o] * hidden_layer_output[i]

            # Atualiza pesos da camada oculta
            for i in range(hidden_size):
                for j in range(input_size):
                    delta_hidden = sum(d_output[o] * weights_hidden_output[i][o] for o in range(output_size))
                    weights_input_hidden[j][i] += learning_rate * delta_hidden * sigmoid_derivative(hidden_layer_output[i]) * input_vector[j]


inputs = [
    [0, 0, 0],  # A, B, Carry-in
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
]

#[Soma, Carry-out]
outputs = [
    [0, 0],
    [1, 0],
    [1, 0],
    [0, 1],
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 1],
]

# Treinamento da rede
train(inputs, outputs, epochs=100000, learning_rate=0.01)

# Teste da rede 
for input_vector in inputs:
    hidden_layer_input = [sum(input_vector[j] * weights_input_hidden[j][i] for j in range(input_size)) for i in range(hidden_size)]
    hidden_layer_output = [sigmoid(x) for x in hidden_layer_input]
    
    output_layer_input = [sum(hidden_layer_output[i] * weights_hidden_output[i][o] for i in range(hidden_size)) for o in range(output_size)]
    output_layer_output = [sigmoid(x) for x in output_layer_input]
    
    # Tomada de decisão
    rounded_output = [round(x) for x in output_layer_output]
    
    print(f"Input: {input_vector}, Output: {output_layer_output}, Rounded Output: {rounded_output}")
