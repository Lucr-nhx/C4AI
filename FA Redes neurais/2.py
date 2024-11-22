import torch
import torch.nn as nn
import torch.optim as optim

# Definindo o dataset para o somador de 4 bits
def generate_4bit_data():
    inputs = []
    outputs_sum = []
    outputs_carry = []
    
    for a in range(16):  # Todos os valores possíveis para um número de 4 bits
        for b in range(16):
            for carry_in in range(2):  # Carry-in pode ser 0 ou 1
                # Entrada binária para A, B e Carry-In
                A = [int(x) for x in f'{a:04b}']
                B = [int(x) for x in f'{b:04b}']
                inputs.append(A + B + [carry_in])  # Concatenando os bits de A, B e Carry-In
                
                # Realizando a soma real (A + B + Carry-In)
                total_sum = a + b + carry_in
                
                # Saída da soma (últimos 4 bits)
                sum_result = [int(x) for x in f'{total_sum:05b}']  # 5 bits no total para acomodar carry-out
                
                # Separar o carry-out e a soma
                outputs_carry.append([sum_result[0]])  # Carry-out
                outputs_sum.append(sum_result[1:])  # Soma total (últimos 4 bits)
    
    # Convertendo para tensores
    inputs = torch.tensor(inputs, dtype=torch.float32)
    outputs_sum = torch.tensor(outputs_sum, dtype=torch.float32)
    outputs_carry = torch.tensor(outputs_carry, dtype=torch.float32)
    
    return inputs, outputs_sum, outputs_carry

# Definindo a rede neural
class FullAdder4BitNN(nn.Module):
    def __init__(self):
        super(FullAdder4BitNN, self).__init__()
        self.hidden = nn.Linear(9, 16)  # 8 bits (A+B) + Carry-In como entrada, 16 neurônios na camada oculta
        self.hidden2 = nn.Linear(16, 16)
        self.out_sum = nn.Linear(16, 4)  # 4 saídas para a soma
        self.out_carry = nn.Linear(16, 1)  # 1 saída para o carry-out
    
    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = torch.sigmoid(self.hidden2(x))
        sum_out = torch.sigmoid(self.out_sum(x))  # Saída da soma
        carry_out = torch.sigmoid(self.out_carry(x))  # Saída do carry-out
        return sum_out, carry_out

# Treinamento
def train_model(model, inputs, targets_sum, targets_carry, epochs=10000, lr=0.01):
    criterion = nn.MSELoss()  # Função de perda (Mean Squared Error)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs_sum, outputs_carry = model(inputs)
        
        # Calcular a perda para soma e carry separadamente
        loss_sum = criterion(outputs_sum, targets_sum)
        loss_carry = criterion(outputs_carry, targets_carry)
        loss = loss_sum + loss_carry
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Exibir a perda a cada 1000 épocas
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

# Prevendo os resultados
def predict(model, inputs):
    model.eval()
    with torch.no_grad():
        sum_out, carry_out = model(inputs)
        # Arredondar para 0 ou 1
        sum_out = torch.round(sum_out)
        carry_out = torch.round(carry_out)
        return sum_out, carry_out

# Função principal
if __name__ == '__main__':
    # Gerar dados para o somador de 4 bits
    inputs, outputs_sum, outputs_carry = generate_4bit_data()
    
    # Criar o modelo
    model = FullAdder4BitNN()
    
    # Treinar o modelo
    train_model(model, inputs, outputs_sum, outputs_carry, epochs=10000, lr=0.01)
    
    # Testar o modelo com os mesmos dados
    predicted_sum, predicted_carry = predict(model, inputs)
    
    # Exibir as previsões
    print("Predicted Sum:\n", predicted_sum)
    print("Predicted Carry-Out:\n", predicted_carry)
