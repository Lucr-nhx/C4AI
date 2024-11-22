import torch
import torch.nn as nn
import torch.optim as optim

# Definindo o somador completo
class FullAdderNN(nn.Module):
    def __init__(self):
        super(FullAdderNN, self).__init__()
        self.hidden = nn.Linear(3, 10)  # 3 entradas, 10 neurônios na camada oculta
        self.output_sum = nn.Linear(10, 1)  # saída para 'sum'
        self.output_carry = nn.Linear(10, 1)  # saída para 'carry-out'
        self.sigmoid = nn.Sigmoid()  # ativação

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))  # camada oculta com ativação
        sum_out = self.sigmoid(self.output_sum(x))  # saída 'sum'
        carry_out = self.sigmoid(self.output_carry(x))  # saída 'carry-out'
        return sum_out, carry_out

# Dados de entrada (A, B, carry-in) e saídas esperadas (sum, carry-out)
inputs = torch.tensor([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
], dtype=torch.float32)

# Saídas esperadas: [sum, carry-out]
outputs = torch.tensor([
    [0, 0],
    [1, 0],
    [1, 0],
    [0, 1],
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 1]
], dtype=torch.float32)

# Inicializando a rede neural
model = FullAdderNN()
criterion = nn.MSELoss()  # função de perda
optimizer = optim.Adam(model.parameters(), lr=0.01)  # otimizador

# Treinamento
for epoch in range(10000):
    optimizer.zero_grad()
    sum_out, carry_out = model(inputs)
    loss_sum = criterion(sum_out, outputs[:, 0].unsqueeze(1))
    loss_carry = criterion(carry_out, outputs[:, 1].unsqueeze(1))
    loss = loss_sum + loss_carry
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Testando a rede neural
with torch.no_grad():
    sum_out, carry_out = model(inputs)
    predicted_sum = torch.round(sum_out)
    predicted_carry = torch.round(carry_out)
    print("Predicted Sum:\n", predicted_sum)
    print("Predicted Carry-Out:\n", predicted_carry)
