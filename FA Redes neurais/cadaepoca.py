import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


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

# Lista para salvar o valor da perda
loss_values = []

# Lista para salvar as previsões em intervalos de épocas
predictions_sum = []
predictions_carry = []

# Treinamento
for epoch in range(10000):
    optimizer.zero_grad()
    sum_out, carry_out = model(inputs)
    loss_sum = criterion(sum_out, outputs[:, 0].unsqueeze(1))
    loss_carry = criterion(carry_out, outputs[:, 1].unsqueeze(1))
    loss = loss_sum + loss_carry
    loss.backward()
    optimizer.step()

    # Salvar a perda a cada época
    loss_values.append(loss.item())
    
    # Salvar previsões a cada 1000 épocas
    if epoch % 1000 == 0:
        predictions_sum.append(sum_out.detach().numpy())
        predictions_carry.append(carry_out.detach().numpy())
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Testando a rede neural
with torch.no_grad():
    sum_out, carry_out = model(inputs)
    predicted_sum = torch.round(sum_out)
    predicted_carry = torch.round(carry_out)
    print("Predicted Sum:\n", predicted_sum)
    print("Predicted Carry-Out:\n", predicted_carry)

# Plotando a evolução da perda
plt.figure()
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Evolução da Perda durante o Treinamento')
plt.show()

# Visualizando as previsões ao longo das épocas
epochs_to_plot = [0, 1000, 5000, 9000]  # Épocas para exibir
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()

for i, epoch in enumerate(epochs_to_plot):
    axes[i].plot(predictions_sum[epoch // 1000], label='Soma predita')
    axes[i].plot(outputs[:, 0].numpy(), label='Soma esperada', linestyle='--')
    axes[i].set_title(f'Época {epoch} - Soma')
    axes[i].legend()

plt.tight_layout()
plt.show()
