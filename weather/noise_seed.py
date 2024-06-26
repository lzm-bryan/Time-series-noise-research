import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 读取CSV数据
df = pd.read_csv('weather.csv', index_col='date', parse_dates=True)

# 选择OT列作为特征和目标列
data = df[['OT']]

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)


# 创建数据集
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back), 0]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)


look_back = 7
X, Y = create_dataset(data_scaled, look_back)

# 划分训练集、验证集和测试集
train_size = int(len(X) * 0.5)
val_size = int(len(X) * 0.2)
test_size = len(X) - train_size - val_size

X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
Y_train, Y_val, Y_test = Y[:train_size], Y[train_size:train_size + val_size], Y[train_size + val_size:]


# 为训练集和验证集添加标签噪声（高斯噪声）
def add_label_noise(labels, noise_ratio, seed):
    if seed is not None:
        np.random.seed(seed)
    noise_std = noise_ratio * labels.min()  # 使用 labels.min() 计算噪声标准差
    noise = np.random.normal(0, noise_std, size=labels.shape)
    labels_noisy = labels + noise
    return labels_noisy


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# 检查是否可以使用 CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        h0 = torch.zeros(1, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        c0 = torch.zeros(1, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        predictions = self.linear(lstm_out[:, -1])
        return predictions


def train_and_evaluate(seed):
    set_seed(seed)
    print(f"Using seed: {seed}")

    noise_ratio = 0.2  # 噪声比例为100%
    Y_train_noisy = add_label_noise(Y_train, noise_ratio, seed)
    Y_val_noisy = add_label_noise(Y_val, noise_ratio, seed)

    # 转换为张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).reshape(-1, look_back, 1)
    Y_train_noisy_tensor = torch.tensor(Y_train_noisy, dtype=torch.float32).reshape(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).reshape(-1, look_back, 1)
    Y_val_noisy_tensor = torch.tensor(Y_val_noisy, dtype=torch.float32).reshape(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).reshape(-1, look_back, 1)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).reshape(-1, 1)

    # 创建数据加载器
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, Y_train_noisy_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val_tensor, Y_val_noisy_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 实例化模型、定义损失函数和优化器
    model = LSTMModel(input_size=1, hidden_layer_size=50, output_size=1).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    epochs = 300
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for i, (batch_X, batch_Y) in enumerate(train_dataloader):
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = loss_function(y_pred, batch_Y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_Y in val_dataloader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                y_pred = model(batch_X)
                loss = loss_function(y_pred, batch_Y)
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'noisebest_model.pth')

        if epoch % 10 == 0:
            print(f'Epoch {epoch} loss: {epoch_loss / len(train_dataloader)}, Validation loss: {val_loss}')

    # 加载最佳模型
    model.load_state_dict(torch.load('noisebest_model.pth'))
    model.eval()
    with torch.no_grad():
        test_predict = model(X_test_tensor.to(device)).cpu().numpy()

    # 反归一化预测结果
    test_predict = scaler.inverse_transform(test_predict)
    Y_test_inv = scaler.inverse_transform(Y_test_tensor.cpu().numpy())

    test_mae = mean_absolute_error(Y_test_inv, test_predict)
    test_mse = mean_squared_error(Y_test_inv, test_predict)

    print(f'Seed {seed}, Test MAE: {test_mae}, Test MSE: {test_mse}')

    return test_mae, test_mse


seeds = [42, 2021, 1234, 5678]
results = []

for seed in seeds:
    result = train_and_evaluate(seed)
    results.append(result)

# 计算平均值和方差
test_mae, test_mse = zip(*results)
mean_mae = np.mean(test_mae)
std_mae = np.std(test_mae)
mean_mse = np.mean(test_mse)
std_mse = np.std(test_mse)

print(f'Average Test MAE: {mean_mae}, Standard Deviation: {std_mae}')
print(f'Average Test MSE: {mean_mse}, Standard Deviation: {std_mse}')

# 可视化结果
fig, ax1 = plt.subplots()

# 绘制MAE
color = 'tab:blue'
ax1.set_xlabel('Seeds')
ax1.set_ylabel('Test MAE', color=color)
ax1.plot(seeds, test_mae, 'o-', color=color, label='Test MAE')
ax1.tick_params(axis='y', labelcolor=color)

# 创建第二个y轴，用于绘制MSE
ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('Test MSE', color=color)
ax2.plot(seeds, test_mse, 'o--', color=color, label='Test MSE')
ax2.tick_params(axis='y', labelcolor=color)

# 添加图例
fig.tight_layout()
fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

# 设置图表标题
plt.title('Test MAE and MSE for Different Seeds')

# 保存并展示图表
plt.savefig('noisetest_metrics_comparison.png')
plt.show()
