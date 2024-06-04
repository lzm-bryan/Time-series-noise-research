import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import ParameterGrid

# 读取CSV数据
df = pd.read_csv('ETTh2.csv', index_col='date', parse_dates=True)

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

print('Y_train', Y_train.shape, Y_train)


# 为训练集和验证集添加标签噪声（高斯噪声）
def add_label_noise(labels, noise_ratio, seed=None):
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size=100, num_layers=2, dropout=0.2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        h0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        c0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        predictions = self.linear(lstm_out[:, -1])
        return predictions


# 定义加权MSE损失函数
class WeightedMSELoss(nn.Module):
    def __init__(self, base_loss=nn.MSELoss()):
        super(WeightedMSELoss, self).__init__()
        self.base_loss = base_loss

    def forward(self, input, target, weight):
        loss = self.base_loss(input, target)
        weighted_loss = loss * weight
        return weighted_loss.mean()


def train_and_evaluate(seed, params):
    set_seed(seed)
    print(f"Using seed: {seed}")

    noise_ratio = 1  # 噪声比例为100%
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

    # 初始化模型、损失函数和优化器
    correction_threshold = params['correction_threshold']
    weight_decay_factor = params['weight_decay_factor']

    model = LSTMModel(input_size=1, hidden_layer_size=100, num_layers=2, dropout=0.2).to(device)
    base_loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 初次训练模型
    epochs = 111
    print(f"Training with params: {params}")
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_Y in train_dataloader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = base_loss_function(y_pred, batch_Y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_Y in val_dataloader:
                    batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                    y_pred = model(batch_X)
                    loss = base_loss_function(y_pred, batch_Y)
                    val_loss += loss.item()

            val_loss /= len(val_dataloader)
            print(f'Epoch {epoch}: Train loss: {epoch_loss / len(train_dataloader)}, Validation loss: {val_loss}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pth')

    # 校正标签并使用加权损失函数重新训练
    model.eval()
    with torch.no_grad():
        train_predict = model(X_train_tensor.to(device)).cpu().numpy()
    Y_train_noisy_np = Y_train_noisy_tensor.cpu().numpy()

    weights = np.ones_like(Y_train_noisy_np)
    for i in range(len(Y_train_noisy_np)):
        if np.abs(Y_train_noisy_np[i] - train_predict[i]) > correction_threshold:
            weights[i] = 0.5

    weights = torch.tensor(weights, dtype=torch.float32).reshape(-1, 1).to(device)

    train_dataset_corrected = TensorDataset(X_train_tensor,
                                            torch.tensor(Y_train_noisy_np, dtype=torch.float32).reshape(-1, 1), weights)
    train_dataloader_corrected = DataLoader(train_dataset_corrected, batch_size=batch_size, shuffle=True)

    model = LSTMModel(input_size=1, hidden_layer_size=100, num_layers=2, dropout=0.2).to(device)
    weighted_loss_function = WeightedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_iterations = 3
    for iteration in range(num_iterations):
        best_val_loss = float('inf')  # 在每次迭代校准和训练之前重新初始化best_val_loss
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for i, (batch_X, batch_Y, batch_weights) in enumerate(train_dataloader_corrected):
                batch_X, batch_Y, batch_weights = batch_X.to(device), batch_Y.to(device), batch_weights.to(device)

                optimizer.zero_grad()
                y_pred = model(batch_X)
                loss = weighted_loss_function(y_pred, batch_Y, batch_weights)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if epoch % 10 == 0 or epoch == epochs - 1:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_Y in val_dataloader:
                        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                        y_pred = model(batch_X)
                        loss = base_loss_function(y_pred, batch_Y)
                        val_loss += loss.item()

                val_loss /= len(val_dataloader)
                print(
                    f'Iteration {iteration}, Epoch {epoch}: Train loss: {epoch_loss / len(train_dataloader_corrected)}, Validation loss: {val_loss}')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), 'best_model.pth')

        model.load_state_dict(torch.load('best_model.pth'))
        model.eval()
        with torch.no_grad():
            train_predict = model(X_train_tensor.to(device)).cpu().numpy()

        for i in range(len(Y_train_noisy_np)):
            if np.abs(Y_train_noisy_np[i] - train_predict[i]) > correction_threshold:
                Y_train_noisy_np[i] = train_predict[i]
                weights[i] = weights[i] * weight_decay_factor

        train_dataset_corrected = TensorDataset(X_train_tensor,
                                                torch.tensor(Y_train_noisy_np, dtype=torch.float32).reshape(-1, 1),
                                                weights)
        train_dataloader_corrected = DataLoader(train_dataset_corrected, batch_size=batch_size, shuffle=True)

    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    with torch.no_grad():
        test_predict = model(X_test_tensor.to(device)).cpu().numpy()

    test_predict = scaler.inverse_transform(test_predict)
    Y_test_inv = scaler.inverse_transform(Y_test_tensor.cpu().numpy())

    test_mae = mean_absolute_error(Y_test_inv, test_predict)
    test_mse = mean_squared_error(Y_test_inv, test_predict)

    print(f'Seed {seed}, Test MAE: {test_mae}, Test MSE: {test_mse}')

    return test_mae, test_mse


seeds = [42, 2021, 1234, 5678]
results = []

# 使用最佳参数训练模型
best_params = {'correction_threshold': 0.1, 'weight_decay_factor': 0.9}

for seed in seeds:
    result = train_and_evaluate(seed, best_params)
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
plt.savefig('test_metrics_comparison.png')
plt.show()
