import numpy as np
import torch
from torch import nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import optim
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_default_dtype(torch.float64)


class Sin(nn.Module):
    def forward(self, input):
        return torch.sin(input)


torch.cuda.empty_cache()
torch.set_printoptions(precision=10)


def train(num_epoch, batch_size, train_loader, num_slices_train, inputs_val, targets_val,
          model, optimizer, scheduler, criterion):
    num_period = int(num_slices_train / batch_size)
    results_epoch = dict()
    results_epoch['loss_train'] = torch.zeros(num_epoch)
    results_epoch['loss_val'] = torch.zeros(num_epoch)
    for epoch in range(num_epoch):
        model.train()
        results_epoch = dict()
        results_epoch['loss_train'] = torch.zeros(num_epoch)
        results_epoch['loss_val'] = torch.zeros(num_epoch)

        with torch.backends.cudnn.flags(enabled=False):
            for period, (inputs_train_batch, targets_train_batch) in enumerate(train_loader):
                # print(period, inputs_train_batch, targets_train_batch)
                p_pred, P_t_pred = model(inputs=inputs_train_batch)
                results_period = dict()
                results_period['loss_train'] = torch.zeros(num_period)
                results_period['var_P'] = torch.zeros(num_period)
                results_period['loss_physics'] = torch.zeros(num_period)

                loss = criterion(
                    targets_P=targets_train_batch,
                    outputs_P=p_pred,
                    dpdt=P_t_pred,
                    mdot_A=inputs_train_batch[:, 2],
                    V=2 * np.exp(-4),
                    bulk_modulus_model='const',
                    air_dissolution_model='off',
                    rho_L_atm=851.6,
                    beta_L_atm=1.46696e+03,
                    beta_gain=0.2,
                    air_fraction=0.005,
                    rho_g_atm=1.225,
                    polytropic_index=1.0,
                    p_atm=0.101325,
                    p_crit=3,
                    p_min=1

                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                results_period['loss_train'][period] = criterion.loss_M.detach()
                if (epoch + 1) % 1 == 0 and (period + 1) % 1 == 0:  # 每 100 次输出结果
                    print(
                        'Epoch: {}, Period: {}, Loss: {:.5f}, Loss_M: {:.5f}, Loss_physics: {:.5f}'.format(
                            epoch + 1, period + 1, loss.item(), criterion.loss_M.item(), criterion.loss_physics.item()))
        results_epoch['loss_train'][epoch] = torch.mean(results_period['loss_train'])
        model.eval()
        P_pred_val, P_t_pred_val = model(inputs=inputs_val)
        loss_val = criterion(
            targets_P=targets_val,
            outputs_P=P_pred_val,
            dpdt=P_t_pred_val,
            mdot_A=inputs_val[:, 2],
            V=2 * np.exp(-4),
            bulk_modulus_model='const',
            air_dissolution_model='off',
            rho_L_atm=851.6,
            beta_L_atm=1.46696e+03,
            beta_gain=0.2,
            air_fraction=0.005,
            rho_g_atm=1.225,
            polytropic_index=1.0,
            p_atm=0.101325,
            p_crit=3,
            p_min=1
        )
        scheduler.step()
        results_epoch['loss_val'][epoch] = criterion.loss_M.detach()
        return model, results_epoch


class TensorDataset(Dataset):
    # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小

    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def inverse_standardize_tensor(data_norm, mean, std):
    data_norm_2D = data_norm.contiguous().view((-1, data_norm.shape[-1]))  # 转为2D
    data_2D = data_norm_2D * std + mean
    data = data_2D.contiguous().view((-1, data_norm.shape[-2], data_norm.shape[-1]))
    return data


class Neural_Net_LSTM(nn.Module):
    def __init__(self, seq_len, inputs_dim, outputs_dim, hidden_size, num_layers, activation='Sin'):
        super(Neural_Net_LSTM, self).__init__()
        self.seq_len = seq_len
        self.inputs_dim = inputs_dim
        self.outputs_dim = outputs_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=inputs_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        ).double()

        # 输出层
        self.fc = nn.Linear(hidden_size, outputs_dim).double()
        nn.init.xavier_normal_(self.fc.weight)

        # 激活函数和Dropout
        self.activation = Sin() if activation == 'Sin' else nn.Tanh()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # 输入x的形状应为 (batch_size, seq_len, input_features)
        x = x.double() if x.dtype != torch.float64 else x
        x = x.reshape(x.shape[1], self.seq_len, self.inputs_dim)
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).double().to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).double().to(x.device)

        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))  # out形状: (batch_size, seq_len, hidden_size)

        # 取最后一个时间步
        out = out[:, -1, :]

        # 激活和Dropout
        out = self.activation(out)
        out = self.dropout(out)

        # 最终输出
        out = self.fc(out)
        return out.view(-1, self.seq_len, self.outputs_dim)
class Neural_Net(nn.Module):
    def __init__(self, seq_len, inputs_dim, outputs_dim, layers, activation='Sin'):
        super(Neural_Net, self).__init__()
        self.seq_len = seq_len
        self.inputs_dim = inputs_dim
        self.outputs_dim = outputs_dim

        # 强制所有层使用 float64 类型
        self.layers = nn.ModuleList()  # 使用 ModuleList 替代普通 list 以正确注册子模块

        # 第一层
        layer = nn.Linear(inputs_dim, layers[0]).double()  # 创建后立即转为 double
        nn.init.xavier_normal_(layer.weight)
        self.layers.append(layer)

        # 激活函数和 Dropout（无参数，无需修改类型）
        if activation == 'Tanh':
            self.layers.append(nn.Tanh())
        elif activation == 'Sin':
            self.layers.append(Sin())
        self.layers.append(nn.Dropout(p=0.2))

        # 中间层
        for l in range(len(layers) - 1):
            layer = nn.Linear(layers[l], layers[l + 1]).double()  # 转为 double
            nn.init.xavier_normal_(layer.weight)
            self.layers.append(layer)

            if activation == 'Tanh':
                self.layers.append(nn.Tanh())
            elif activation == 'Sin':
                self.layers.append(Sin())
            self.layers.append(nn.Dropout(p=0.2))

        # 输出层
        layer = nn.Linear(layers[-1], outputs_dim).double()  # 转为 double
        nn.init.xavier_normal_(layer.weight)
        self.layers.append(layer)

        # 构建 Sequential
        self.NN = nn.Sequential(*self.layers)

    def forward(self, x):
        # 确保输入转为 double
        x = x.double() if x.dtype != torch.float64 else x

        self.x = x.contiguous().view(-1, self.inputs_dim)
        NN_out_2D = self.NN(self.x)  # 输入已确保为 double
        self.p_pred = NN_out_2D.view(-1, self.seq_len, self.outputs_dim)

        return self.p_pred


class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_M = torch.tensor(0.0, requires_grad=True)
        self.loss_physics = torch.tensor(0.0, requires_grad=True)

    def mixture_density_derivative(self, p, bulk_modulus_model, air_dissolution_model,
                                   rho_L_atm, beta_L_atm, beta_gain, air_fraction,
                                   rho_g_atm, polytropic_index, p_atm, p_crit, p_min):
        # Determine p_used
        p_used = p if p >= p_min else p_min

        # Calculate theta (fraction of air entrained)
        if air_dissolution_model == 'off':
            theta = 1.0
        else:
            if p_used <= p_atm:
                theta = 1.0
            elif p_used >= p_crit:
                theta = 0.0
            else:
                L = p_crit - p_atm
                x = (p_used - p_atm) / L
                theta = 1 - 3 * x ** 2 + 2 * x ** 3

        # Calculate dtheta_dp (derivative of theta)
        if air_dissolution_model == 'off':
            dtheta_dp = 0.0
        else:
            if p_used <= p_atm or p_used >= p_crit:
                dtheta_dp = 0.0
            else:
                L = p_crit - p_atm
                dtheta_dp = 6 * (p_used - p_atm) * (p_used - p_crit) / (L ** 3)

        # Calculate p_denom
        if air_fraction == 0:
            p_denom = 0.0
        else:
            p_denom = (air_fraction / (1 - air_fraction)) * (p_atm / p_used) ** (1 / polytropic_index) * theta

        # Calculate p_ratio
        if air_fraction == 0:
            p_ratio = 0.0
        else:
            if air_dissolution_model == 'off':
                p_ratio = p_denom / (p_used * polytropic_index)
            else:
                term1 = (air_fraction / (1 - air_fraction)) * (p_atm / p_used) ** (1 / polytropic_index)
                term2 = (theta / (p_used * polytropic_index)) - dtheta_dp
                p_ratio = term1 * term2

        # Calculate exp_term based on conditions
        if air_fraction == 0:
            if bulk_modulus_model == 'const':
                exp_term = np.exp((p_used - p_atm) / beta_L_atm) / beta_L_atm
            else:
                base = 1 + beta_gain * (p_used - p_atm) / beta_L_atm
                exponent = (-1 + 1 / beta_gain)
                exp_term = (base ** exponent) / beta_L_atm
        else:
            if bulk_modulus_model == 'const':
                with torch.no_grad():
                    exp_term = np.exp(-(p_used - p_atm) / beta_L_atm) / beta_L_atm
                # exp_term = np.exp(-(p_used - p_atm) / beta_L_atm) / beta_L_atm
            else:
                base = 1 + beta_gain * (p_used - p_atm) / beta_L_atm
                exponent = (-1 - 1 / beta_gain)
                exp_term = (base ** exponent) / beta_L_atm

        # Compute initial mixture density
        rho_mix_init = rho_L_atm + rho_g_atm * (air_fraction / (1 - air_fraction))

        # Final computation of drho_mix_dp
        if air_fraction == 0:
            drho_mix_dp = rho_L_atm * exp_term
        else:
            if bulk_modulus_model == 'const':
                denominator = beta_L_atm * exp_term + p_denom
            else:
                beta_term = 1 + beta_gain * (p_used - p_atm) / beta_L_atm
                denominator = beta_L_atm * exp_term * beta_term + p_denom
            numerator = exp_term + p_ratio
            drho_mix_dp = rho_mix_init * numerator / (denominator ** 2)

        return drho_mix_dp

    def forward(self, targets_P, dpdt, mdot_A, V, outputs_P, bulk_modulus_model, air_dissolution_model,
                rho_L_atm, beta_L_atm, beta_gain, air_fraction,
                rho_g_atm, polytropic_index, p_atm, p_crit, p_min):

        num = targets_P.shape[0]
        theta =1
        loss = torch.tensor(0.0, requires_grad=True)
        loss_M = torch.tensor(0.0, requires_grad=True)
        loss_physics = torch.tensor(0.0, requires_grad=True)
        for i in range(num):
            # print(dpdt.shape)
            if abs(dpdt[0][i][0]) < 1e-12:
                continue

            loss_physics = V * self.mixture_density_derivative(outputs_P[i] * 0.1, bulk_modulus_model,
                                                               air_dissolution_model, rho_L_atm, beta_L_atm, beta_gain,
                                                               air_fraction, rho_g_atm, polytropic_index, p_atm, p_crit,
                                                               p_min) * dpdt[0][i][0] - mdot_A[i]
            # print(f"dpdt:{dpdt.item()},density_der:{self.mixture_density_derivative(outputs_P[i], bulk_modulus_model, air_dissolution_model, rho_L_atm, beta_L_atm, beta_gain, air_fraction,rho_g_atm, polytropic_index, p_atm, p_crit,p_min).item()};Q:{inputs_Q[i].item()}")

            loss_physics = abs(loss_physics)
            loss_M = (outputs_P[i] - targets_P[i])**2
            loss = loss + loss_M + theta * loss_physics
        self.loss_M = loss_M
        self.loss_physics = loss_physics
        return loss


class TriplexPINN(nn.Module):
    def __init__(self, seq_len, inputs_dim, outputs_dim, hidden_size, num_layers, scaler_inputs, scaler_targets):
        super(TriplexPINN, self).__init__()
        self.seq_len, self.inputs_dim, self.outputs_dim = seq_len, inputs_dim, outputs_dim
        self.scaler_inputs, self.scaler_targets = scaler_inputs, scaler_targets

        # self.surrogateNN = Neural_Net(
        #     seq_len=self.seq_len,
        #     inputs_dim=self.inputs_dim,
        #     outputs_dim=self.outputs_dim,
        #     layers=layers
        # )
        self.surrogateNN = Neural_Net_LSTM(
            seq_len=self.seq_len,
            inputs_dim=self.inputs_dim,
            outputs_dim=self.outputs_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation='Sin'
        )

    def forward(self, inputs):
        s = inputs[:, 1:]
        t = inputs[:, 0:1]

        # 标准化处理
        s_norm, _, _ = standardize_tensor(s, mode='transform',
                                          mean=self.scaler_inputs[0][1:],
                                          std=self.scaler_inputs[1][1:])
        t_norm, _, _ = standardize_tensor(t, mode='transform',
                                          mean=self.scaler_inputs[0][0:1],
                                          std=self.scaler_inputs[1][0:1])
        t_norm.requires_grad_(True)

        # 拼接输入并添加序列维度
        combined = torch.cat((s_norm, t_norm), dim=2)

        # LSTM处理
        P_norm = self.surrogateNN(combined)

        # 逆标准化
        P = inverse_standardize_tensor(P_norm,
                                       mean=self.scaler_targets[0],
                                       std=self.scaler_targets[1])

        # 计算梯度
        grad_outputs = torch.ones_like(P)
        P_t = torch.autograd.grad(
            P, t_norm,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]

        return P, P_t


def standardize_tensor(data, mode, mean=0, std=1):
    data_2D = data.contiguous().view((-1, data.shape[-1]))  # 转为2D
    if mode == 'fit':
        mean = torch.mean(data_2D, dim=0)
        std = torch.std(data_2D, dim=0)
    data_norm_2D = (data_2D - mean) / (std + 1e-8)
    data_norm = data_norm_2D.contiguous().view((-1, data.shape[-2], data.shape[-1]))
    return data_norm, mean, std


# TODO 确定密度与压力的梯度函数，以及函数里面的参数
seq_len = 1
data = pd.read_csv('combined_all_425_1.csv')

X = data.drop(['pOut'], axis=1)
Y = data['pOut']
# 按照时间顺序划分训练集、验证集和测试集
# 假设训练集占60%，验证集占20%，测试集占20%
train_size = int(len(X) * 0.6)
val_size = int(len(X) * 0.2)

# 训练集
X_train = X.iloc[:train_size]
y_train = Y.iloc[:train_size]

# 验证集
X_val = X.iloc[train_size:train_size + val_size]
y_val = Y.iloc[train_size:train_size + val_size]

# 测试集
X_test = X.iloc[train_size + val_size:]
y_test = Y.iloc[train_size + val_size:]

# 将数据转换为torch.Tensor格式
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float64)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float64).unsqueeze(1)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float64)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float64).unsqueeze(1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float64)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float64).unsqueeze(1)


# print(y_test_tensor)
# print(X_test_tensor)
# print(y_test_tensor)
class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden_layer = nn.Linear(4, 8)
        self.hidden_layer_activation = nn.ReLU()
        self.hidden_to_output_layer = nn.Linear(8, 1)

    def forward(self, x):
        x = self.input_to_hidden_layer(x)
        x = self.hidden_layer_activation(x)
        x = self.hidden_to_output_layer(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
settings = {}
batch_size = 16
num_epoch = 2000
num_layers = 5
num_neurons = 64
num_rounds = 5

metric_mean = dict()
metric_std = dict()
metric_mean['train'] = np.zeros((1, 1))
metric_mean['val'] = np.zeros((1, 1))
metric_mean['test'] = np.zeros((1, 1))
metric_std['train'] = np.zeros((1, 1))
metric_std['val'] = np.zeros((1, 1))
metric_std['test'] = np.zeros((1, 1))
layers = num_layers * [num_neurons]
metric_rounds = dict()
metric_rounds['train'] = np.zeros(num_rounds)
metric_rounds['val'] = np.zeros(num_rounds)
metric_rounds['test'] = np.zeros(num_rounds)
for round in range(num_rounds):
    inputs = dict()
    targets = dict()

    inputs['train'] = X_train_tensor
    inputs['val'] = X_val_tensor
    inputs['test'] = X_test_tensor

    targets['train'] = y_train_tensor
    targets['val'] = y_val_tensor
    targets['test'] = y_test_tensor
    inputs_dict, targets_dict = inputs, targets

    inputs_train = inputs_dict['train'].to(device)
    inputs_val = inputs_dict['val'].to(device)
    inputs_test = inputs_dict['test'].to(device)
    targets_train = targets_dict['train'].to(device)
    targets_val = targets_dict['val'].to(device)
    targets_test = targets_dict['test'].to(device)

    inputs_dim = 4
    outputs_dim = 1
    num = inputs_train.shape[0]
    _, mean_inputs_train, std_inputs_train = standardize_tensor(torch.reshape(inputs_train, (num, 1, 4)), mode='fit')
    _, mean_targets_train, std_targets_train = standardize_tensor(targets_train, mode='fit')

    num = inputs_train.shape[0]
    test = torch.reshape(inputs_train, (num, 1, 4))
    train_set = TensorDataset(inputs_train, targets_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    model = TriplexPINN(
        seq_len=seq_len,
        inputs_dim=inputs_dim,
        outputs_dim=outputs_dim,
        hidden_size=64,  # LSTM隐藏层大小
        num_layers=2,  # LSTM层数
        scaler_inputs=(mean_inputs_train, std_inputs_train),
        scaler_targets=(mean_targets_train, std_targets_train)
    ).to(device)
    criterion = My_loss()
    params = ([p for p in model.parameters()])
    optimizer = optim.Adam(params, lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.1)
    model, results_epoch = train(
        num_epoch=num_epoch,
        batch_size=batch_size,
        train_loader=train_loader,
        num_slices_train=inputs_train.shape[0],
        inputs_val=inputs_val,
        targets_val=targets_val,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,

    )
    model.eval()
    P_pred_train, P_t_pred_train = model(inputs=inputs_train)
    # RMSPE_train = torch.sqrt(torch.mean((P_pred_train - targets_train) ** 2, dim=1))
    chunk_size = 1024
    RMSPE_train_chunks = []
    for i in range(0, P_pred_train.size(0), chunk_size):
        chunk_pred = P_pred_train[i:i + chunk_size]
        chunk_target = targets_train[i:i + chunk_size]
        rmspe_chunk = torch.sqrt(torch.mean((chunk_pred - chunk_target) ** 2, dim=1))
        RMSPE_train_chunks.append(rmspe_chunk)

    RMSPE_train = torch.cat(RMSPE_train_chunks, dim=0)
    print(f"RMSPE_train:{RMSPE_train}")
    P_pred_val, P_t_pred_val = model(inputs=inputs_val)
    RMSPE_val = torch.sqrt(torch.mean((P_pred_val - targets_val) ** 2, dim=1))
    print(f"RMSPE_val:{RMSPE_val}")
    P_pred_test, P_t_pred_test = model(inputs=inputs_test)
    RMSPE_test = torch.sqrt(torch.mean((P_pred_test - targets_test) ** 2, dim=1))
    print(f"RMSPE_test:{RMSPE_test}")
    # metric_rounds['train'][round] = RMSPE_train.detach().cpu().numpy()
    # metric_rounds['val'][round] = RMSPE_val.detach().cpu().numpy()
    # metric_rounds['test'][round] = RMSPE_test.detach().cpu().numpy()
    # metric_mean['train'] = np.mean(metric_rounds['train'])
    # metric_mean['val'] = np.mean(metric_rounds['val'])
    # metric_mean['test'] = np.mean(metric_rounds['test'])
    # metric_std['train'] = np.std(metric_rounds['train'])
    # metric_std['val'] = np.std(metric_rounds['val'])
    # metric_std['test'] = np.std(metric_rounds['test'])
model.eval()
inputs_test = inputs_dict['test'].to(device)
targets_test = targets_dict['test'].to(device)
P_pred_test, P_t_pred_test = model(inputs=inputs_test)

results = dict()
results['P_true'] = targets_test.detach().cpu().numpy().squeeze()
results['P_pred'] = P_pred_test.detach().cpu().numpy().squeeze()
results['Cycles'] = inputs_test[:, 0].detach().cpu().numpy().squeeze()
results['Epochs'] = np.arange(0, num_epoch)
torch.save(results, 'testTriplex.pth')
