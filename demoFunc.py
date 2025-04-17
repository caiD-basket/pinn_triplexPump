import numpy as np
import torch
from torch import nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader


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


class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
    def mixture_density_derivative(p, bulk_modulus_model, air_dissolution_model,
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
                exp_term = np.exp(-(p_used - p_atm) / beta_L_atm) / beta_L_atm
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
    def forward(self,t0,p0,t,rho,V,m,p,pout, bulk_modulus_model, air_dissolution_model,
                                   rho_L_atm, beta_L_atm, beta_gain, air_fraction,
                                   rho_g_atm, polytropic_index, p_atm, p_crit, p_min):
        theta =0.1
        dpdt = (pout-p0)/(t-t0)
        loss_physics = V * self.mixture_density_derivative(p, bulk_modulus_model, air_dissolution_model,
                                                           rho_L_atm, beta_L_atm, beta_gain, air_fraction,
                                                           rho_g_atm, polytropic_index, p_atm, p_crit, p_min)*dpdt-m/rho
        loss_physics = abs(loss_physics)
        loss_M = abs(pout-p)
        loss = loss_M + theta*loss_physics
        return loss
class TriplexPINN(nn.Module):
    def __init__(self, seq_len, inputs_dim, outputs_dim, layers, scaler_inputs, scaler_targets):
        super(TriplexPINN, self).__init__()
        self.seq_len, self.inputs_dim, self.outputs_dim = seq_len, inputs_dim, outputs_dim
        self.scaler_inputs, self.scaler_targets = scaler_inputs, scaler_targets

def standardize_tensor(data, mode, mean=0, std=1):
    data_2D = data.contiguous().view((-1, data.shape[-1]))  # 转为2D
    if mode == 'fit':
        mean = torch.mean(data_2D, dim=0)
        std = torch.std(data_2D, dim=0)
    data_norm_2D = (data_2D - mean) / (std + 1e-8)
    data_norm = data_norm_2D.contiguous().view((-1, data.shape[-2], data.shape[-1]))
    return data_norm, mean, std
#TODO 确定密度与压力的梯度函数，以及函数里面的参数
data = pd.read_csv('PumpData5.csv')
X = data.drop(['pOut','fault'],axis=1)
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
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
print(X_train_tensor.shape)
# print(y_test_tensor)
# print(X_test_tensor)
# print(y_test_tensor)
class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden_layer = nn.Linear(5,8)
        self.hidden_layer_activation = nn.ReLU()
        self.hidden_to_output_layer = nn.Linear(8,1)
    def forward(self, x):
        x = self.input_to_hidden_layer(x)
        x = self.hidden_layer_activation(x)
        x = self.hidden_to_output_layer(x)
        return x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
settings = {}
batch_size =  8
num_epoch = 50
num_layers  = 5
num_neurons= 64
num_rounds = 1

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

    inputs_dim = 5
    targets_dim = 1
    _, mean_inputs_train, std_inputs_train = standardize_tensor(torch.reshape(inputs_train,(594,1,5)), mode='fit')
    test = torch.reshape(inputs_train, (594, 1, 5))
    train_set = TensorDataset(inputs_train, targets_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers=0,drop_last=True)

    print(train_loader.dataset)

