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


class Neural_Net(nn.Module):
    def __init__(self, seq_len, inputs_dim, outputs_dim, layers, activation='Sin'):
        super(Neural_Net, self).__init__()
        self.seq_len = seq_len
        self.inputs_dim = inputs_dim
        self.outputs_dim = outputs_dim

        # RNN layer
        self.rnn = nn.RNN(
            input_size=inputs_dim,
            hidden_size=layers[0],
            num_layers=len(layers),
            batch_first=True
        ).double()

        # Output layer
        self.fc = nn.Linear(layers[-1], outputs_dim).double()
        nn.init.xavier_normal_(self.fc.weight)

        # Activation function
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sin':
            self.activation = Sin()

        # Dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # Ensure input is double precision
        x = x.double() if x.dtype != torch.float64 else x

        # Reshape input for RNN
        x = x.contiguous().view(-1, self.seq_len, self.inputs_dim)

        # RNN forward pass
        rnn_out, _ = self.rnn(x)

        # Apply activation and dropout
        rnn_out = self.activation(rnn_out)
        rnn_out = self.dropout(rnn_out)

        # Final output layer
        output = self.fc(rnn_out)

        return output


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
        theta = 1.9
        loss = torch.tensor(0.0, requires_grad=True)
        loss_M = torch.tensor(0.0, requires_grad=True)
        loss_physics = torch.tensor(0.0, requires_grad=True)
        for i in range(num):
            if abs(dpdt[i][0]) < 1e-12:
                continue

            loss_physics = V * self.mixture_density_derivative(outputs_P[i] * 0.1, bulk_modulus_model,
                                                               air_dissolution_model, rho_L_atm, beta_L_atm, beta_gain,
                                                               air_fraction, rho_g_atm, polytropic_index, p_atm, p_crit,
                                                               p_min) * dpdt[i][0] - mdot_A[i]
            # print(f"dpdt:{dpdt.item()},density_der:{self.mixture_density_derivative(outputs_P[i], bulk_modulus_model, air_dissolution_model, rho_L_atm, beta_L_atm, beta_gain, air_fraction,rho_g_atm, polytropic_index, p_atm, p_crit,p_min).item()};Q:{inputs_Q[i].item()}")

            loss_physics = abs(loss_physics)
            loss_M = abs(outputs_P[i] - targets_P[i])
            loss = loss + loss_M + theta * loss_physics
        self.loss_M = loss_M
        self.loss_physics = loss_physics
        return loss


class TriplexPINN(nn.Module):
    def __init__(self, seq_len, inputs_dim, outputs_dim, layers, scaler_inputs, scaler_targets):
        super(TriplexPINN, self).__init__()
        self.seq_len, self.inputs_dim, self.outputs_dim = seq_len, inputs_dim, outputs_dim
        self.scaler_inputs, self.scaler_targets = scaler_inputs, scaler_targets

        self.surrogateNN = Neural_Net(
            seq_len=self.seq_len,
            inputs_dim=self.inputs_dim,
            outputs_dim=self.outputs_dim,
            layers=layers
        )

    def forward(self, inputs):
        # inputs = inputs.unsqueeze(1)

        s = inputs[:, 1:]
        t = inputs[:, 0:1]
        # print(f"s.shape:{s.shape}")
        # print(f"t.shape:{t.shape}")
        s_norm, _, _ = standardize_tensor(s, mode='transform', mean=self.scaler_inputs[0][1:],
                                          std=self.scaler_inputs[1][1:])
        t.requires_grad_(True)
        t_norm, _, _ = standardize_tensor(t, mode='transform', mean=self.scaler_inputs[0][0:1],
                                          std=self.scaler_inputs[1][0:1])
        t_norm.requires_grad_(True)

        P_norm = self.surrogateNN(x=torch.cat((s_norm, t_norm), dim=2))
        P = inverse_standardize_tensor(P_norm, mean=self.scaler_targets[0], std=self.scaler_targets[1])
        grad_outputs = torch.ones_like(P)

        P_t = torch.autograd.grad(
            P, t,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
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
        layers=layers,
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

     
    # Calculate metrics for training set
    def calculate_metrics_in_batches(predictions, targets, batch_size=1024):
        """分批计算评估指标以节省内存"""
        total_samples = predictions.size(0)
        rmse_sum = 0.0
        mae_sum = 0.0
        mape_sum = 0.0
        count = 0

        # 分批计算
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            pred_batch = predictions[i:end_idx]
            target_batch = targets[i:end_idx]

            # 计算批次的评估指标
            batch_rmse = torch.sqrt(torch.mean((pred_batch - target_batch) ** 2))
            batch_mae = torch.mean(torch.abs(pred_batch - target_batch))

            # 避免除以零
            non_zero_indices = target_batch != 0
            if torch.any(non_zero_indices):
                batch_mape = torch.mean(torch.abs((pred_batch[non_zero_indices] - target_batch[non_zero_indices]) /
                                                  target_batch[non_zero_indices])) * 100
            else:
                batch_mape = torch.tensor(0.0, device=pred_batch.device)

            # 累加
            batch_size_actual = end_idx - i
            rmse_sum += batch_rmse * batch_size_actual
            mae_sum += batch_mae * batch_size_actual
            mape_sum += batch_mape * batch_size_actual
            count += batch_size_actual

        # 计算平均值
        rmse = rmse_sum / count
        mae = mae_sum / count
        mape = mape_sum / count

        return rmse, mae, mape


    RMSE_train, MAE_train, MAPE_train = calculate_metrics_in_batches(P_pred_train, targets_train)
    print(f"RMSE_train:{RMSE_train}")
    print(f"MAE_train:{MAE_train}")
    print(f"MAPE_train:{MAPE_train}")

    # Calculate metrics for validation set
    P_pred_val, P_t_pred_val = model(inputs=inputs_val)
    RMSE_val, MAE_val, MAPE_val = calculate_metrics_in_batches(P_pred_val, targets_val)
    print(f"RMSE_val:{RMSE_val}")
    print(f"MAE_val:{MAE_val}")
    print(f"MAPE_val:{MAPE_val}")

    # Calculate metrics for test set
    P_pred_test, P_t_pred_test = model(inputs=inputs_test)
    RMSE_test, MAE_test, MAPE_test = calculate_metrics_in_batches(P_pred_test, targets_test)
    print(f"RMSE_test:{RMSE_test}")
    print(f"MAE_test:{MAE_test}")
    print(f"MAPE_test:{MAPE_test}")


    def calculate_r2_in_batches(predictions, targets, batch_size=1024):
        """分批计算R²以节省内存"""
        # 确保输入是一维的
        if predictions.dim() > 1:
            predictions = predictions.squeeze()
        if targets.dim() > 1:
            targets = targets.squeeze()

        total_samples = predictions.size(0)
        sum_squared_error = 0.0

        # 计算目标值的均值
        target_mean_sum = 0.0
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            target_batch = targets[i:end_idx]
            target_mean_sum += torch.sum(target_batch)

        target_mean = target_mean_sum / total_samples

        # 计算SSE和SST
        sum_squared_error = 0.0
        sum_squared_total = 0.0

        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            pred_batch = predictions[i:end_idx]
            target_batch = targets[i:end_idx]

            # 累加误差平方和
            sum_squared_error += torch.sum((target_batch - pred_batch) ** 2)

            # 累加总平方和
            sum_squared_total += torch.sum((target_batch - target_mean) ** 2)

        # 避免分母接近零的情况
        epsilon = 1e-10
        if sum_squared_total < epsilon:
            print(f"Warning: sum_squared_total is very small: {sum_squared_total}")
            return torch.tensor(0.0)  # 返回0而不是可能的大负值

        # 打印中间值进行诊断
        print(f"SSE: {sum_squared_error.item()}, SST: {sum_squared_total.item()}")

        # 计算R²
        r2 = 1 - (sum_squared_error / sum_squared_total)

        return r2


    # 使用分批计算R²
    # 替换原始的R²计算
    R2_train = calculate_r2_in_batches(P_pred_train, targets_train)
    print(f"R-squared (train): {R2_train.item():.4f}")

    R2_val = calculate_r2_in_batches(P_pred_val, targets_val)
    print(f"R-squared (val): {R2_val.item():.4f}")

    R2_test = calculate_r2_in_batches(P_pred_test, targets_test)
    print(f"R-squared (test): {R2_test.item():.4f}")

    # Store metrics in dictionaries
    metric_rounds['train'][round] = RMSE_train.item()
    metric_rounds['val'][round] = RMSE_val.item()
    metric_rounds['test'][round] = RMSE_test.item()

    # 对于均值，我们在所有轮次结束后再计算
    if round == num_rounds - 1:
        metric_mean['train'] = np.mean(metric_rounds['train'])
        metric_mean['val'] = np.mean(metric_rounds['val'])
        metric_mean['test'] = np.mean(metric_rounds['test'])

        metric_std['train'] = np.std(metric_rounds['train'])
        metric_std['val'] = np.std(metric_rounds['val'])
        metric_std['test'] = np.std(metric_rounds['test'])

        print(f"\nAverage metrics over {num_rounds} rounds:")
        print(f"Train RMSE: {metric_mean['train']:.4f} ± {metric_std['train']:.4f}")
        print(f"Val RMSE: {metric_mean['val']:.4f} ± {metric_std['val']:.4f}")
        print(f"Test RMSE: {metric_mean['test']:.4f} ± {metric_std['test']:.4f}")
    
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
