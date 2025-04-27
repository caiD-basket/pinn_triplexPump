import numpy as np
import torch
from torch import nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import optim
import os
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置 PyTorch 环境
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_default_dtype(torch.float64)
torch.cuda.empty_cache()
torch.set_printoptions(precision=10)

# 使用 CUDA 如果可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自定义 Sin 激活函数
class Sin(nn.Module):
    def forward(self, input):
        return torch.sin(input)

# 模型超参数
BATCH_SIZE = 16
NUM_EPOCH = 20
NUM_LAYERS = 5
NUM_NEURONS = 64
NUM_ROUNDS = 2
SEQ_LEN = 1


def train(num_epoch, batch_size, train_loader, num_slices_train, inputs_val, targets_val,
          model, optimizer, scheduler, criterion):
    num_period = int(num_slices_train / batch_size)
    train_losses = []  # 记录训练损失
    val_losses = []    # 记录验证损失
    
    for epoch in range(num_epoch):
        model.train()
        epoch_train_loss = 0.0
        
        with torch.backends.cudnn.flags(enabled=False):
            for period, (inputs_train_batch, targets_train_batch) in enumerate(train_loader):
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
                epoch_train_loss += loss.item()
                
                if (epoch + 1) % 1 == 0 and (period + 1) % 1 == 0:
                    print(
                        'Epoch: {}, Period: {}, Loss: {:.5f}, Loss_M: {:.5f}, Loss_physics: {:.5f}'.format(
                            epoch + 1, period + 1, loss.item(), criterion.loss_M.item(), criterion.loss_physics.item()))
        
        # 计算平均训练损失
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 计算验证损失
        model.eval()
        # 确保输入张量需要梯度
        inputs_val.requires_grad_(True)
        P_pred_val, P_t_pred_val = model(inputs=inputs_val)
        val_loss = criterion(
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
        val_losses.append(val_loss.item())
        
        scheduler.step()
        
    return model, train_losses, val_losses


class TensorDataset(Dataset):
    """自定义数据集类，用于包装张量数据"""
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def standardize_tensor(data, mode, mean=0, std=1):
    """标准化张量数据
    
    Args:
        data: 要标准化的张量
        mode: 'fit'计算均值和标准差，'transform'使用提供的均值和标准差
        mean: 均值，mode='transform'时使用
        std: 标准差，mode='transform'时使用
    
    Returns:
        标准化后的数据，均值，标准差
    """
    data_2D = data.contiguous().view((-1, data.shape[-1]))  # 转为2D
    if mode == 'fit':
        mean = torch.mean(data_2D, dim=0)
        std = torch.std(data_2D, dim=0)
    data_norm_2D = (data_2D - mean) / (std + 1e-8)
    data_norm = data_norm_2D.contiguous().view((-1, data.shape[-2], data.shape[-1]))
    return data_norm, mean, std


def inverse_standardize_tensor(data_norm, mean, std):
    """反标准化张量数据
    
    Args:
        data_norm: 标准化后的数据
        mean: 均值
        std: 标准差
    
    Returns:
        反标准化后的数据
    """
    data_norm_2D = data_norm.contiguous().view((-1, data_norm.shape[-1]))  # 转为2D
    data_2D = data_norm_2D * std + mean
    data = data_2D.contiguous().view((-1, data_norm.shape[-2], data_norm.shape[-1]))
    return data


def calculate_metrics_in_batches(predictions, targets, batch_size=1024):
    """分批计算评估指标以节省内存
    
    Args:
        predictions: 预测值
        targets: 真实值
        batch_size: 批次大小
    
    Returns:
        RMSE, MAE, MAPE
    """
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


def calculate_r2_in_batches(predictions, targets, batch_size=1024):
    """分批计算R²以节省内存
    
    Args:
        predictions: 预测值
        targets: 真实值
        batch_size: 批次大小
    
    Returns:
        R² 值
    """
    # 确保输入是一维的
    if predictions.dim() > 1:
        predictions = predictions.squeeze()
    if targets.dim() > 1:
        targets = targets.squeeze()
        
    total_samples = predictions.size(0)
    
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


class Neural_Net(nn.Module):
    def __init__(self, seq_len, inputs_dim, outputs_dim, layers, activation='Tanh'):
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
    def __init__(self, physics_weight=0.1):  # 添加物理损失权重参数
        super().__init__()
        self.physics_weight = physics_weight
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
        batch_loss_M = torch.tensor(0.0, requires_grad=True)
        batch_loss_physics = torch.tensor(0.0, requires_grad=True)

        for i in range(num):
            if abs(dpdt[i][0]) < 1e-12:
                continue

            # 计算物理损失
            physics_loss = V * self.mixture_density_derivative(
                outputs_P[i] * 0.1, bulk_modulus_model,
                air_dissolution_model, rho_L_atm, beta_L_atm, beta_gain,
                air_fraction, rho_g_atm, polytropic_index, p_atm, p_crit,
                p_min) * dpdt[i][0] - mdot_A[i]
            
            # 计算测量值损失（使用MSE）
            mae_loss = torch.abs(outputs_P[i] - targets_P[i])

            # 累加批次损失
            batch_loss_physics =batch_loss_physics + torch.abs(physics_loss)  # 平方以确保正值
            batch_loss_M =batch_loss_M + mae_loss

        # 计算平均损失
        self.loss_physics = batch_loss_physics / num
        self.loss_M = batch_loss_M / num

        # 总损失 = 测量损失 + 权重 * 物理损失
        total_loss = self.loss_M + self.physics_weight * self.loss_physics

        return total_loss


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


# ==============================================================================
# 主程序
# ==============================================================================

def main():
    """主程序函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. 数据加载和预处理
    print("1. 加载和预处理数据...")
    data = pd.read_csv('combined_all_425_1.csv')
    
    X = data.drop(['pOut'], axis=1)
    Y = data['pOut']
    
    # 按时间顺序划分训练集、验证集和测试集 (60%, 20%, 20%)
    train_size = int(len(X) * 0.6)
    val_size = int(len(X) * 0.2)
    
    # 划分数据集
    X_train = X.iloc[:train_size]
    y_train = Y.iloc[:train_size]
    X_val = X.iloc[train_size:train_size + val_size]
    y_val = Y.iloc[train_size:train_size + val_size]
    X_test = X.iloc[train_size + val_size:]
    y_test = Y.iloc[train_size + val_size:]
    
    # 转换为torch.Tensor
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float64)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float64).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float64)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float64).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float64)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float64).unsqueeze(1)
    
    # 创建字典存储数据
    inputs = {
        'train': X_train_tensor,
        'val': X_val_tensor,
        'test': X_test_tensor
    }
    
    targets = {
        'train': y_train_tensor,
        'val': y_val_tensor,
        'test': y_test_tensor
    }
    
    # 移动数据到设备
    inputs_train = inputs['train'].to(device)
    inputs_val = inputs['val'].to(device)
    inputs_test = inputs['test'].to(device)
    targets_train = targets['train'].to(device)
    targets_val = targets['val'].to(device)
    targets_test = targets['test'].to(device)
    
    # 定义模型参数
    inputs_dim = 4
    outputs_dim = 1
    layers = NUM_LAYERS * [NUM_NEURONS]
    
    # 数据标准化
    num = inputs_train.shape[0]
    _, mean_inputs_train, std_inputs_train = standardize_tensor(torch.reshape(inputs_train, (num, 1, 4)), mode='fit')
    _, mean_targets_train, std_targets_train = standardize_tensor(targets_train, mode='fit')
    
    # 创建训练集DataLoader
    train_set = TensorDataset(inputs_train, targets_train)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True)
    
    # 初始化指标记录
    metric_rounds = {
        'train': np.zeros(NUM_ROUNDS),
        'val': np.zeros(NUM_ROUNDS),
        'test': np.zeros(NUM_ROUNDS)
    }
    
    # 2. 多轮训练和评估
    print(f"2. 开始{NUM_ROUNDS}轮训练和评估...")
    
    all_train_losses = []
    all_val_losses = []
    
    for round in range(NUM_ROUNDS):
        print(f"\n=== 第 {round+1}/{NUM_ROUNDS} 轮 ===")
        
        # 初始化模型
        model = TriplexPINN(
            seq_len=SEQ_LEN,
            inputs_dim=inputs_dim,
            outputs_dim=outputs_dim,
            layers=layers,
            scaler_inputs=(mean_inputs_train, std_inputs_train),
            scaler_targets=(mean_targets_train, std_targets_train)
        ).to(device)
        
        # 初始化损失函数和优化器
        criterion = My_loss()
        params = [p for p in model.parameters()]
        optimizer = optim.Adam(params, lr=0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.1)
        
        # 训练模型
        print(f"训练模型中...")
        model, train_losses, val_losses = train(
            num_epoch=NUM_EPOCH,
            batch_size=BATCH_SIZE,
            train_loader=train_loader,
            num_slices_train=inputs_train.shape[0],
            inputs_val=inputs_val,
            targets_val=targets_val,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion
        )
        
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        
        # 评估模型
        print(f"评估模型中...")
        model.eval()
        
        # 计算训练集指标
        P_pred_train, P_t_pred_train = model(inputs=inputs_train)
        RMSE_train, MAE_train, MAPE_train = calculate_metrics_in_batches(P_pred_train, targets_train)
        print(f"RMSE_train:{RMSE_train}")
        print(f"MAE_train:{MAE_train}")
        print(f"MAPE_train:{MAPE_train}")
        
        # 计算验证集指标
        P_pred_val, P_t_pred_val = model(inputs=inputs_val)
        RMSE_val, MAE_val, MAPE_val = calculate_metrics_in_batches(P_pred_val, targets_val)
        print(f"RMSE_val:{RMSE_val}")
        print(f"MAE_val:{MAE_val}")
        print(f"MAPE_val:{MAPE_val}")
        
        # 计算测试集指标
        P_pred_test, P_t_pred_test = model(inputs=inputs_test)
        RMSE_test, MAE_test, MAPE_test = calculate_metrics_in_batches(P_pred_test, targets_test)
        print(f"RMSE_test:{RMSE_test}")
        print(f"MAE_test:{MAE_test}")
        print(f"MAPE_test:{MAPE_test}")
        
        # 计算R²指标
        R2_train = calculate_r2_in_batches(P_pred_train, targets_train)
        print(f"R-squared (train): {R2_train.item():.4f}")
        
        R2_val = calculate_r2_in_batches(P_pred_val, targets_val)
        print(f"R-squared (val): {R2_val.item():.4f}")
        
        R2_test = calculate_r2_in_batches(P_pred_test, targets_test)
        print(f"R-squared (test): {R2_test.item():.4f}")
        
        # 记录每轮结果
        metric_rounds['train'][round] = RMSE_train.item()
        metric_rounds['val'][round] = RMSE_val.item()
        metric_rounds['test'][round] = RMSE_test.item()
    
    # 3. 计算多轮平均结果
    print("\n3. 计算多轮平均结果...")
    
    metric_mean = {
        'train': np.mean(metric_rounds['train']),
        'val': np.mean(metric_rounds['val']),
        'test': np.mean(metric_rounds['test'])
    }
    
    metric_std = {
        'train': np.std(metric_rounds['train']),
        'val': np.std(metric_rounds['val']),
        'test': np.std(metric_rounds['test'])
    }
    
    print(f"\n平均指标 ({NUM_ROUNDS}轮):")
    print(f"训练集 RMSE: {metric_mean['train']:.4f} ± {metric_std['train']:.4f}")
    print(f"验证集 RMSE: {metric_mean['val']:.4f} ± {metric_std['val']:.4f}")
    print(f"测试集 RMSE: {metric_mean['test']:.4f} ± {metric_std['test']:.4f}")
    
    # 绘制损失函数走势图
    plt.figure(figsize=(12, 6))
    
    # 计算所有轮次的平均损失
    avg_train_losses = np.mean(all_train_losses, axis=0)
    avg_val_losses = np.mean(all_val_losses, axis=0)
    
    # 绘制训练损失和验证损失
    plt.plot(range(1, NUM_EPOCH + 1), avg_train_losses, label='训练损失')
    plt.plot(range(1, NUM_EPOCH + 1), avg_val_losses, label='验证损失')
    
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.title(f'损失函数走势图 (平均值，{NUM_ROUNDS}轮)')
    plt.legend()
    plt.grid(True)
    
    # 保存图片
    plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
    print("\n损失函数走势图已保存到 'loss_curve.png'")
    
    # 4. 保存最终模型和结果
    print("\n4. 保存结果...")
    
    # 使用最后一轮的模型进行最终预测
    model.eval()
    P_pred_test, P_t_pred_test = model(inputs=inputs_test)
    
    # 保存结果
    results = {
        'P_true': targets_test.detach().cpu().numpy().squeeze(),
        'P_pred': P_pred_test.detach().cpu().numpy().squeeze(),
        'Cycles': inputs_test[:, 0].detach().cpu().numpy().squeeze(),
        'Epochs': np.arange(0, NUM_EPOCH)
    }
    
    torch.save(results, 'testTriplex.pth')
    print("结果已保存到 'testTriplex.pth'")
    
    return model, results


if __name__ == "__main__":
    main()
