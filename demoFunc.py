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
NUM_ROUNDS = 20
SEQ_LEN = 10


def train(num_epoch, batch_size, train_loader, num_slices_train, inputs_val, targets_val,
          model, optimizer, scheduler, criterion):
    num_period = int(num_slices_train / batch_size)
    train_losses = []  # 记录训练损失
    val_losses = []  # 记录验证损失

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
    # 判断输入类型，并正确处理
    if isinstance(data, np.ndarray):
        # 如果是numpy数组，转为torch张量
        data = torch.tensor(data, dtype=torch.float64)

    # 获取设备信息
    device = data.device

    # 确保mean和std在相同设备上
    if mode == 'transform':
        if isinstance(mean, torch.Tensor):
            mean = mean.to(device)
        else:
            mean = torch.tensor(mean, dtype=torch.float64, device=device)

        if isinstance(std, torch.Tensor):
            std = std.to(device)
        else:
            std = torch.tensor(std, dtype=torch.float64, device=device)

    # 确保是2维张量 [samples, features]
    if data.dim() == 1:
        data = data.unsqueeze(-1)  # 添加特征维度

    if data.dim() > 2:
        # 如果是高维张量，展平为2D
        original_shape = data.shape
        data_2D = data.reshape(-1, data.shape[-1])
    else:
        # 已经是2D
        data_2D = data

    # 计算或使用均值和标准差
    if mode == 'fit':
        mean = torch.mean(data_2D, dim=0)
        std = torch.std(data_2D, dim=0)

    # 标准化
    data_norm_2D = (data_2D - mean) / (std + 1e-8)

    # 恢复原始形状
    if data.dim() > 2:
        data_norm = data_norm_2D.reshape(original_shape)
    else:
        data_norm = data_norm_2D

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
    # 判断输入类型，并正确处理
    if isinstance(data_norm, np.ndarray):
        # 如果是numpy数组，转为torch张量
        data_norm = torch.tensor(data_norm, dtype=torch.float64)

    # 保存原始形状
    original_shape = data_norm.shape

    # 转为2D张量以便计算
    if data_norm.dim() > 2:
        # 高维张量
        data_norm_2D = data_norm.reshape(-1, data_norm.shape[-1])
    else:
        # 已经是2D或1D
        data_norm_2D = data_norm if data_norm.dim() == 2 else data_norm.unsqueeze(-1)

    # 反标准化
    data_2D = data_norm_2D * std + mean

    # 恢复原始形状
    if data_norm.dim() > 2:
        data = data_2D.reshape(original_shape)
    else:
        data = data_2D

    return data


def calculate_metrics_in_batches(predictions, targets):
    """计算RMSE, MAE和MAPE指标，同时安全处理零值情况"""
    # 确保输入是numpy数组
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # 计算RMSE和MAE
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mae = np.mean(np.abs(predictions - targets))

    # 防止除以零，对零或接近零的值进行处理
    nonzero_mask = np.abs(targets) > 1e-10
    if np.any(nonzero_mask):
        mape = np.mean(np.abs((predictions[nonzero_mask] - targets[nonzero_mask]) / targets[nonzero_mask])) * 100
    else:
        mape = float('inf')
        print("警告: 计算MAPE时，所有目标值都接近零")

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
        self.layers.append(nn.Dropout(p=0.1))

        # 中间层
        for l in range(len(layers) - 1):
            layer = nn.Linear(layers[l], layers[l + 1]).double()  # 转为 double
            nn.init.xavier_normal_(layer.weight)
            self.layers.append(layer)

            if activation == 'Tanh':
                self.layers.append(nn.Tanh())
            elif activation == 'Sin':
                self.layers.append(Sin())
            self.layers.append(nn.Dropout(p=0.1))

        # 输出层
        layer = nn.Linear(layers[-1], outputs_dim).double()  # 转为 double
        nn.init.xavier_normal_(layer.weight)
        self.layers.append(layer)

        # 构建 Sequential
        self.NN = nn.Sequential(*self.layers)

    def forward(self, x):
        # 确保输入转为 double
        x = x.double() if x.dtype != torch.float64 else x

        # 保存原始输入形状
        original_shape = x.shape
        batch_size = original_shape[0]

        # 检查输入是否为3D张量 [batch, seq, features]
        is_3d = len(original_shape) == 3

        # 将输入调整为2D以便通过网络处理
        if is_3d:
            # 如果是3D输入 [batch, seq, features]，展平为[batch*seq, features]
            self.x = x.contiguous().view(-1, self.inputs_dim)
            NN_out_2D = self.NN(self.x)  # 输入已确保为 double
            # 恢复3D形状
            self.p_pred = NN_out_2D.view(batch_size, -1, self.outputs_dim)
        else:
            # 如果是2D输入 [batch, features]，直接使用
            self.x = x
            NN_out_2D = self.NN(self.x)  # 输入已确保为 double
            # 将输出调整为3D形状以保持一致的输出接口 [batch, 1, outputs_dim]
            self.p_pred = NN_out_2D.view(batch_size, 1, self.outputs_dim)

        return self.p_pred


class My_loss(nn.Module):
    def __init__(self, physics_weight=0.5, cls_weight=0.5):  # 添加物理损失权重和分类损失权重参数
        super().__init__()
        self.physics_weight = physics_weight
        self.cls_weight = cls_weight
        self.loss_M = torch.tensor(0.0, requires_grad=True)
        self.loss_physics = torch.tensor(0.0, requires_grad=True)
        self.loss_cls = torch.tensor(0.0, requires_grad=True)
        self.cls_criterion = nn.BCEWithLogitsLoss()

    def mixture_density_derivative(self, p, bulk_modulus_model, air_dissolution_model,
                                   rho_L_atm, beta_L_atm, beta_gain, air_fraction,
                                   rho_g_atm, polytropic_index, p_atm, p_crit, p_min):
        # 转换为PyTorch张量
        device = p.device if isinstance(p, torch.Tensor) else torch.device('cpu')

        if not isinstance(p, torch.Tensor):
            p = torch.tensor(p, dtype=torch.float64, device=device)

        # 转换常量为张量
        p_atm = torch.tensor(p_atm, dtype=torch.float64, device=device)
        p_crit = torch.tensor(p_crit, dtype=torch.float64, device=device)
        p_min = torch.tensor(p_min, dtype=torch.float64, device=device)
        rho_L_atm = torch.tensor(rho_L_atm, dtype=torch.float64, device=device)
        beta_L_atm = torch.tensor(beta_L_atm, dtype=torch.float64, device=device)
        beta_gain = torch.tensor(beta_gain, dtype=torch.float64, device=device)
        air_fraction = torch.tensor(air_fraction, dtype=torch.float64, device=device)
        rho_g_atm = torch.tensor(rho_g_atm, dtype=torch.float64, device=device)
        polytropic_index = torch.tensor(polytropic_index, dtype=torch.float64, device=device)

        # Determine p_used
        p_used = torch.max(p, p_min)

        # Calculate theta (fraction of air entrained)
        if air_dissolution_model == 'off':
            theta = torch.tensor(1.0, dtype=torch.float64, device=device)
        else:
            if p_used <= p_atm:
                theta = torch.tensor(1.0, dtype=torch.float64, device=device)
            elif p_used >= p_crit:
                theta = torch.tensor(0.0, dtype=torch.float64, device=device)
            else:
                L = p_crit - p_atm
                x = (p_used - p_atm) / L
                theta = 1 - 3 * x ** 2 + 2 * x ** 3

        # Calculate dtheta_dp (derivative of theta)
        if air_dissolution_model == 'off':
            dtheta_dp = torch.tensor(0.0, dtype=torch.float64, device=device)
        else:
            if p_used <= p_atm or p_used >= p_crit:
                dtheta_dp = torch.tensor(0.0, dtype=torch.float64, device=device)
            else:
                L = p_crit - p_atm
                dtheta_dp = 6 * (p_used - p_atm) * (p_used - p_crit) / (L ** 3)

        # Calculate p_denom
        if air_fraction == 0:
            p_denom = torch.tensor(0.0, dtype=torch.float64, device=device)
        else:
            p_denom = (air_fraction / (1 - air_fraction)) * (p_atm / p_used) ** (1 / polytropic_index) * theta

        # Calculate p_ratio
        if air_fraction == 0:
            p_ratio = torch.tensor(0.0, dtype=torch.float64, device=device)
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
                exp_term = torch.exp((p_used - p_atm) / beta_L_atm) / beta_L_atm
            else:
                base = 1 + beta_gain * (p_used - p_atm) / beta_L_atm
                exponent = (-1 + 1 / beta_gain)
                exp_term = (base ** exponent) / beta_L_atm
        else:
            if bulk_modulus_model == 'const':
                exp_term = torch.exp(-(p_used - p_atm) / beta_L_atm) / beta_L_atm
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

    def forward(self, p_pred, p_t_pred, fault_logit, targets_p, targets_fault, mdot_A, V,
                targets_p_orig=None,  # 添加原始pOut参数
                bulk_modulus_model='const', air_dissolution_model='off',
                rho_L_atm=851.6, beta_L_atm=1.46696e+03, beta_gain=0.2, air_fraction=0.005,
                rho_g_atm=1.225, polytropic_index=1.0, p_atm=0.101325, p_crit=3, p_min=1,
                mean_p=None, std_p=None):
        """
        多任务损失函数：结合物理约束损失和故障分类损失

        参数:
            p_pred: 预测压力 [batch, seq_len, 1] - 已反标准化
            p_t_pred: 压力导数 [batch, seq_len, 1]
            fault_logit: 故障分类预测 [batch, 1]
            targets_p: 真实压力 [batch, seq_len, 1] - 标准化状态
            targets_fault: 真实故障标签 [batch]
            mdot_A: 质量流量 [batch, seq_len]
            V: 体积常数
            targets_p_orig: 原始未标准化的压力值 [batch, seq_len, 1]
            mean_p: 压力标准化均值 (可选)
            std_p: 压力标准化标准差 (可选)
            其他参数: 物理模型参数

        返回:
            总损失，测量损失，物理损失，分类损失的列表
        """
        batch_size = p_pred.shape[0]
        seq_len = p_pred.shape[1]
        device = p_pred.device

        # 打印调试信息 - 检查值范围
        if torch.rand(1).item() < 0.01:  # 随机打印1%的批次避免日志过多
            p_min_val = torch.min(p_pred).item()
            p_max_val = torch.max(p_pred).item()
            t_min_val = torch.min(targets_p).item() if targets_p is not None else 0
            t_max_val = torch.max(targets_p).item() if targets_p is not None else 0
            print(
                f"损失函数 - p_pred范围(反标准化): [{p_min_val:.4f}, {p_max_val:.4f}], targets_p范围(标准化): [{t_min_val:.4f}, {t_max_val:.4f}]")

            if targets_p_orig is not None:
                orig_min_val = torch.min(targets_p_orig).item()
                orig_max_val = torch.max(targets_p_orig).item()
                print(f"targets_p_orig范围(原始): [{orig_min_val:.4f}, {orig_max_val:.4f}]")

        # 初始化损失
        batch_loss_M = torch.tensor(0.0, device=device, requires_grad=True)
        batch_loss_physics = torch.tensor(0.0, device=device, requires_grad=True)

        # 确保targets_p_orig在正确的设备上并且是pytorch张量
        if targets_p_orig is not None:
            if not isinstance(targets_p_orig, torch.Tensor):
                targets_p_orig = torch.tensor(targets_p_orig, dtype=torch.float64, device=device)
            elif targets_p_orig.device != device:
                targets_p_orig = targets_p_orig.to(device)

            # 确保维度正确
            if targets_p_orig.dim() == 2:
                targets_p_orig = targets_p_orig.unsqueeze(-1)
        else:
            # 如果未提供原始数据，则将标准化的targets_p反标准化
            if mean_p is not None and std_p is not None:
                # 确保标准化参数在正确的设备上
                if isinstance(mean_p, np.ndarray):
                    mean_p = torch.tensor(mean_p, dtype=torch.float64, device=device)
                elif mean_p.device != device:
                    mean_p = mean_p.to(device)

                if isinstance(std_p, np.ndarray):
                    std_p = torch.tensor(std_p, dtype=torch.float64, device=device)
                elif std_p.device != device:
                    std_p = std_p.to(device)

                # 反标准化targets_p
                targets_p_orig = targets_p * std_p + mean_p

                if torch.rand(1).item() < 0.01:  # 随机打印
                    targets_denorm_min = torch.min(targets_p_orig).item()
                    targets_denorm_max = torch.max(targets_p_orig).item()
                    print(
                        f"损失函数 - targets_p_orig范围(反标准化计算): [{targets_denorm_min:.4f}, {targets_denorm_max:.4f}], mean_p={mean_p.item():.4f}, std_p={std_p.item():.4f}")
            else:
                # 如果未提供标准化参数，则只能使用标准化状态的targets_p
                print(f"警告: 未提供原始数据和标准化参数，直接使用标准化状态的targets_p")
                targets_p_orig = targets_p

        # 计算每个批次和时间步的物理损失和MAE损失
        for b in range(batch_size):
            for s in range(seq_len):
                # 跳过导数接近0的情况
                if abs(p_t_pred[b, s, 0]) < 1e-12:
                    continue

                # 计算物理损失 - 密度导数和质量流量关系
                physics_loss = V * self.mixture_density_derivative(
                    p_pred[b, s, 0] * 0.1,  # 转换单位
                    bulk_modulus_model,
                    air_dissolution_model,
                    rho_L_atm, beta_L_atm, beta_gain,
                    air_fraction, rho_g_atm, polytropic_index,
                    p_atm, p_crit, p_min
                ) * p_t_pred[b, s, 0] - mdot_A[b, s]

                # 计算测量损失 - 直接使用原始pOut数据
                mae_loss = torch.abs(targets_p_orig[b, s, 0] - p_pred[b, s, 0])

                # 累加批次损失
                batch_loss_physics = batch_loss_physics + torch.abs(physics_loss)
                batch_loss_M = batch_loss_M + mae_loss

        # 计算分类损失
        cls_loss = self.cls_criterion(fault_logit.squeeze(-1), targets_fault.float())

        # 计算平均损失
        total_elements = batch_size * seq_len
        self.loss_physics = batch_loss_physics / total_elements
        self.loss_M = batch_loss_M / total_elements
        self.loss_cls = cls_loss

        # 总损失 = 测量损失 + 物理权重 * 物理损失 + 分类权重 * 分类损失
        total_loss = self.loss_M + self.physics_weight * self.loss_physics + self.cls_weight * self.loss_cls

        return [total_loss, self.loss_M, self.loss_physics, self.loss_cls]


class MultiTaskPINN(nn.Module):
    def __init__(self, seq_len, inputs_dim, outputs_dim, layers, scaler_inputs, scaler_targets, activation='Tanh'):
        super().__init__()
        self.seq_len = seq_len
        self.inputs_dim = inputs_dim
        self.outputs_dim = outputs_dim
        self.scaler_inputs = scaler_inputs
        self.scaler_targets = scaler_targets

        # 预测压力的PINN网络
        self.pinn = Neural_Net(seq_len, inputs_dim, outputs_dim, layers, activation)

        # 故障分类分支 - 将所有压力预测合并处理后输出单个故障概率
        self.fc_fault = nn.Sequential(
            nn.Linear(seq_len * outputs_dim, 64).double(),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32).double(),
            nn.ReLU(),
            nn.Linear(32, 1).double()
        )

    def forward(self, x):
        """
        处理序列输入并输出每帧的压力预测和整体故障预测

        参数:
            x: 输入序列 [batch, seq_len, features] 或 [batch, features]
        返回:
            P: 每帧压力预测 [batch, seq_len, 1]
            P_t: 压力对时间的导数 [batch, seq_len, 1]
            fault_logit: 故障类型预测 [batch, 1]
        """
        # 保存设备信息
        device = x.device
        batch_size = x.shape[0]

        # 检查输入维度，如果是2D，调整为3D
        if x.dim() == 2:
            # 如果输入是 [batch, features]，扩展为 [batch, 1, features]
            x = x.unsqueeze(1)
            print(f"警告: 输入张量维度为2，已自动调整为3D: {x.shape}")

        # 准备标准化参数
        mean_inputs = self.scaler_inputs[0]
        std_inputs = self.scaler_inputs[1]

        # 确保mean和std是在正确设备上的torch张量
        if isinstance(mean_inputs, np.ndarray):
            mean_inputs = torch.tensor(mean_inputs, dtype=torch.float64, device=device)
        else:
            mean_inputs = mean_inputs.to(device)

        if isinstance(std_inputs, np.ndarray):
            std_inputs = torch.tensor(std_inputs, dtype=torch.float64, device=device)
        else:
            std_inputs = std_inputs.to(device)

        # 分离时间特征和其他特征
        t = x[:, :, 0:1]  # 时间特征
        t.requires_grad_(True)  # 确保时间特征需要梯度计算

        # 其他特征
        s = x[:, :, 1:].clone()

        # 分别标准化特征 - 确保时间特征放在正确的位置
        s_norm, _, _ = standardize_tensor(s, mode='transform',
                                          mean=mean_inputs[1:],
                                          std=std_inputs[1:])

        t_norm, _, _ = standardize_tensor(t, mode='transform',
                                          mean=mean_inputs[0:1],
                                          std=std_inputs[0:1])
        t_norm.requires_grad_(True)
        # 构建标准化后的输入张量 - 确保特征顺序正确
        # x_norm = torch.cat([t_norm, s_norm], dim=2)

        # 通过PINN网络获取压力预测
        P_norm = self.pinn(torch.cat([t_norm, s_norm], dim=2))

        # 准备反标准化参数
        mean_targets = self.scaler_targets[0]
        std_targets = self.scaler_targets[1]

        # 确保mean和std是在正确设备上的torch张量
        if isinstance(mean_targets, np.ndarray):
            mean_targets = torch.tensor(mean_targets, dtype=torch.float64, device=device)
        else:
            mean_targets = mean_targets.to(device)

        if isinstance(std_targets, np.ndarray):
            std_targets = torch.tensor(std_targets, dtype=torch.float64, device=device)
        else:
            std_targets = std_targets.to(device)

        # 反标准化压力预测
        P = P_norm * std_targets + mean_targets
        grad_outputs = torch.ones_like(P)
        # 计算压力导数
        try:
            P_t_norm = torch.autograd.grad(
                outputs=P,
                inputs=t_norm,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
        except Exception as e:
            # 如果在评估模式下出现梯度计算错误，提供零梯度
            print(f"梯度计算错误: {e}")
            P_t_norm = torch.zeros_like(t_norm, device=device)

        # 还原为 dP/dt_raw
        P_t = P_t_norm / std_inputs[0]  # std_inputs[0] 是 time 的标准差

        # 故障分类 - 使用整个序列的压力预测
        feat_cls = P.reshape(batch_size, -1)  # [batch, seq_len*1]
        fault_logit = self.fc_fault(feat_cls)

        return P, P_t, fault_logit


class SequenceDataset(Dataset):
    def __init__(self, X, y_p, y_fault, seq_len, y_p_orig, X_orig=None):
        """
        序列数据集，用于创建滑动窗口数据

        参数:
            X: 输入特征，shape [num_samples, features]
            y_p: 压力标签，shape [num_samples, 1]
            y_fault: 故障标签，shape [num_samples]
            seq_len: 序列长度
            y_p_orig: 原始压力标签
            X_orig: 原始输入特征，shape [num_samples, features]
        """
        self.X = X
        self.y_p = y_p
        self.y_fault = y_fault
        self.seq_len = seq_len
        self.length = max(0, len(X) - seq_len + 1)  # 确保长度非负
        self.y_p_orig = y_p_orig
        self.X_orig = X_orig  # 存储原始输入特征

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        获取滑动窗口数据

        返回:
            x_seq: 输入序列, shape [seq_len, features]
            y_p_seq: 目标压力序列, shape [seq_len, 1]
            y_fault_label: 最后一帧的故障标签
            y_p_orig_seq: 原始压力序列, shape [seq_len, 1]
            x_orig_seq: 原始输入序列, shape [seq_len, features] (如果有提供)
        """
        # 获取从idx开始的seq_len长度的序列
        x_seq = self.X[idx:idx + self.seq_len]
        y_p_seq = self.y_p[idx:idx + self.seq_len]
        y_p_orig_seq = self.y_p_orig[idx:idx + self.seq_len]

        # 故障标签使用序列的最后一帧
        y_fault_label = self.y_fault[idx + self.seq_len - 1]
        
        # 如果提供了原始输入特征，也返回对应的序列
        if self.X_orig is not None:
            x_orig_seq = self.X_orig[idx:idx + self.seq_len]
            return x_seq, y_p_seq, y_fault_label, y_p_orig_seq, x_orig_seq
        else:
            return x_seq, y_p_seq, y_fault_label, y_p_orig_seq


def visualize_sliding_window(data, seq_len, num_examples=3):
    """可视化滑动窗口的工作原理"""
    total_samples = len(data)
    total_windows = total_samples - seq_len + 1

    print(f"数据总长度: {total_samples}")
    print(f"窗口长度: {seq_len}")
    print(f"窗口总数: {total_windows}")

    for i in range(min(num_examples, total_windows)):
        window_indices = list(range(i, i + seq_len))
        print(f"窗口 #{i + 1}: 包含索引 {window_indices} (长度: {len(window_indices)})")


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
    data = pd.read_csv('combined_all_f1.csv')

    # 提取特征和标签
    X = data[['time', 'pIn', 'mdot_A', 'iMotor']].values  # 输入特征
    y_p = data['pOut'].values.reshape(-1, 1)  # 压力输出 - 原始pOut
    y_fault = data['label'].values  # 故障标签

    # 保存原始特征数据用于物理模型
    X_orig = X.copy()

    # 标准化数据
    X_norm, X_mean, X_std = standardize_tensor(torch.tensor(X, dtype=torch.float64), 'fit')
    y_p_norm, y_p_mean, y_p_std = standardize_tensor(torch.tensor(y_p, dtype=torch.float64), 'fit')

    # 将PyTorch张量转回NumPy数组
    X_norm = X_norm.cpu().numpy()
    y_p_norm = y_p_norm.cpu().numpy()

    # 输出原始形状，帮助调试
    print(f"标准化后的数据形状:")
    print(f"X_norm 形状: {X_norm.shape}")
    print(f"y_p_norm 形状: {y_p_norm.shape}")

    # 按时间顺序划分训练集、验证集和测试集 (60%, 20%, 20%)
    train_size = int(len(X_norm) * 0.6)
    val_size = int(len(X_norm) * 0.2)

    # 划分数据集 - 标准化后的特征和标签
    X_train = X_norm[:train_size]
    X_val = X_norm[train_size:train_size + val_size]
    X_test = X_norm[train_size + val_size:]

    y_p_train = y_p_norm[:train_size]
    y_p_val = y_p_norm[train_size:train_size + val_size]
    y_p_test = y_p_norm[train_size + val_size:]

    # 划分原始数据 - 用于MAE损失计算
    y_p_orig = y_p  # 保存原始pOut数据
    y_p_train_orig = y_p_orig[:train_size]
    y_p_val_orig = y_p_orig[train_size:train_size + val_size]
    y_p_test_orig = y_p_orig[train_size + val_size:]

    # 划分原始特征数据 - 用于物理损失计算
    X_train_orig = X_orig[:train_size]
    X_val_orig = X_orig[train_size:train_size + val_size]
    X_test_orig = X_orig[train_size + val_size:]

    y_fault_train = y_fault[:train_size]
    y_fault_val = y_fault[train_size:train_size + val_size]
    y_fault_test = y_fault[train_size + val_size:]

    # 输出数据集大小，帮助调试
    print(f"数据集大小信息：")
    print(f"原始数据: {len(data)} 条记录")
    print(f"X_train 形状: {X_train.shape}")
    print(f"y_p_train 形状: {y_p_train.shape}")
    print(f"y_p_train_orig 形状: {y_p_train_orig.shape}")
    print(f"y_fault_train 形状: {y_fault_train.shape}")
    print(f"X_val 形状: {X_val.shape}")
    print(f"X_test 形状: {X_test.shape}")
    print(f"序列长度 (SEQ_LEN): {SEQ_LEN}")
    print(f"训练集窗口数量计算: {len(X_train)} - {SEQ_LEN} + 1 = {len(X_train) - SEQ_LEN + 1}")

    # 创建序列数据集
    print(f"2. 创建滑动窗口数据集（窗口大小={SEQ_LEN}）...")

    # 训练集
    train_set = SequenceDataset(X_train, y_p_train, y_fault_train, SEQ_LEN, y_p_train_orig, X_train_orig)
    print(f"训练集实际窗口数: {len(train_set)}")

    # 验证集
    val_set = SequenceDataset(X_val, y_p_val, y_fault_val, SEQ_LEN, y_p_val_orig, X_val_orig)

    # 测试集
    test_set = SequenceDataset(X_test, y_p_test, y_fault_test, SEQ_LEN, y_p_test_orig, X_test_orig)

    # 计算有效窗口数量
    print(f"训练集窗口数: {len(train_set)}")
    print(f"验证集窗口数: {len(val_set)}")
    print(f"测试集窗口数: {len(test_set)}")

    # 确保数据集不为空再创建DataLoader
    print(f"创建DataLoader，batch_size={BATCH_SIZE}...")
    if len(train_set) > 0 and len(val_set) > 0 and len(test_set) > 0:
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
        print(f"DataLoader创建成功！")
    else:
        raise ValueError("数据集长度为0，无法创建DataLoader。请检查数据集长度和序列长度参数。")

    # 模型参数定义
    inputs_dim = X.shape[1]  # 特征维度
    outputs_dim = 1  # 输出维度 (压力)
    layers = NUM_LAYERS * [NUM_NEURONS]  # 隐藏层配置

    # 初始化指标记录
    metric_rounds = {
        'train': np.zeros(NUM_ROUNDS),
        'val': np.zeros(NUM_ROUNDS),
        'test': np.zeros(NUM_ROUNDS)
    }

    # 3. 多轮训练和评估
    print(f"3. 开始{NUM_ROUNDS}轮训练和评估...")

    all_losses = {'train': [], 'val': [], 'test': []}

    for round_idx in range(NUM_ROUNDS):
        print(f"\n=== 第 {round_idx + 1}/{NUM_ROUNDS} 轮 ===")

        # 初始化模型
        model = MultiTaskPINN(
            seq_len=SEQ_LEN,
            inputs_dim=inputs_dim,
            outputs_dim=outputs_dim,
            layers=layers,
            scaler_inputs=(X_mean, X_std),
            scaler_targets=(y_p_mean, y_p_std)
        ).to(device)

        # 初始化损失函数和优化器
        criterion = My_loss(physics_weight=0.5, cls_weight=0.5)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.1)

        # 训练模型
        train_losses = []
        val_losses = []
        test_losses = []

        # 初始化测试数据收集列表（整个训练过程的）
        all_epoch_p_pred_test = []
        all_epoch_fault_logits_test = []
        all_epoch_targets_p_test = []
        all_epoch_targets_fault_test = []
        all_epoch_targets_p_orig_test = []

        # 初始化早停变量
        best_val_loss = float('inf')
        no_improvement_count = 0
        patience = 10  # 早停耐心值，连续10轮无改善就停止

        for epoch in range(NUM_EPOCH):
            # 训练阶段
            model.train()
            epoch_train_loss = 0.0
            epoch_train_pinn_loss = 0.0
            epoch_train_cls_loss = 0.0
            num_batches = 0

            try:
                for batch_idx, (inputs_batch, targets_p_batch, targets_fault_batch, targets_p_orig_batch, inputs_orig_batch) in enumerate(
                        train_loader):
                    # 打印调试信息
                    if epoch == 0 and batch_idx == 0:
                        print(f"\n调试信息 - 第一个批次:")
                        print(f"inputs_batch 形状: {inputs_batch.shape}")
                        print(f"targets_p_batch 形状: {targets_p_batch.shape}")
                        print(f"targets_fault_batch 形状: {targets_fault_batch.shape}")
                        print(f"targets_p_orig_batch 形状: {targets_p_orig_batch.shape}")
                        print(f"inputs_orig_batch 形状: {inputs_orig_batch.shape}")

                    # 移动数据到设备
                    inputs_batch = inputs_batch.to(device)
                    targets_p_batch = targets_p_batch.to(device).unsqueeze(-1)  # 确保形状为 [batch, seq_len, 1]
                    targets_fault_batch = targets_fault_batch.to(device)
                    targets_p_orig_batch = targets_p_orig_batch.to(device).unsqueeze(-1)  # 确保形状为 [batch, seq_len, 1]
                    inputs_orig_batch = inputs_orig_batch.to(device)

                    # 获取原始 mdot_A 值
                    mdot_A_orig_batch = inputs_orig_batch[:, :, 2]  # [batch, seq_len]

                    # 前向传播
                    p_pred, p_t_pred, fault_logit = model(inputs_batch)

                    # 打印预测结果形状
                    if epoch == 0 and batch_idx == 0:
                        print(f"p_pred 形状: {p_pred.shape}")
                        print(f"p_t_pred 形状: {p_t_pred.shape}")
                        print(f"fault_logit 形状: {fault_logit.shape}")
                        print(f"mdot_A_orig 形状: {mdot_A_orig_batch.shape}")
                        print(f"mdot_A_orig 值范围: [{mdot_A_orig_batch.min().item():.6f}, {mdot_A_orig_batch.max().item():.6f}]")
                        print(f"标准化后 mdot_A 值范围: [{inputs_batch[:, :, 2].min().item():.6f}, {inputs_batch[:, :, 2].max().item():.6f}]")

                    # 计算损失
                    loss = criterion(
                        p_pred, p_t_pred, fault_logit,
                        targets_p_batch, targets_fault_batch,
                        mdot_A_orig_batch,  # 使用原始 mdot_A 数据
                        V=2 * 1e-4,  # 体积参数
                        targets_p_orig=targets_p_orig_batch,  # 传入原始pOut数据
                        mean_p=y_p_mean,  # 压力均值
                        std_p=y_p_std  # 压力标准差
                    )

                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss[0].backward()  # 总损失
                    optimizer.step()

                    # 累加损失
                    epoch_train_loss += loss[0].item()
                    epoch_train_pinn_loss += loss[2].item()  # 物理损失
                    epoch_train_cls_loss += loss[3].item()  # 分类损失
                    num_batches += 1

                    # 打印训练信息
                    if (epoch + 1) % 1 == 0 and (batch_idx + 1) % 10 == 0:
                        print(
                            'Epoch: {}/{}, Batch: {}/{}, Total Loss: {:.5f}, MAE Loss: {:.5f}, PINN Loss: {:.5f}, CLS Loss: {:.5f}'.format(
                                epoch + 1, NUM_EPOCH, batch_idx + 1, len(train_loader),
                                loss[0].item(), loss[1].item(), loss[2].item(), loss[3].item()
                            )
                        )
            except Exception as e:
                print(f"训练过程中发生错误: {str(e)}")
                import traceback
                traceback.print_exc()
                # 继续下一个 epoch

            # 计算平均训练损失
            if num_batches > 0:
                avg_train_loss = epoch_train_loss / num_batches
                train_losses.append(avg_train_loss)
            else:
                print("警告: 当前epoch没有成功处理任何批次")
                avg_train_loss = float('inf')
                train_losses.append(avg_train_loss)

            # 验证阶段
            model.eval()
            epoch_val_loss = 0.0
            num_val_batches = 0

            try:
                # 使用验证数据加载器进行批量处理
                with torch.no_grad():
                    for val_batch_idx, (inputs_val_batch, targets_p_val_batch, targets_fault_val_batch,
                                        targets_p_orig_val_batch, inputs_orig_val_batch) in enumerate(val_loader):
                        # 移动数据到设备
                        inputs_val_batch = inputs_val_batch.to(device)

                        # 修正维度问题
                        if targets_p_val_batch.dim() > 2:
                            # 调整维度，如果有多余的维度
                            while targets_p_val_batch.dim() > 2:
                                targets_p_val_batch = targets_p_val_batch.squeeze(-1)
                        # 然后确保形状为 [batch, seq_len, 1]
                        targets_p_val_batch = targets_p_val_batch.to(device).unsqueeze(-1)

                        # 同样处理原始压力数据
                        if targets_p_orig_val_batch.dim() > 2:
                            # 调整维度，如果有多余的维度
                            while targets_p_orig_val_batch.dim() > 2:
                                targets_p_orig_val_batch = targets_p_orig_val_batch.squeeze(-1)
                        # 然后确保形状为 [batch, seq_len, 1]
                        targets_p_orig_val_batch = targets_p_orig_val_batch.to(device).unsqueeze(-1)

                        targets_fault_val_batch = targets_fault_val_batch.to(device)
                        inputs_orig_val_batch = inputs_orig_val_batch.to(device)

                        # 获取原始 mdot_A 值
                        mdot_A_orig_val_batch = inputs_orig_val_batch[:, :, 2]  # [batch, seq_len]

                        # 启用梯度计算用于前向传播
                        with torch.set_grad_enabled(True):
                            # 前向传播
                            p_pred_val, p_t_pred_val, fault_logit_val = model(inputs_val_batch)

                            # 打印验证批次张量形状（仅第一轮第一批次）
                            if epoch == 0 and val_batch_idx == 0:
                                print(f"\n验证批次调试信息:")
                                print(f"inputs_val_batch 形状: {inputs_val_batch.shape}")
                                print(f"targets_p_val_batch 形状: {targets_p_val_batch.shape}")
                                print(f"targets_fault_val_batch 形状: {targets_fault_val_batch.shape}")
                                print(f"targets_p_orig_val_batch 形状: {targets_p_orig_val_batch.shape}")
                                print(f"inputs_orig_val_batch 形状: {inputs_orig_val_batch.shape}")
                                print(f"p_pred_val 形状: {p_pred_val.shape}")
                                print(f"p_t_pred_val 形状: {p_t_pred_val.shape}")
                                print(f"fault_logit_val 形状: {fault_logit_val.shape}")
                                print(f"mdot_A_orig_val 值范围: [{mdot_A_orig_val_batch.min().item():.6f}, {mdot_A_orig_val_batch.max().item():.6f}]")

                            # 计算损失
                            val_loss_batch = criterion(
                                p_pred_val, p_t_pred_val, fault_logit_val,
                                targets_p_val_batch, targets_fault_val_batch,
                                mdot_A_orig_val_batch,  # 使用原始 mdot_A 数据
                                V=2 * 1e-4,
                                targets_p_orig=targets_p_orig_val_batch,
                                mean_p=y_p_mean,
                                std_p=y_p_std
                            )

                        # 累加验证损失
                        epoch_val_loss += val_loss_batch[0].item()
                        num_val_batches += 1

                        # 显示第一个批次的损失值
                        if val_batch_idx == 0 and (epoch + 1) % 1 == 0:
                            print(
                                'Epoch: {}/{}, Val Batch: {}, Loss: {:.5f}, MAE Loss: {:.5f}, PINN Loss: {:.5f}, CLS Loss: {:.5f}'.format(
                                    epoch + 1, NUM_EPOCH, val_batch_idx,
                                    val_loss_batch[0].item(), val_loss_batch[1].item(),
                                    val_loss_batch[2].item(), val_loss_batch[3].item()
                                )
                            )
            except Exception as e:
                print(f"验证过程中发生错误: {str(e)}")
                import traceback
                traceback.print_exc()
                # 继续下一个 epoch

            # 计算平均验证损失
            if num_val_batches > 0:
                avg_val_loss = epoch_val_loss / num_val_batches
                val_losses.append(avg_val_loss)

                if best_val_loss > avg_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(),
                               'D:\cyh\project\PINN-Battery-Prognostics-main\PINN-Battery-Prognostics-main\my_model_weights.pth')
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                print("警告: 当前epoch没有成功处理任何验证批次")
                avg_val_loss = float('inf')
                val_losses.append(avg_val_loss)
                no_improvement_count += 1

            # 测试阶段
            model.eval()
            epoch_test_loss = 0.0
            num_test_batches = 0

            # 初始化当前epoch的测试数据收集列表
            all_p_pred_test = []
            all_fault_logits_test = []
            all_targets_p_test = []
            all_targets_fault_test = []
            all_targets_p_orig_test = []

            try:
                # 使用测试数据加载器进行批量处理
                with torch.no_grad():
                    for test_batch_idx, (inputs_test_batch, targets_p_test_batch, targets_fault_test_batch,
                                         targets_p_orig_test_batch, inputs_orig_test_batch) in enumerate(test_loader):
                        # 移动数据到设备
                        inputs_test_batch = inputs_test_batch.to(device)

                        # 修正维度问题
                        if targets_p_test_batch.dim() > 2:
                            # 调整维度，如果有多余的维度
                            while targets_p_test_batch.dim() > 2:
                                targets_p_test_batch = targets_p_test_batch.squeeze(-1)
                        # 然后确保形状为 [batch, seq_len, 1]
                        targets_p_test_batch = targets_p_test_batch.to(device).unsqueeze(-1)

                        # 同样处理原始压力数据
                        if targets_p_orig_test_batch.dim() > 2:
                            # 调整维度，如果有多余的维度
                            while targets_p_orig_test_batch.dim() > 2:
                                targets_p_orig_test_batch = targets_p_orig_test_batch.squeeze(-1)
                        # 然后确保形状为 [batch, seq_len, 1]
                        targets_p_orig_test_batch = targets_p_orig_test_batch.to(device).unsqueeze(-1)

                        targets_fault_test_batch = targets_fault_test_batch.to(device)
                        inputs_orig_test_batch = inputs_orig_test_batch.to(device)

                        # 获取原始 mdot_A 值
                        mdot_A_orig_test_batch = inputs_orig_test_batch[:, :, 2]  # [batch, seq_len]

                        # 启用梯度计算用于前向传播
                        with torch.set_grad_enabled(True):
                            # 前向传播
                            p_pred_test, p_t_pred_test, fault_logit_test = model(inputs_test_batch)

                            # 打印测试批次张量形状（仅第一轮第一批次）
                            if epoch == 0 and test_batch_idx == 0:
                                print(f"\n测试批次调试信息:")
                                print(f"inputs_test_batch 形状: {inputs_test_batch.shape}")
                                print(f"targets_p_test_batch 形状: {targets_p_test_batch.shape}")
                                print(f"targets_fault_test_batch 形状: {targets_fault_test_batch.shape}")
                                print(f"targets_p_orig_test_batch 形状: {targets_p_orig_test_batch.shape}")
                                print(f"inputs_orig_test_batch 形状: {inputs_orig_test_batch.shape}")
                                print(f"p_pred_test 形状: {p_pred_test.shape}")
                                print(f"p_t_pred_test 形状: {p_t_pred_test.shape}")
                                print(f"fault_logit_test 形状: {fault_logit_test.shape}")
                                print(f"mdot_A_orig_test 值范围: [{mdot_A_orig_test_batch.min().item():.6f}, {mdot_A_orig_test_batch.max().item():.6f}]")

                        # 计算测试损失
                        test_loss_batch = criterion(
                            p_pred_test, p_t_pred_test, fault_logit_test,
                            targets_p_test_batch, targets_fault_test_batch,
                            mdot_A_orig_test_batch,  # 使用原始 mdot_A 数据
                            V=2 * 1e-4,
                            targets_p_orig=targets_p_orig_test_batch,
                            mean_p=y_p_mean,
                            std_p=y_p_std
                        )

                        # 累加测试损失
                        epoch_test_loss += test_loss_batch[0].item()
                        num_test_batches += 1

                        # 显示第一个批次的损失值
                        if test_batch_idx == 0 and (epoch + 1) % 1 == 0:
                            print(
                                'Epoch: {}/{}, Test Batch: {}, Loss: {:.5f}, MAE Loss: {:.5f}, PINN Loss: {:.5f}, CLS Loss: {:.5f}'.format(
                                    epoch + 1, NUM_EPOCH, test_batch_idx,
                                    test_loss_batch[0].item(), test_loss_batch[1].item(),
                                    test_loss_batch[2].item(), test_loss_batch[3].item()
                                )
                            )

                        # 移动到CPU并转换为NumPy数组以便于后续处理
                        p_pred_test_np = p_pred_test.cpu().detach().numpy()
                        fault_logit_test_np = fault_logit_test.cpu().detach().numpy()
                        targets_p_test_np = targets_p_test_batch.cpu().detach().numpy()
                        targets_fault_test_np = targets_fault_test_batch.cpu().detach().numpy()
                        targets_p_orig_test_np = targets_p_orig_test_batch.cpu().detach().numpy()

                        # 添加批次预测和真实值到列表
                        all_p_pred_test.append(p_pred_test_np)
                        all_fault_logits_test.append(fault_logit_test_np)
                        all_targets_p_test.append(targets_p_test_np)
                        all_targets_fault_test.append(targets_fault_test_np)
                        all_targets_p_orig_test.append(targets_p_orig_test_np)

                        # 同时添加到整个训练过程的列表
                        all_epoch_p_pred_test.append(p_pred_test_np)
                        all_epoch_fault_logits_test.append(fault_logit_test_np)
                        all_epoch_targets_p_test.append(targets_p_test_np)
                        all_epoch_targets_fault_test.append(targets_fault_test_np)
                        all_epoch_targets_p_orig_test.append(targets_p_orig_test_np)
            except Exception as e:
                print(f"测试过程中发生错误: {str(e)}")
                import traceback
                traceback.print_exc()
                # 继续下一个 epoch

            # 计算平均测试损失
            if num_test_batches > 0:
                avg_test_loss = epoch_test_loss / num_test_batches
                test_losses.append(avg_test_loss)
            else:
                print("警告: 当前epoch没有成功处理任何测试批次")
                avg_test_loss = float('inf')
                test_losses.append(avg_test_loss)

            # 打印epoch结果
            print(f"Epoch {epoch + 1}/{NUM_EPOCH} completed:")
            print(f"  Train Loss: {avg_train_loss:.6f}")
            print(f"  Val Loss: {avg_val_loss:.6f}")
            print(f"  Test Loss: {avg_test_loss:.6f}")

            # 更新学习率
            scheduler.step()

        all_losses['train'].append(train_losses)
        all_losses['val'].append(val_losses)
        all_losses['test'].append(test_losses)

        # 评估模型
        print(f"评估模型中...")
        model.eval()

        # 加载最佳模型
        model.load_state_dict(torch.load(
            'D:\cyh\project\PINN-Battery-Prognostics-main\PINN-Battery-Prognostics-main\my_model_weights.pth'))
        model.eval()

        # 汇集所有测试预测和真实标签以计算最终指标
        all_p_pred_test = []
        all_fault_logits_test = []
        all_targets_p_test = []
        all_targets_fault_test = []
        all_targets_p_orig_test = []  # 用于存储原始pOut数据

        try:
            with torch.no_grad():
                # 使用测试数据加载器进行批量处理
                for test_batch_idx, (inputs_test_batch, targets_p_test_batch, targets_fault_test_batch,
                                     targets_p_orig_test_batch, inputs_orig_test_batch) in enumerate(test_loader):
                    # 移动数据到设备
                    inputs_test_batch = inputs_test_batch.to(device)

                    # 修正维度问题
                    if targets_p_test_batch.dim() > 2:
                        # 调整维度，如果有多余的维度
                        while targets_p_test_batch.dim() > 2:
                            targets_p_test_batch = targets_p_test_batch.squeeze(-1)
                    # 然后确保形状为 [batch, seq_len, 1]
                    targets_p_test_batch = targets_p_test_batch.to(device).unsqueeze(-1)

                    # 同样处理原始压力数据
                    if targets_p_orig_test_batch.dim() > 2:
                        # 调整维度，如果有多余的维度
                        while targets_p_orig_test_batch.dim() > 2:
                            targets_p_orig_test_batch = targets_p_orig_test_batch.squeeze(-1)
                    # 然后确保形状为 [batch, seq_len, 1]
                    targets_p_orig_test_batch = targets_p_orig_test_batch.to(device).unsqueeze(-1)

                    targets_fault_test_batch = targets_fault_test_batch.to(device)
                    inputs_orig_test_batch = inputs_orig_test_batch.to(device)

                    # 获取原始 mdot_A 值
                    mdot_A_orig_test_batch = inputs_orig_test_batch[:, :, 2]  # [batch, seq_len]

                    # 启用梯度计算用于前向传播
                    with torch.set_grad_enabled(True):
                        # 前向传播
                        p_pred_test, p_t_pred_test, fault_logit_test = model(inputs_test_batch)

                    # 移动到CPU并转换为NumPy数组以便于后续处理
                    p_pred_test_np = p_pred_test.cpu().detach().numpy()
                    fault_logit_test_np = fault_logit_test.cpu().detach().numpy()
                    targets_p_test_np = targets_p_test_batch.cpu().detach().numpy()
                    targets_fault_test_np = targets_fault_test_batch.cpu().detach().numpy()
                    targets_p_orig_test_np = targets_p_orig_test_batch.cpu().detach().numpy()

                    # 添加批次预测和真实值到列表
                    all_p_pred_test.append(p_pred_test_np)
                    all_fault_logits_test.append(fault_logit_test_np)
                    all_targets_p_test.append(targets_p_test_np)
                    all_targets_fault_test.append(targets_fault_test_np)
                    all_targets_p_orig_test.append(targets_p_orig_test_np)
        except Exception as e:
            print(f"最终评估时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()

        # 如果成功收集了评估数据，才计算指标
        if all_p_pred_test and all_targets_p_test:
            # 连接所有批次数据
            all_p_pred_test = np.concatenate(all_p_pred_test, axis=0)
            all_fault_logits_test = np.concatenate(all_fault_logits_test, axis=0)
            all_targets_p_test = np.concatenate(all_targets_p_test, axis=0)
            all_targets_fault_test = np.concatenate(all_targets_fault_test, axis=0)
            all_targets_p_orig_test = np.concatenate(all_targets_p_orig_test, axis=0)

            # 确保维度匹配 - 如果维度不匹配，进行调整
            if all_p_pred_test.shape != all_targets_p_orig_test.shape:
                print(
                    f"警告: 维度不匹配，调整中。 p_pred: {all_p_pred_test.shape}, targets_orig: {all_targets_p_orig_test.shape}")
                # 确保两个数组有相同的形状
                if all_targets_p_orig_test.ndim > all_p_pred_test.ndim:
                    # 降维targets_p_orig_test
                    all_targets_p_orig_test = all_targets_p_orig_test.squeeze()
                elif all_p_pred_test.ndim > all_targets_p_orig_test.ndim:
                    # 降维all_p_pred_test
                    all_p_pred_test = all_p_pred_test.squeeze()

            # 使用原始数据计算指标，无需反标准化
            print('使用原始pOut直接计算指标...')
            print(f"指标计算前的形状: p_pred: {all_p_pred_test.shape}, targets_orig: {all_targets_p_orig_test.shape}")

            # 计算压力预测指标
            rmse, mae, mape = calculate_metrics_in_batches(all_p_pred_test, all_targets_p_orig_test)
            print(all_targets_p_orig_test)
            # 计算故障分类精度
            all_fault_pred_test = (all_fault_logits_test >= 0.5).astype(int)
            if len(all_fault_pred_test) > 0 and len(all_targets_fault_test) > 0:
                accuracy = np.mean(all_fault_pred_test == all_targets_fault_test)
            else:
                accuracy = 0.0
                print("警告: 无法计算分类精度，预测或目标数据为空")

            print('最终测试集评估指标:')
            print('RMSE: {:.4f}'.format(rmse))
            print('MAE: {:.4f}'.format(mae))
            print('MAPE: {:.4f}%'.format(mape))
            print('故障分类精度: {:.4f}'.format(accuracy))

            # 将结果保存到文件
            results = {
                'p_true': all_targets_p_orig_test.flatten(),
                'p_pred': all_p_pred_test.flatten(),
                'fault_true': all_targets_fault_test.flatten(),
                'fault_pred': all_fault_pred_test.flatten(),
                'fault_prob': 1 / (1 + np.exp(-all_fault_logits_test)).flatten() if isinstance(all_fault_logits_test,
                                                                                               np.ndarray) else all_fault_logits_test.flatten(),
                'metrics': {
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape,
                    'accuracy': accuracy
                }
            }
            torch.save(results, 'multitask_results.pth')

            # 创建和保存损失曲线
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.plot(test_losses, label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss Curves')
            plt.legend()
            plt.savefig('multitask_loss_curve.png')
            plt.close()

            print('训练完成。结果已保存。')
        else:
            print("警告: 无法计算最终评估指标，因为评估数据收集失败")

    # 4. 计算多轮平均结果
    print("\n4. 计算多轮平均结果...")

    metric_mean = np.mean(metric_rounds['test'])
    metric_std = np.std(metric_rounds['test'])
    print(f"\n平均指标 ({NUM_ROUNDS}轮):")
    print(f"测试集 RMSE: {metric_mean:.6f} ± {metric_std:.6f}")

    # 将最终测试数据合并并计算指标
    if len(all_epoch_p_pred_test) > 0:
        print(f"合并整个训练过程中收集的测试数据...")
        # 连接所有批次数据
        try:
            all_p_pred_test = np.concatenate(all_epoch_p_pred_test, axis=0)
            all_fault_logits_test = np.concatenate(all_epoch_fault_logits_test, axis=0)
            all_targets_p_test = np.concatenate(all_epoch_targets_p_test, axis=0)
            all_targets_p_orig_test = np.concatenate(all_epoch_targets_p_orig_test, axis=0)
            all_targets_fault_test = np.concatenate(all_epoch_targets_fault_test, axis=0)

            print(f"合并后的数据形状:")
            print(f"预测压力: {all_p_pred_test.shape}")
            print(f"预测故障: {all_fault_logits_test.shape}")
            print(f"真实压力: {all_targets_p_test.shape}")
            print(f"原始压力: {all_targets_p_orig_test.shape}")
            print(f"真实故障: {all_targets_fault_test.shape}")
        except Exception as e:
            print(f"合并测试数据时出错: {e}")
            # 创建空数组作为替代
            all_p_pred_test = np.array([])
            all_fault_logits_test = np.array([])
            all_targets_p_test = np.array([])
            all_targets_p_orig_test = np.array([])
            all_targets_fault_test = np.array([])
    else:
        print("警告: 训练过程中未收集到任何测试数据")
        # 创建空数组
        all_p_pred_test = np.array([])
        all_fault_logits_test = np.array([])
        all_targets_p_test = np.array([])
        all_targets_p_orig_test = np.array([])
        all_targets_fault_test = np.array([])

    # 绘制损失函数走势图
    plt.figure(figsize=(12, 6))

    # 计算所有轮次的平均损失
    if all_losses['train'] and all(isinstance(losses, list) and losses for losses in all_losses['train']):
        avg_train_losses = np.mean([losses for losses in all_losses['train'] if losses], axis=0)
        avg_val_losses = np.mean([losses for losses in all_losses['val'] if losses], axis=0)
        avg_test_losses = np.mean([losses for losses in all_losses['test'] if losses], axis=0)

        # 绘制训练损失和验证损失
        plt.plot(range(1, len(avg_train_losses) + 1), avg_train_losses, label='训练损失')
        plt.plot(range(1, len(avg_val_losses) + 1), avg_val_losses, label='验证损失')
        plt.plot(range(1, len(avg_test_losses) + 1), avg_test_losses, label='测试损失')

        plt.xlabel('训练轮次')
        plt.ylabel('损失值')
        plt.title(f'损失函数走势图 (平均值，{NUM_ROUNDS}轮)')
        plt.legend()
        plt.grid(True)

        # 保存图片
        plt.savefig('multitask_loss_curve.png', dpi=300, bbox_inches='tight')
        print("\n损失函数走势图已保存到 'multitask_loss_curve.png'")

    # 5. 保存最终模型和结果
    print("\n5. 保存结果...")

    # 检查评估数据是否存在
    if len(all_p_pred_test) > 0 and len(all_targets_p_orig_test) > 0:
        # 确保维度匹配 - 如果维度不匹配，进行调整
        if all_p_pred_test.shape != all_targets_p_orig_test.shape:
            print(
                f"警告: 维度不匹配，调整中。 p_pred: {all_p_pred_test.shape}, targets_orig: {all_targets_p_orig_test.shape}")

            # 缩减维度以匹配
            while all_p_pred_test.ndim > all_targets_p_orig_test.ndim:
                all_p_pred_test = all_p_pred_test.squeeze(-1)
            while all_targets_p_orig_test.ndim > all_p_pred_test.ndim:
                all_targets_p_orig_test = all_targets_p_orig_test.squeeze(-1)

            # 确保形状相同
            min_shape = [min(s1, s2) for s1, s2 in zip(all_p_pred_test.shape, all_targets_p_orig_test.shape)]
            all_p_pred_test = all_p_pred_test[tuple(slice(0, s) for s in min_shape)]
            all_targets_p_orig_test = all_targets_p_orig_test[tuple(slice(0, s) for s in min_shape)]

        # 使用calculate_metrics_in_batches计算最终指标
        print(f"计算指标前的数据形状: p_pred: {all_p_pred_test.shape}, targets_orig: {all_targets_p_orig_test.shape}")
        rmse, mae, mape = calculate_metrics_in_batches(all_p_pred_test, all_targets_p_orig_test)

        # 计算故障分类精度
        if len(all_fault_logits_test) > 0 and len(all_targets_fault_test) > 0:
            all_fault_pred_test = (all_fault_logits_test >= 0.5).astype(int)
            accuracy = np.mean(all_fault_pred_test == all_targets_fault_test)
        else:
            accuracy = 0.0
            all_fault_pred_test = np.array([])
            print("警告: 无法计算分类精度，预测或目标数据为空")

        print('最终测试集评估指标:')
        print('RMSE: {:.4f}'.format(rmse))
        print('MAE: {:.4f}'.format(mae))
        print('MAPE: {:.4f}%'.format(mape))
        print('故障分类精度: {:.4f}'.format(accuracy))

        # 保存结果到字典
        results = {
            'p_true': all_targets_p_orig_test.flatten() if len(all_targets_p_orig_test.shape) > 0 else np.array([]),
            'p_pred': all_p_pred_test.flatten() if len(all_p_pred_test.shape) > 0 else np.array([]),
            'fault_true': all_targets_fault_test.flatten() if len(all_targets_fault_test.shape) > 0 else np.array([]),
            'fault_pred': all_fault_pred_test.flatten() if len(all_fault_pred_test.shape) > 0 else np.array([]),
            'fault_prob': 1 / (1 + np.exp(-all_fault_logits_test)).flatten() if len(
                all_fault_logits_test.shape) > 0 else np.array([]),
            'metrics': {
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'accuracy': accuracy
            }
        }

        torch.save(results, 'multitask_results.pth')
        print("结果已保存到 'multitask_results.pth'")
    else:
        print("警告: 无法保存最终结果，评估数据不完整")

    # 在main函数中添加调用
    print("\n演示滑动窗口工作原理:")
    visualize_sliding_window(X_train, SEQ_LEN, num_examples=3)

    # 输出标准化前后的mdot_A值范围，用于验证修改
    print("mdot_A 值范围对比:")
    print(f"原始 mdot_A 值范围: [{X[:, 2].min():.6f}, {X[:, 2].max():.6f}]")
    print(f"标准化后 mdot_A 值范围: [{X_norm[:, 2].min():.6f}, {X_norm[:, 2].max():.6f}]")
    print(f"mdot_A 标准化参数: 均值={X_mean[2].item():.6f}, 标准差={X_std[2].item():.6f}")

    return model


if __name__ == "__main__":
    main()
