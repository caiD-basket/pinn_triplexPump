import numpy as np
import torch
from torch import nn
import pandas as pd


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
#TODO 确定密度与压力的梯度函数，以及函数里面的参数
data = pd.read_csv('PumpData4.csv')
x = data.drop(['pOut','fault'],axis=1)
y = data['pOut']
train_size = int(len(x)*0.8)
X_train = x.iloc[:train_size]
y_train = y.iloc[:train_size]
X_test = x.iloc[train_size:]
y_test = y.iloc[train_size:]

# 将数据转换为 torch.Tensor 格式
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
print(X_test_tensor.shape)
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
mynet = MyNeuralNet().to(device)
loss_func = nn.MSELoss()
_Y = mynet(x)
loss_value = loss_func(_Y,Y)
print(loss_value)
# tensor(127.4498, device='cuda:0', grad_fn=<MseLossBackward>)

