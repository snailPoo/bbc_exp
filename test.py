import torch
from model.model_pool import Channel_wise_AR

model = Channel_wise_AR(xorz=False, z1_cond=False, kernel_size=3, hidden_size=32, num_layers=1, dp_rate=0)

x = torch.randn((1,32,4,4))
D_params = model(x)

hidden = None
lstm_input = torch.zeros((1, 1, 1, 4, 4)) # z3: (B, c=1, 1, h, w)
param, hidden = model.ar_model(lstm_input, hidden=hidden) # z3: (B, c=1, 2, h, w)
print(f'{param}\n{D_params[:,0,]}')# (B, 2, h, w)
param, hidden = model.ar_model(x[:,1,].unsqueeze(1).unsqueeze(1), hidden=hidden) # z3: (B, c=1, 2, h, w)
print(f'{param}\n{D_params[:,1,]}')# (B, 2, h, w)
