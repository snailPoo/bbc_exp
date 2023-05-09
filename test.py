import torch
from model.model_pool import Channel_wise_AR

model = Channel_wise_AR(xorz=False, z1_cond=False, kernel_size=3, hidden_size=4, num_layers=1, dp_rate=0)

x = torch.randn((1,10,3,3))
D_params = model(x)

hidden = None
lstm_input = torch.zeros((1, 1, 1, 3, 3)) # z3: (B, c=1, 1, h, w)
param, hidden = model.ar_model(lstm_input, hidden=hidden) # z3: (B, c=1, 2, h, w)
# print(f'single channel input:\n{param}\nall channel input\n{D_params[:,0,]}')# (B, 2, h, w)
# print(f'hidden\n{hidden}')
for i in range(9):
    param, hidden = model.ar_model(x[:,i,].unsqueeze(1).unsqueeze(1), hidden=hidden) # z3: (B, c=1, 2, h, w)
    print(f'single channel input:\n{param}\nall channel input\n{D_params[:,i+1,]}')# (B, 2, h, w)
