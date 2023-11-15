import torch
import torch.nn as nn

x = torch.rand(256)
noise = 0.2 * torch.randn(x.size())
k = 2
b = 5
y = k * x + b + noise


class LinearModel(nn.Module):
    def __init__(self, in_fea, out_fea):
        super(LinearModel, self).__init__()
        self.out = nn.Linear(in_features=in_fea, out_features=out_fea)

    def forward(self, x):
        x = self.out(x)
        return x


model = LinearModel(in_fea=256, out_fea=256)

loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

for step in range(400):
    pred = model(x)
    loss = loss_func(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 20 == 0:
        print('Step {}, Loss: {:.4f}'.format(step, loss.item()))
