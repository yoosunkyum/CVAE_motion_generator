import torch

a = torch.randn((9,3,1))
print(a)

dataset = torch.utils.data.TensorDataset(a)
loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

for x in loader:
    print(x)
