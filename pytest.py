import torch
first_name = 'ADA'
last_name = 'LOVELACE'
full_name = f"{first_name} {last_name}"
print(f"Hello,{full_name.title()}")

a = torch.arange(6).reshape((1, 2, 3))
b = torch.arange(2).reshape((2, 1))
print(a)
print(b)
b * a