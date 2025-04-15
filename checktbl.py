import torch
print("CUDA доступен:", torch.cuda.is_available())
print("Устройство:", torch.cuda.get_device_name(0))
