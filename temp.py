import torch
print(torch.__version__)
print(torch.version.cuda)  # Phiên bản CUDA mà PyTorch sử dụng
print(torch.cuda.is_available())  # Kiểm tra xem PyTorch có nhận GPU không
print(torch.cuda.get_device_name(0))