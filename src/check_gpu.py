import torch

print("PyTorch version:", torch.__version__)
print("CUDA available: ", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:            ", torch.cuda.get_device_name(0))
    print("Built for CUDA: ", torch.version.cuda)
    print("Supports sm_120:", "sm_120" in torch.cuda.get_arch_list())
    x = torch.randn(4000, 4000, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("GPU test calculation: OK")
