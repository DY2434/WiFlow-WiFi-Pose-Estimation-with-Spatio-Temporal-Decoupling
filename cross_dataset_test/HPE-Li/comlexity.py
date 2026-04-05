import torch
from thop import profile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize the model
model = torch.load("/home/aip/Downloads/LKZ/wiflow_mmfi/HPE-Li-ECCV2024-main/att_mmfi/best.pt").to(device)
# Define the specific input size
input_size = (1, 3, 114, 10)  
input_tensor = torch.randn(input_size).to(device)

# Use `thop` to calculate FLOPs and parameters
flops, params = profile(model, inputs=(input_tensor, ))

print(f"Number of Parameters: {params/1e6}")
print(f"FLOPs: {flops/1e9}")
