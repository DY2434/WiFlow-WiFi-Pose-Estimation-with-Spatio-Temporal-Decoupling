import torch
from thop import profile

config ={
    "Original": "/home/jackson-devworks/Desktop/ECCV_2024/output/AWGN/Original/0.1/best.pt",
    "1-layer": "/home/jackson-devworks/Desktop/ECCV_2024/output/SPN/OneLayerDenosing/0.1/best.pt",
    "2-layer": "/home/jackson-devworks/Desktop/ECCV_2024/output/SPN/TwoLayerDenosing/0.1/best.pt",
    "3-layer": "/home/jackson-devworks/Desktop/ECCV_2024/output/SPN/ThreeLayerDenosing/0.1/best.pt",
    "4-layer": "/home/jackson-devworks/Desktop/ECCV_2024/output/SPN/FourLayerDenosing/0.1/best.pt",
    "5-layer": "/home/jackson-devworks/Desktop/ECCV_2024/output/SPN/FiveLayerDenosing/0.1/best.pt",
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for k,v in config.items():
    model = torch.load(v)
    model = model.to(device)
    input_tensor = torch.randn(1, 3, 114, 10).to(device)  # Example input shape
    flops, params = profile(model, inputs=(input_tensor,))
    print(f'in {k}, FLOPs: {flops}, Parameters: {params}')
