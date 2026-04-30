import torch


def get_device() -> torch.device:
    """
    Usa CUDA si está disponible; si no, usa CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_device_info(device: torch.device) -> None:
    print(f"[Device] Usando: {device}")

    if device.type == "cuda":
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Device] CUDA disponible: {torch.cuda.is_available()}")