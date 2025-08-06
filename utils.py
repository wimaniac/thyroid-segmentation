import torch
def load_model_checkpoint(model, checkpoint_path, device='cpu'):
    state_dict = torch.load(checkpoint_path, map_location=device)
    # Nếu đã lưu bằng model.state_dict()
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
