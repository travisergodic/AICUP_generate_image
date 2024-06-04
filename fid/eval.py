import torch
from tqdm import tqdm

@torch.no_grad()
def make_image_features(loader, model, device):
    model.eval()
    feats, names = [], []
    for batch in tqdm(loader):
        data = batch["data"].to(device)
        name = batch["name"]
        with torch.cuda.amp.autocast(enabled=True):
            feat = model(data).cpu()

        feats.append(feat)
        names += name
    feats = torch.concat(feats, dim=0).numpy()
    return feats, names