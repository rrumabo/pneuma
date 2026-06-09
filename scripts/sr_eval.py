# scripts/sr_eval.py
import argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from PIL import Image

def to01(a):
    a = np.asarray(a, np.float32)
    mn, mx = float(np.nanmin(a)), float(np.nanmax(a))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn: return np.zeros_like(a)
    a = (a - mn) / (mx - mn)
    return np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)

def to_uint8(a01): return np.clip(np.rint(a01*255),0,255).astype(np.uint8)

class SRCNNSmall(nn.Module):
    def __init__(self, c=1):
        super().__init__()
        self.c1 = nn.Conv2d(c, 64, 9, padding=4)
        self.c2 = nn.Conv2d(64, 32, 1)
        self.c3 = nn.Conv2d(32, c, 5, padding=2)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.act(self.c1(x)); x = self.act(self.c2(x)); x = self.c3(x)
        return torch.clamp(x, 0.0, 1.0)

def save_triptych(path: Path, hr, bic, pred):
    hr8, b8, p8 = map(lambda t: to_uint8(t.squeeze().cpu().numpy()), (hr, bic, pred))
    H, W = hr8.shape
    canvas = Image.new("L", (3*W, H))
    canvas.paste(Image.fromarray(hr8), (0,0))
    canvas.paste(Image.fromarray(b8),  (W,0))
    canvas.paste(Image.fromarray(p8),  (2*W,0))
    canvas.save(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--input", required=True, help="NPZ cache (e.g., cache/*.npz)")
    ap.add_argument("-m","--model", required=True, help="trained .pt (e.g., outputs/sr/model_best.pt)")
    ap.add_argument("-o","--out",   default="outputs/sr_eval.png")
    ap.add_argument("-s","--scale", type=int, default=2)
    args = ap.parse_args()

    z = np.load(args.input)
    hr = z["h0"] if "h0" in z else next(np.asarray(z[k]) for k in z.files if z[k].ndim>=2)
    while hr.ndim>2: hr = hr[0]
    hr = to01(hr)  # (H,W) in [0,1]

    H, W = hr.shape
    t_hr = torch.from_numpy(hr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    # build LR by downsampling then bicubic-upsample to HR size
    lr = F.interpolate(t_hr, size=(H//args.scale, W//args.scale), mode="bicubic", align_corners=False)
    bic = F.interpolate(lr, size=(H, W), mode="bicubic", align_corners=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRCNNSmall().to(device).eval()
    model.load_state_dict(torch.load(args.model, map_location=device))

    with torch.no_grad():
        pred = model(bic.to(device)).cpu()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_triptych(Path(args.out), t_hr, bic, pred)
    print(f"Saved {args.out} (HR | Bicubic | CNN)")

if __name__ == "__main__":
    main()