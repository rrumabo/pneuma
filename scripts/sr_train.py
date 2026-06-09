#!/usr/bin/env python3
import argparse, glob, json, math, os, random
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def psnr(x, y, eps=1e-10):
    # x,y in [0,1], tensor
    mse = torch.mean((x - y) ** 2)
    return 20.0 * torch.log10(1.0 / torch.sqrt(mse + eps))

def to_uint8(img01: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img01 * 255.0), 0, 255).astype(np.uint8)

def load_hr_fields(cache_glob: str) -> list[np.ndarray]:
    paths = sorted(glob.glob(cache_glob or ""))
    hr_list: list[np.ndarray] = []
    for p in paths:
        try:
            z = np.load(p)
            # prefer 'h0', else first array
            if "h0" in z:
                a = np.array(z["h0"])
            else:
                # take first key that is 2D
                key = next((k for k in z.files if z[k].ndim >= 2), None)
                if key is None: continue
                a = np.array(z[key])
                # squeeze to 2D if time/channel exists
                while a.ndim > 2:
                    a = a[0]
            a = np.asarray(a, dtype=np.float32)
            # normalize per-field to [0,1]
            amin, amax = float(np.nanmin(a)), float(np.nanmax(a))
            if not np.isfinite(amin) or not np.isfinite(amax) or amax <= amin:
                continue
            a = (a - amin) / (amax - amin)
            a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
            hr_list.append(a)
        except Exception:
            continue
    return hr_list

def synth_fields(n=8, Ny=192, Nx=192) -> list[np.ndarray]:
    xs = np.linspace(0, 2*np.pi, Nx, dtype=np.float32)
    ys = np.linspace(0, 2*np.pi, Ny, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    out = []
    for i in range(n):
        f = ( np.sin((i+1)*X)*0.4 + np.cos((i+2)*Y)*0.3
            + np.sin(0.5*X+0.8*Y)*0.2 )
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        out.append(f.astype(np.float32))
    return out

class SRPatchSet(Dataset):
    def __init__(self, fields: list[np.ndarray], patch: int, scale: int):
        self.fields = fields
        self.patch = patch
        self.scale = scale
        assert scale in (2,4), "scale must be 2 or 4"

    def __len__(self):
        # virtual length to allow random sampling
        return max(2000, len(self.fields)*200)

    def __getitem__(self, idx):
        f = random.choice(self.fields)  # (H,W) in [0,1]
        H, W = f.shape
        p = self.patch
        if H < p or W < p:
            # pad reflect if needed
            pad_h = max(0, p - H); pad_w = max(0, p - W)
            f = np.pad(f, ((pad_h,0),(pad_w,0)), mode="reflect")
            H, W = f.shape
        y0 = random.randrange(0, H - p + 1)
        x0 = random.randrange(0, W - p + 1)
        hr = f[y0:y0+p, x0:x0+p]  # (p,p)

        # make LR by downsampling, then upsample back (bicubic)
        t = torch.from_numpy(hr).unsqueeze(0).unsqueeze(0)      # (1,1,p,p)
        lr = F.interpolate(t, size=(p//self.scale, p//self.scale), mode="bicubic", align_corners=False)
        lr_up = F.interpolate(lr, size=(p,p), mode="bicubic", align_corners=False)
        return lr_up.squeeze(0), t.squeeze(0)   # both (1,p,p)

class SRCNNSmall(nn.Module):
    # Simple SRCNN-style: upsample outside, then refine
    def __init__(self, c=1):
        super().__init__()
        self.c1 = nn.Conv2d(c, 64, 9, padding=4)
        self.c2 = nn.Conv2d(64, 32, 1)
        self.c3 = nn.Conv2d(32, c, 5, padding=2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.c1(x))
        x = self.act(self.c2(x))
        x = self.c3(x)
        return torch.clamp(x, 0.0, 1.0)

def save_triptych(path: Path, hr: torch.Tensor, bic: torch.Tensor, pred: torch.Tensor):
    # tensors (1,1,H,W) on CPU in [0,1]
    hr8  = to_uint8(hr.squeeze().numpy())
    b8   = to_uint8(bic.squeeze().numpy())
    p8   = to_uint8(pred.squeeze().numpy())
    Wtot = hr8.shape[1]*3
    H    = hr8.shape[0]
    canvas = Image.new("L", (Wtot, H))
    canvas.paste(Image.fromarray(hr8),  (0,0))
    canvas.paste(Image.fromarray(b8),   (hr8.shape[1],0))
    canvas.paste(Image.fromarray(p8),   (2*hr8.shape[1],0))
    canvas.save(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config)) or {}
    data = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    out = cfg.get("out", {})

    cache_glob = data.get("cache_glob", "")
    patch = int(data.get("patch", 64))
    scale = int(data.get("scale", 2))

    epochs = int(train_cfg.get("epochs", 3))
    bs = int(train_cfg.get("batch_size", 16))
    lr = float(train_cfg.get("lr", 1e-3))
    val_split = float(train_cfg.get("val_split", 0.1))
    steps_per_epoch = train_cfg.get("steps_per_epoch", 200)
    seed = int(train_cfg.get("seed", 42))

    out_dir = Path(out.get("dir","outputs/sr")); out_dir.mkdir(parents=True, exist_ok=True)
    save_samples = int(out.get("save_samples", 4))

    set_seed(seed)
    fields = load_hr_fields(cache_glob)
    if not fields:
        print("No cache matched; using synthetic fields.")
        fields = synth_fields(n=8)

    dataset = SRPatchSet(fields, patch=patch, scale=scale)
    n_total = len(fields)
    # Build a small validation set from full-res fields (not from virtual dataset length)
    val_count = max(1, math.ceil(n_total * val_split))
    train_fields = fields[:-val_count] if n_total > 1 else fields
    val_fields   = fields[-val_count:] if n_total > 1 else fields

    train_ds = SRPatchSet(train_fields, patch=patch, scale=scale)
    val_ds   = SRPatchSet(val_fields,   patch=patch, scale=scale)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRCNNSmall().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_psnr = -1.0
    metrics_log = []

    for ep in range(1, epochs+1):
        model.train()
        iters = 0
        for lr_up, hr in train_dl:
            lr_up = lr_up.to(device)      # (B,1,H,W)
            hr    = hr.to(device)
            pred = model(lr_up)
            loss = F.mse_loss(pred, hr)
            opt.zero_grad(); loss.backward(); opt.step()
            iters += 1
            if steps_per_epoch and iters >= steps_per_epoch:
                break

        # ----- validation -----
        model.eval()
        psnr_bic, psnr_cnn = [], []
        samp_saved = 0
        with torch.no_grad():
            for lr_up, hr in val_dl:
                lr_up = lr_up.to(device); hr = hr.to(device)
                # bicubic baseline is just the input (already upsampled)
                bic = lr_up
                pred = model(lr_up)
                psnr_bic.append(psnr(bic, hr).item())
                psnr_cnn.append(psnr(pred, hr).item())

                # save a few
                if samp_saved < save_samples:
                    save_triptych(
                        out_dir / f"ep{ep:02d}_sample{samp_saved:02d}.png",
                        hr.cpu(), bic.cpu(), pred.cpu()
                    )
                    samp_saved += 1

        m_bic = float(np.mean(psnr_bic)) if psnr_bic else float("nan")
        m_cnn = float(np.mean(psnr_cnn)) if psnr_cnn else float("nan")
        print(f"[ep {ep}] PSNR bicubic={m_bic:.2f}  cnn={m_cnn:.2f}")

        metrics_log.append({"epoch": ep, "psnr_bic": m_bic, "psnr_cnn": m_cnn})
        with open(out_dir/"metrics.json","w") as f:
            json.dump(metrics_log, f, indent=2)

        if m_cnn > best_psnr:
            best_psnr = m_cnn
            torch.save(model.state_dict(), out_dir/"model_best.pt")

    # final save
    torch.save(model.state_dict(), out_dir/"model_last.pt")
    print(f"Done. Best PSNR (cnn) = {best_psnr:.2f} dB. Artifacts in {out_dir}")

if __name__ == "__main__":
    main()