# resunet_ensemble_p10_p50_p90_uncertainty.py
# ResUNet-only Deep Ensemble with p10/p50/p90 uncertainty, CRPS, coverage, PINAW, pinball losses, and rich maps.

import os, random, h5py, numpy as np, pandas as pd, imageio.v2 as imageio
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

# ---------------------------
# Dataset (compatible with your current HDF5 layout)
# ---------------------------
class CO2DatasetNoRelPerm(Dataset):
    """
    Inputs: [img, p, Ux, Uy, porosity_map, permeability_map] (6,H,W)
    Target: CO2 sequence from (1 - alpha_water) * img → [T,H,W]
    """
    def __init__(self, root_dirs, target_size=256, forecast_steps=10):
        self.file_list = []
        for root in root_dirs:
            if not os.path.isdir(root):  # skip missing dirs
                continue
            for fname in os.listdir(root):
                if fname.endswith('.hdf5'):
                    self.file_list.append(os.path.join(root, fname))
        self.target_size = target_size
        self.forecast_steps = forecast_steps

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        h5file = self.file_list[idx]
        with h5py.File(h5file, 'r') as f:
            img = f['img'][:].astype(np.float32)     # [H,W]
            p   = f['p'][-1].astype(np.float32)      # [H,W] final
            ux  = f['Ux'][-1].astype(np.float32)     # [H,W]
            uy  = f['Uy'][-1].astype(np.float32)     # [H,W]

            def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-8)
            img, p, ux, uy = norm(img), norm(p), norm(ux), norm(uy)
            input_stack = np.stack([img, p, ux, uy])  # (4,H,W)

            # optional CSV features (porosity, permeability)
            folder = os.path.dirname(h5file)
            csv_folder = [d for d in os.listdir(folder) if d.endswith('csv_files')]
            porosity, permeability = 0.0, 0.0
            if csv_folder:
                csv_path = os.path.join(folder, csv_folder[0], 'poroPerm.csv')
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    porosity = float(df['porosity'].mean()) if 'porosity' in df.columns else 0.0
                    permeability = float(df['permeability'].mean()) if 'permeability' in df.columns else 0.0

            features = np.array([porosity, permeability], dtype=np.float32)
            features_map = np.repeat(features[:, None], self.target_size * self.target_size, axis=1
                            ).reshape(2, self.target_size, self.target_size)

            # resize inputs to target_size
            input_stack_t = torch.tensor(input_stack)
            input_stack_t = F.interpolate(
                input_stack_t.unsqueeze(0), size=(self.target_size, self.target_size),
                mode='bilinear', align_corners=False
            ).squeeze(0)  # [4,TH,TW]
            input_stack = np.concatenate([input_stack_t.numpy(), features_map], axis=0)  # (6,TH,TW)

            # target sequence
            alpha_water = f['alpha_water'][-self.forecast_steps:].astype(np.float32)  # [T,H,W]
            co2_seq = (1.0 - alpha_water) * img                                       # [T,H,W]
            co2_seq = torch.tensor(co2_seq)
            co2_seq = torch.stack([
                F.interpolate(co2_seq[i][None, None, :, :], size=(self.target_size, self.target_size),
                              mode='bilinear', align_corners=False).squeeze()
                for i in range(self.forecast_steps)
            ])  # [T,TH,TW]

        return torch.tensor(input_stack), co2_seq, features

# ---------------------------
# ResUNet backbone (with residual blocks and optional dropout)
# ---------------------------
class ResidualBlock(nn.Module):
    def __init__(self, c, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(c)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(c)
        self.do    = nn.Dropout2d(dropout) if dropout>0 else nn.Identity()

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.do(self.bn2(self.conv2(y)))
        return self.relu(x + y)

class ResUNetForecast(nn.Module):
    """UNet-style encoder/decoder with residual blocks; outputs [B, T, H, W]."""
    def __init__(self, in_ch, out_ch, feats=(64,128,256,512), blocks_per_stage=2, dropout=0.0):
        super().__init__()
        self.enc_convs, self.enc_res, self.pools = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        c = in_ch
        for f in feats:
            self.enc_convs.append(nn.Conv2d(c, f, 3, padding=1))
            self.enc_res.append(nn.Sequential(*[ResidualBlock(f, dropout) for _ in range(blocks_per_stage)]))
            self.pools.append(nn.MaxPool2d(2))
            c = f

        self.bottleneck = nn.Sequential(
            nn.Conv2d(feats[-1], feats[-1]*2, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(feats[-1]*2, dropout),
        )

        self.up_convs, self.merge_convs, self.dec_res = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        up_in = feats[-1]*2
        for f in reversed(feats):
            self.up_convs.append(nn.ConvTranspose2d(up_in, f, 2, stride=2))
            self.merge_convs.append(nn.Conv2d(f + f, f, kernel_size=1))
            self.dec_res.append(nn.Sequential(*[ResidualBlock(f, dropout) for _ in range(blocks_per_stage)]))
            up_in = f

        self.out = nn.Conv2d(feats[0], out_ch, 1)

    def forward(self, x):
        skips, c = [], x
        for conv, res, pool in zip(self.enc_convs, self.enc_res, self.pools):
            c = F.relu(conv(c), inplace=True)
            c = res(c)
            skips.append(c)
            c = pool(c)

        c = self.bottleneck(c)
        skips = skips[::-1]
        for up, merge, res, sk in zip(self.up_convs, self.merge_convs, self.dec_res, skips):
            c = up(c)
            if c.shape[-2:] != sk.shape[-2:]:
                diffY = sk.size(2) - c.size(2)
                diffX = sk.size(3) - c.size(3)
                c = F.pad(c, [0, max(diffX, 0), 0, max(diffY, 0)])
                if diffY < 0 or diffX < 0:
                    sk = sk[..., :c.size(2), :c.size(3)]
            c = res(merge(torch.cat([c, sk], dim=1)))
        return torch.sigmoid(self.out(c))  # [B, T, H, W]

# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def bootstrap_subset(dataset, frac=1.0, seed=0):
    """Bootstrap (with replacement) indices from a Subset or Dataset."""
    if isinstance(dataset, Subset):
        # Map bootstrapped picks to original indices
        base_indices = dataset.indices
        rng = np.random.default_rng(seed)
        n = int(len(base_indices) * frac)
        picks = rng.integers(0, len(base_indices), size=n)
        return Subset(dataset.dataset, [base_indices[i] for i in picks])
    else:
        rng = np.random.default_rng(seed)
        n = int(len(dataset) * frac)
        picks = rng.integers(0, len(dataset), size=n)
        return Subset(dataset, picks.tolist())

def crps_ensemble(y_true, y_samples):
    """
    Approximate CRPS using ensemble samples (Gneiting & Raftery, 2007).
    y_true: [T,H,W], y_samples: [M,T,H,W]
    """
    M = y_samples.shape[0]
    term1 = np.mean(np.abs(y_samples - y_true), axis=0)
    diffs = np.abs(y_samples[:, None, ...] - y_samples[None, :, ...])  # [M,M,T,H,W]
    term2 = np.mean(diffs, axis=(0,1))
    return term1 - 0.5 * term2  # [T,H,W]

def pinball_loss(y_true, y_pred_q, tau=0.5):
    """Quantile (pinball) loss averaged over all pixels and times."""
    diff = y_true - y_pred_q
    return float(np.mean(np.maximum(tau*diff, (tau-1)*diff)))

def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

# ---------------------------
# Train a single ensemble member
# ---------------------------
def train_member(member_id, train_loader, device, forecast_steps, epochs=20, dropout=0.1, lr=1e-3, save_ckpt=True):
    name = f"ResUNet_member{member_id}"
    model = ResUNetForecast(in_ch=6, out_ch=forecast_steps, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        model.train(); running = 0.0
        for x, y, _ in tqdm(train_loader, desc=f"{name} Epoch {ep+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        print(f"{name} epoch {ep+1} loss: {running/len(train_loader):.4f}")

    if save_ckpt:
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/{name}.pt")
    return model

# ---------------------------
# Validate a single member (optionally with MC Dropout)
# ---------------------------
def validate_member(model, val_loader, device, forecast_steps, member_id, use_mc=False, mc_samples=8):
    model.eval()
    member_preds = []
    sample_counter = 0
    for x, y, _ in val_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            if use_mc:
                enable_mc_dropout(model)
                preds_mc = [model(x).cpu().numpy() for _ in range(mc_samples)]  # [mc,1,T,H,W]
                pred = np.mean(np.stack(preds_mc, axis=0), axis=0)               # [1,T,H,W]
            else:
                pred = model(x).cpu().numpy()                                   # [1,T,H,W]

        member_preds.append(pred[0])  # [T,H,W]

        # quick QA plots for first few
        if sample_counter < 3:
            gt_seq = y.cpu().numpy()[0]
            pr_seq = pred[0]
            abs_err_seq = np.abs(gt_seq - pr_seq)
            fig, axs = plt.subplots(gt_seq.shape[0], 3, figsize=(12, 3*gt_seq.shape[0]))
            for t in range(gt_seq.shape[0]):
                im0 = axs[t,0].imshow(gt_seq[t], cmap='viridis'); axs[t,0].set_title(f'GT t{t+1}'); axs[t,0].axis('off')
                c0 = plt.colorbar(im0, ax=axs[t,0], fraction=0.046, pad=0.04); c0.set_label('CO₂ sat', rotation=270, labelpad=12)
                im1 = axs[t,1].imshow(pr_seq[t], cmap='viridis'); axs[t,1].set_title(f'Pred t{t+1}'); axs[t,1].axis('off')
                c1 = plt.colorbar(im1, ax=axs[t,1], fraction=0.046, pad=0.04); c1.set_label('CO₂ sat', rotation=270, labelpad=12)
                im2 = axs[t,2].imshow(abs_err_seq[t], cmap='hot'); axs[t,2].set_title('|Error|'); axs[t,2].axis('off')
                c2 = plt.colorbar(im2, ax=axs[t,2], fraction=0.046, pad=0.04); c2.set_label('|Error|', rotation=270, labelpad=12)
            fig.legend(handles=[
                Patch(facecolor='none', edgecolor='black', label='GT (viridis)'),
                Patch(facecolor='none', edgecolor='black', label='Pred (viridis)'),
                Patch(facecolor='none', edgecolor='black', label='Abs Error (hot)')],
                loc='upper center', ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.02))
            plt.suptitle(f'ResUNet member {member_id} – val sample {sample_counter+1}')
            plt.tight_layout(rect=[0,0,1,0.98])
            plt.savefig(f"member{member_id}_val_sample_{sample_counter+1}.png", dpi=180); plt.close()

            # GIF
            images=[]
            for t in range(gt_seq.shape[0]):
                fig = plt.figure(figsize=(6,3))
                ax1 = plt.subplot(1,2,1); ax1.imshow(gt_seq[t], cmap='viridis'); ax1.set_title('GT'); ax1.axis('off')
                ax2 = plt.subplot(1,2,2); ax2.imshow(pr_seq[t], cmap='viridis'); ax2.set_title('Pred'); ax2.axis('off')
                handles=[Line2D([0],[0], linestyle='none', marker='s', markersize=10, label='GT'),
                         Line2D([0],[0], linestyle='none', marker='s', markersize=10, label='Pred')]
                fig.legend(handles=handles, loc='lower center', ncol=2, frameon=True, bbox_to_anchor=(0.5,-0.02))
                plt.tight_layout()
                fname=f"member{member_id}_val_sample_{sample_counter+1}_t{t}.png"
                plt.savefig(fname, dpi=140, bbox_inches='tight'); plt.close()
                images.append(imageio.imread(fname))
            imageio.mimsave(f"member{member_id}_val_sample_{sample_counter+1}.gif", images, fps=2)
        sample_counter += 1

    return member_preds  # list length = |val_set|; each [T,H,W]

# ---------------------------
# Ensemble aggregation, p10/p50/p90 maps, diagnostics & CSVs
# ---------------------------
def ensemble_evaluate(ensemble_preds, val_loader, forecast_steps, out_prefix="resunet_ens"):
    """
    ensemble_preds: list over members, each is list over val samples → [T,H,W]
    Saves:
      - final-step GT/p50/|GT-p50|/width(p90-p10) panels
      - final-step hit-maps GT∈[p10,p90]
      - final-step mean/std/width panels
      - error vs std / error vs width
      - optional all-time-step p10/p50/p90 panels (toggle)
      - quantile reliability CSV & per-sample metrics CSV
    """
    M = len(ensemble_preds)
    N = len(ensemble_preds[0])
    preds = np.stack([[ensemble_preds[m][i] for i in range(N)] for m in range(M)], axis=0)  # [M,N,T,H,W]

    SAVE_ALL_STEPS = False  # True → dump p10/p50/p90/width for all T (first 2 samples)
    os.makedirs("uncertainty_maps", exist_ok=True)
    val_items = list(val_loader)
    results, coverages_95 = [], []

    # PIT-style reliability rows
    pit_rows = []
    probs = {'p10':0.10, 'p50':0.50, 'p90':0.90}

    for i, (_, y, features) in enumerate(val_items):
        gt = y.numpy().squeeze()       # [T,H,W]
        stack = preds[:, i, ...]       # [M,T,H,W]

        mean_map = np.mean(stack, axis=0)
        std_map  = np.std(stack, axis=0)
        q05, q10, q50, q90, q95 = np.quantile(stack, [0.05, 0.10, 0.50, 0.90, 0.95], axis=0)
        band_10_90 = q90 - q10

        # Classic metrics (mean)
        flat_gt, flat_mean = gt.flatten(), mean_map.flatten()
        mae  = mean_absolute_error(flat_gt, flat_mean)
        rmse = np.sqrt(mean_squared_error(flat_gt, flat_mean))
        r2   = r2_score(flat_gt, flat_mean)

        # CRPS (approx)
        crps = float(np.mean(crps_ensemble(gt, stack)))

        # Coverage (95%) and (10–90)
        cover95 = float(np.mean((gt >= q05) & (gt <= q95)))
        cover_10_90 = float(np.mean((gt >= q10) & (gt <= q90)))
        coverages_95.append(cover95)

        # Interval width (PINAW absolute; normalize if needed)
        pinaw_10_90 = float(np.mean(band_10_90))
        # To normalize by global GT range, uncomment:
        # pinaw_10_90 /= (np.max(gt) - np.min(gt) + 1e-8)

        # Pinball losses (all T, all pixels)
        ql10_all = pinball_loss(gt, q10, tau=0.10)
        ql50_all = pinball_loss(gt, q50, tau=0.50)
        ql90_all = pinball_loss(gt, q90, tau=0.90)

        # Final-step diagnostics
        t_fin   = -1
        gt_fin  = gt[t_fin]
        p50_fin = q50[t_fin]
        err_p50 = np.abs(gt_fin - p50_fin)
        width_fin = band_10_90[t_fin]
        hit_fin = ((gt_fin >= q10[t_fin]) & (gt_fin <= q90[t_fin])).astype(float)

        # Panel: GT / p50 / |GT-p50| / width(p90-p10)
        plt.figure(figsize=(14,4))
        ax1 = plt.subplot(1,4,1); im1 = ax1.imshow(gt_fin, cmap='viridis'); ax1.set_title('GT (final)'); ax1.axis('off')
        c1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04); c1.set_label('CO₂ sat', rotation=270, labelpad=12)
        ax2 = plt.subplot(1,4,2); im2 = ax2.imshow(p50_fin, cmap='viridis'); ax2.set_title('p50 (final)'); ax2.axis('off')
        c2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04); c2.set_label('CO₂ sat', rotation=270, labelpad=12)
        ax3 = plt.subplot(1,4,3); im3 = ax3.imshow(err_p50, cmap='hot'); ax3.set_title('|GT−p50|'); ax3.axis('off')
        c3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04); c3.set_label('|Error|', rotation=270, labelpad=12)
        ax4 = plt.subplot(1,4,4); im4 = ax4.imshow(width_fin, cmap='magma'); ax4.set_title('width (p90−p10)'); ax4.axis('off')
        c4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04); c4.set_label('Width', rotation=270, labelpad=12)
        plt.tight_layout(); plt.savefig(f"uncertainty_maps/sample{i+1}_final_panel_gt_p50_err_width.png", dpi=180); plt.close()

        # Hit-map GT∈[p10,p90] (final step)
        plt.figure(); hm = plt.imshow(hit_fin, vmin=0, vmax=1)
        cb = plt.colorbar(hm); cb.set_label('Hit (1) / Miss (0)', rotation=270, labelpad=12)
        plt.title(f'Hit-map GT∈[p10,p90] – sample {i+1} (final)'); plt.axis('off')
        plt.savefig(f"uncertainty_maps/sample{i+1}_final_hitmap_p10_p90.png", dpi=180); plt.close()

        # Mean/Std/Width triple (final)
        plt.figure(figsize=(12,4))
        ax1 = plt.subplot(1,3,1); im1 = ax1.imshow(mean_map[t_fin], cmap='viridis'); ax1.set_title('Ensemble mean (final)'); ax1.axis('off')
        c1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04); c1.set_label('CO₂ sat', rotation=270, labelpad=12)
        ax2 = plt.subplot(1,3,2); im2 = ax2.imshow(std_map[t_fin], cmap='magma'); ax2.set_title('Predictive std'); ax2.axis('off')
        c2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04); c2.set_label('Std', rotation=270, labelpad=12)
        ax3 = plt.subplot(1,3,3); im3 = ax3.imshow(width_fin, cmap='magma'); ax3.set_title('Width (p90−p10)'); ax3.axis('off')
        c3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04); c3.set_label('Width', rotation=270, labelpad=12)
        plt.tight_layout(); plt.savefig(f"uncertainty_maps/sample{i+1}_final_mean_std_width.png", dpi=180); plt.close()

        # Error–uncertainty diagnostics (final)
        abs_err_fin = np.abs(gt_fin - mean_map[t_fin]).flatten()
        std_fin     = std_map[t_fin].flatten()
        width_fin_f = width_fin.flatten()
        plt.figure(figsize=(5,4))
        plt.scatter(std_fin, abs_err_fin, s=5, alpha=0.3)
        plt.xlabel('Predictive std (final)'); plt.ylabel('|Error|'); plt.title('Error vs Std')
        plt.tight_layout(); plt.savefig(f"uncertainty_maps/sample{i+1}_error_vs_std.png", dpi=170); plt.close()
        plt.figure(figsize=(5,4))
        plt.scatter(width_fin_f, abs_err_fin, s=5, alpha=0.3)
        plt.xlabel('Interval width p90−p10 (final)'); plt.ylabel('|Error|'); plt.title('Error vs Width')
        plt.tight_layout(); plt.savefig(f"uncertainty_maps/sample{i+1}_error_vs_width.png", dpi=170); plt.close()

        # Optional: full T panels for first 2 samples
        if SAVE_ALL_STEPS and i < 2:
            T = gt.shape[0]
            for t in range(T):
                fig, axs = plt.subplots(1,5, figsize=(18,3.8))
                ims = []
                ims.append(axs[0].imshow(gt[t],  cmap='viridis')); axs[0].set_title(f'GT t{t+1}');   axs[0].axis('off')
                ims.append(axs[1].imshow(q10[t], cmap='viridis')); axs[1].set_title('p10');          axs[1].axis('off')
                ims.append(axs[2].imshow(q50[t], cmap='viridis')); axs[2].set_title('p50');          axs[2].axis('off')
                ims.append(axs[3].imshow(q90[t], cmap='viridis')); axs[3].set_title('p90');          axs[3].axis('off')
                ims.append(axs[4].imshow(q90[t]-q10[t], cmap='magma')); axs[4].set_title('width p90−p10'); axs[4].axis('off')
                for k in range(4):
                    cb = plt.colorbar(ims[k], ax=axs[k], fraction=0.046, pad=0.04); cb.set_label('CO₂ sat', rotation=270, labelpad=12)
                cb = plt.colorbar(ims[4], ax=axs[4], fraction=0.046, pad=0.04); cb.set_label('Width', rotation=270, labelpad=12)
                plt.tight_layout()
                plt.savefig(f"uncertainty_maps/sample{i+1}_t{t+1}_p10_p50_p90_width.png", dpi=170); plt.close()

        # Record per-sample metrics (using p50 as point estimate)
        farr = features.numpy().flatten()
        poro = float(farr[0]) if farr.shape[0] > 0 else 0.0
        perm = float(farr[1]) if farr.shape[0] > 1 else 0.0
        results.append(dict(
            sample=i+1, porosity=poro, permeability=perm,
            MAE_p50 = mean_absolute_error(gt.flatten(), q50.flatten()),
            RMSE_p50= np.sqrt(mean_squared_error(gt.flatten(), q50.flatten())),
            R2_p50  = r2_score(gt.flatten(), q50.flatten()),
            MAE_mean= mae, RMSE_mean= rmse, R2_mean= r2,
            CRPS=float(crps), COVER95=cover95,
            PICP_10_90=cover_10_90, PINAW_10_90=pinaw_10_90,
            QL_tau10=ql10_all, QL_tau50=ql50_all, QL_tau90=ql90_all
        ))

    # Quantile reliability (PIT-like): how often GT ≤ qτ vs τ
    for qname, tau in probs.items():
        counts = []
        for i, (_, y, _) in enumerate(val_items):
            gt = y.numpy().squeeze()
            stack = preds[:, i, ...]
            qtau = np.quantile(stack, tau, axis=0)
            counts.append(np.mean(gt <= qtau))
        pit_rows.append({'quantile': qname, 'target_prob': tau, 'empirical_prob': float(np.mean(counts))})
    pd.DataFrame(pit_rows).to_csv(f"{out_prefix}_quantile_reliability.csv", index=False)

    # Save per-sample table
    df = pd.DataFrame(results)
    df.to_csv(f"{out_prefix}_val_metrics.csv", index=False)
    print("Average COVER95:", np.mean([r['COVER95'] for r in results]).round(4))
    print("Average PICP_10_90:", np.mean([r['PICP_10_90'] for r in results]).round(4))
    return df

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # ---- Config ----
    root_dirs = ['./het_1', './het_2', './het_3', './het_4', './het_5']
    target_size = 256
    forecast_steps = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # Ensemble settings
    ENSEMBLE_SIZE   = 5       # increase to 7–10 for smoother uncertainty
    EPOCHS          = 20
    BATCH_SIZE      = 2
    VAL_RATIO       = 0.2
    BOOTSTRAP       = True    # bootstrap train set per member
    BOOT_FRAC       = 1.0
    MEMBER_DROPOUT  = 0.10    # dropout inside residual blocks
    USE_MC_DROPOUT  = False   # set True for extra MC samples at validation
    MC_SAMPLES      = 8

    # ---- Data ----
    dataset = CO2DatasetNoRelPerm(root_dirs, target_size=target_size, forecast_steps=forecast_steps)
    if len(dataset) == 0:
        raise RuntimeError("No .hdf5 files found under provided root_dirs.")

    val_size = max(1, int(len(dataset) * VAL_RATIO))
    train_size = max(1, len(dataset) - val_size)
    base_train, base_val = random_split(dataset, [train_size, val_size])

    val_loader = DataLoader(base_val, batch_size=1, shuffle=False)

    # ---- Train ensemble members ----
    ensemble_preds = []
    os.makedirs("checkpoints", exist_ok=True)
    set_seed(1234)  # base seed for split reproducibility

    for m in range(ENSEMBLE_SIZE):
        seed = 1234 + m
        set_seed(seed)

        if BOOTSTRAP:
            train_subset = bootstrap_subset(base_train, frac=BOOT_FRAC, seed=seed)
        else:
            train_subset = base_train
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)

        print(f"\n===== Train ResUNet Ensemble Member {m+1}/{ENSEMBLE_SIZE} (seed={seed}) =====\n")
        model = train_member(
            member_id=m+1, train_loader=train_loader, device=device,
            forecast_steps=forecast_steps, epochs=EPOCHS,
            dropout=MEMBER_DROPOUT, lr=1e-3, save_ckpt=True
        )

        print(f"\n----- Validate Member {m+1} -----\n")
        member_preds = validate_member(
            model, val_loader, device, forecast_steps,
            member_id=m+1, use_mc=USE_MC_DROPOUT, mc_samples=MC_SAMPLES
        )
        ensemble_preds.append(member_preds)

    # ---- Aggregate & Uncertainty ----
    print("\n===== Ensemble Aggregation & Uncertainty (p10/p50/p90) =====\n")
    _ = ensemble_evaluate(ensemble_preds, val_loader, forecast_steps, out_prefix="resunet_ensemble")

    print("Done. See: checkpoints/, uncertainty_maps/, member*_val_sample_*.png/.gif, and CSVs.")
