# puredl_with_legends_fixed_resunet.py
# End-to-end training + visualization (with legends & labeled colorbars)
# Models: DeepCNN, UNetForecast, ResUNetForecast(fixed), FNO2d

import os, random, warnings, h5py, numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import imageio.v2 as imageio

# ---------------------------
# Dataset
# ---------------------------
class CO2DatasetNoRelPerm(Dataset):
    """
    Loads HDF5 samples and builds inputs [img, p, Ux, Uy, porosity_map, permeability_map].
    Target is the CO2 sequence computed from alpha_water and img.
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

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        h5file = self.file_list[idx]
        with h5py.File(h5file, 'r') as f:
            img = f['img'][:].astype(np.float32)          # [H,W]
            p   = f['p'][-1].astype(np.float32)           # [H,W] final step
            ux  = f['Ux'][-1].astype(np.float32)          # [H,W]
            uy  = f['Uy'][-1].astype(np.float32)          # [H,W]

            # min-max to [0,1] (per-sample)
            def norm(x):
                return (x - x.min()) / (x.max() - x.min() + 1e-8)

            img, p, ux, uy = norm(img), norm(p), norm(ux), norm(uy)
            input_stack = np.stack([img, p, ux, uy])  # (4, H, W)

            # porosity/permeability from CSV (optional)
            folder = os.path.dirname(h5file)
            csv_folder = [d for d in os.listdir(folder) if d.endswith('csv_files')]
            porosity, permeability = 0.0, 0.0
            if csv_folder:
                csv_path = os.path.join(folder, csv_folder[0], 'poroPerm.csv')
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    porosity = float(df['porosity'].mean()) if 'porosity' in df.columns else 0.0
                    permeability = float(df['permeability'].mean()) if 'permeability' in df.columns else 0.0

            # broadcast 2 scalars to 2 maps
            features = np.array([porosity, permeability], dtype=np.float32)
            features_map = np.repeat(features[:, None], self.target_size * self.target_size, axis=1
                            ).reshape(2, self.target_size, self.target_size)

            # resize input_stack to target_size
            input_stack_t = torch.tensor(input_stack)  # [4,H,W]
            input_stack_t = F.interpolate(input_stack_t.unsqueeze(0), size=(self.target_size, self.target_size),
                                          mode='bilinear', align_corners=False).squeeze(0)  # [4,T_H,T_W]
            input_stack = np.concatenate([input_stack_t.numpy(), features_map], axis=0)  # (6,H,W)

            # build target sequence
            alpha_water = f['alpha_water'][-self.forecast_steps:].astype(np.float32)  # [T,H,W]
            co2_seq = (1.0 - alpha_water) * img                                      # [T,H,W]; uses original img
            co2_seq = torch.tensor(co2_seq)                                          # [T,H,W]
            co2_seq = torch.stack([
                F.interpolate(co2_seq[i][None, None, :, :], size=(self.target_size, self.target_size),
                              mode='bilinear', align_corners=False).squeeze()
                for i in range(self.forecast_steps)
            ])  # [T,H,W]

        return torch.tensor(input_stack), co2_seq, features


# ---------------------------
# Models: DeepCNN, UNetForecast, ResUNetForecast(fixed)
# ---------------------------
class DeepCNN(nn.Module):
    """Simple deep CNN that maps C_in → T_out channels."""
    def __init__(self, in_ch, out_ch, features=64, layers=6, dropout=0.0):
        super().__init__()
        mods = []
        c = in_ch
        for _ in range(layers):
            mods += [nn.Conv2d(c, features, 3, padding=1), nn.ReLU(inplace=True)]
            if dropout > 0:
                mods += [nn.Dropout2d(dropout)]
            c = features
        self.body = nn.Sequential(*mods)
        self.head = nn.Conv2d(features, out_ch, 1)

    def forward(self, x):
        x = self.body(x)
        x = self.head(x)
        return torch.sigmoid(x)  # [B, T, H, W]


class UNetForecast(nn.Module):
    """Standard U-Net that outputs T_out channels."""
    def __init__(self, in_ch, out_ch, feats=(64,128,256,512), dropout=0.0):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        ch = in_ch
        # encoder
        for f in feats:
            self.downs.append(self._block(ch, f, dropout))
            ch = f
        # bottleneck
        self.bottleneck = self._block(feats[-1], feats[-1]*2, dropout)
        # decoder
        for f in reversed(feats):
            self.ups.append(nn.ConvTranspose2d(f*2, f, 2, stride=2))
            self.ups.append(self._block(f*2, f, dropout))
        self.out = nn.Conv2d(feats[0], out_ch, 1)

    def _block(self, c_in, c_out, dropout):
        layers = [
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)
        ]
        if dropout > 0: layers.append(nn.Dropout2d(dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            # pad if shapes mismatch due to pooling
            sk = skips[i//2]
            if x.shape[-2:] != sk.shape[-2:]:
                diffY = sk.size(2) - x.size(2)
                diffX = sk.size(3) - x.size(3)
                x = F.pad(x, [0, diffX, 0, diffY])
            x = torch.cat([x, sk], dim=1)
            x = self.ups[i+1](x)
        x = self.out(x)
        return torch.sigmoid(x)  # [B, T, H, W]


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
        y = self.conv1(x); y = self.bn1(y); y = self.relu(y)
        y = self.conv2(y); y = self.bn2(y); y = self.do(y)
        return self.relu(x + y)


class ResUNetForecast(nn.Module):
    """UNet-style encoder/decoder with residual blocks (fixed channel merge)."""
    def __init__(self, in_ch, out_ch, feats=(64,128,256,512), blocks_per_stage=2, dropout=0.0):
        super().__init__()
        self.enc_convs = nn.ModuleList()
        self.enc_res   = nn.ModuleList()
        self.pools     = nn.ModuleList()

        c = in_ch
        for f in feats:
            # 1 conv to set channels -> residual stack
            self.enc_convs.append(nn.Conv2d(c, f, 3, padding=1))
            self.enc_res.append(nn.Sequential(*[ResidualBlock(f, dropout) for _ in range(blocks_per_stage)]))
            self.pools.append(nn.MaxPool2d(2))
            c = f

        self.bottleneck = nn.Sequential(
            nn.Conv2d(feats[-1], feats[-1]*2, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(feats[-1]*2, dropout),
        )

        # Decoder (upsample, merge 2f->f, residual stack)
        self.up_convs    = nn.ModuleList()
        self.merge_convs = nn.ModuleList()   # 1x1 to reduce 2f -> f after concat
        self.dec_res     = nn.ModuleList()

        up_in = feats[-1]*2
        for f in reversed(feats):
            self.up_convs.append(nn.ConvTranspose2d(up_in, f, 2, stride=2))
            self.merge_convs.append(nn.Conv2d(f + f, f, kernel_size=1))
            self.dec_res.append(nn.Sequential(*[ResidualBlock(f, dropout) for _ in range(blocks_per_stage)]))
            up_in = f

        self.out = nn.Conv2d(feats[0], out_ch, 1)

    def forward(self, x):
        skips = []
        c = x
        for conv, res, pool in zip(self.enc_convs, self.enc_res, self.pools):
            c = conv(c)
            c = F.relu(c, inplace=True)
            c = res(c)
            skips.append(c)
            c = pool(c)

        c = self.bottleneck(c)
        skips = skips[::-1]

        for up, merge, res, sk in zip(self.up_convs, self.merge_convs, self.dec_res, skips):
            c = up(c)
            # pad if shapes mismatch due to pooling/odd sizes
            if c.shape[-2:] != sk.shape[-2:]:
                diffY = sk.size(2) - c.size(2)
                diffX = sk.size(3) - c.size(3)
                c = F.pad(c, [0, max(diffX, 0), 0, max(diffY, 0)])
                if diffY < 0 or diffX < 0:
                    sk = sk[..., :c.size(2), :c.size(3)]
            c = torch.cat([c, sk], dim=1)  # channels = 2f
            c = merge(c)                   # 2f -> f
            c = res(c)

        c = self.out(c)
        return torch.sigmoid(c)  # [B, T, H, W]


# ---------------------------
# FNO (2D)
# ---------------------------
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, 2))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights[..., 0] + 1j * weights[..., 1])

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, in_ch=6, out_ch=10):
        super().__init__()
        self.width = width
        self.fc0 = nn.Linear(in_ch, self.width)
        self.conv0 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_ch)

    def forward(self, x):
        # x: (B, H, W, in_ch)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)   # (B, C, H, W)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)   # (B, H, W, C)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)     # (B, H, W, T)


# ---------------------------
# Train/Evaluate + Visualization (with legends)
# ---------------------------
def train_and_evaluate(model, name, train_loader, val_loader, forecast_steps, device, out_csv_prefix, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y, _ in tqdm(train_loader, desc=f"{name} Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            if 'FNO' in name:
                x = x.permute(0, 2, 3, 1)       # (B,H,W,C)
            pred = model(x)
            if 'FNO' in name:
                pred = pred.permute(0, 3, 1, 2) # (B,T,H,W)

            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"{name} Epoch {epoch+1} avg train loss: {running_loss/len(train_loader):.4f}")

    # --- Evaluation ---
    results = []
    model.eval()
    for i, (x, y, features) in enumerate(val_loader):
        x, y = x.to(device), y.to(device)
        if 'FNO' in name:
            x = x.permute(0, 2, 3, 1)
        with torch.no_grad():
            pred = model(x)
        if 'FNO' in name:
            pred = pred.permute(0, 3, 1, 2)

        gt_seq = y.cpu().squeeze().numpy()      # [T,H,W]
        pred_seq = pred.cpu().squeeze().numpy() # [T,H,W]

        # PNG grids (with legends & labeled colorbars)
        abs_err_seq = np.abs(gt_seq - pred_seq)
        fig, axs = plt.subplots(forecast_steps, 3, figsize=(12, 3*forecast_steps))
        for t in range(forecast_steps):
            im0 = axs[t,0].imshow(gt_seq[t], cmap='viridis')
            axs[t,0].set_title(f'GT t{t+1}')
            axs[t,0].axis('off')
            c0 = plt.colorbar(im0, ax=axs[t,0], fraction=0.046, pad=0.04); c0.set_label('CO₂ saturation', rotation=270, labelpad=12)

            im1 = axs[t,1].imshow(pred_seq[t], cmap='viridis')
            axs[t,1].set_title(f'Pred t{t+1}')
            axs[t,1].axis('off')
            c1 = plt.colorbar(im1, ax=axs[t,1], fraction=0.046, pad=0.04); c1.set_label('CO₂ saturation', rotation=270, labelpad=12)

            im2 = axs[t,2].imshow(abs_err_seq[t], cmap='hot')
            axs[t,2].set_title('Abs Error')
            axs[t,2].axis('off')
            c2 = plt.colorbar(im2, ax=axs[t,2], fraction=0.046, pad=0.04); c2.set_label('|Error|', rotation=270, labelpad=12)

        legend_elems = [
            Patch(facecolor='none', edgecolor='black', label='GT (viridis)'),
            Patch(facecolor='none', edgecolor='black', label='Prediction (viridis)'),
            Patch(facecolor='none', edgecolor='black', label='Abs Error (hot)'),
        ]
        fig.legend(handles=legend_elems, loc='upper center', ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.02))
        plt.suptitle(f'{name} validation sample {i+1}')
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(f'{name}_val_sample_{i+1}.png', dpi=200)
        plt.close()

        # GIF frames (with a small legend)
        images = []
        for t in range(forecast_steps):
            fig = plt.figure(figsize=(6,3))
            ax1 = plt.subplot(1,2,1); ax1.imshow(gt_seq[t], cmap='viridis');  ax1.set_title('GT');  ax1.axis('off')
            ax2 = plt.subplot(1,2,2); ax2.imshow(pred_seq[t], cmap='viridis'); ax2.set_title('Pred'); ax2.axis('off')
            handles = [
                Line2D([0],[0], linestyle='none', marker='s', markersize=10, label='GT (viridis)'),
                Line2D([0],[0], linestyle='none', marker='s', markersize=10, label='Pred (viridis)'),
            ]
            fig.legend(handles=handles, loc='lower center', ncol=2, frameon=True, bbox_to_anchor=(0.5, -0.02))
            plt.tight_layout()
            fname = f"{name}_val_sample_{i+1}_t{t}.png"
            plt.savefig(fname, dpi=140, bbox_inches='tight')
            plt.close()
            images.append(imageio.imread(fname))
        imageio.mimsave(f"{name}_val_sample_{i+1}_animation.gif", images, fps=2)

        # Metrics
        flat_gt = gt_seq.flatten()
        flat_pred = pred_seq.flatten()
        mae  = mean_absolute_error(flat_gt, flat_pred)
        rmse = np.sqrt(mean_squared_error(flat_gt, flat_pred))
        r2   = r2_score(flat_gt, flat_pred)

        feature_arr = features.numpy().flatten()
        poro = float(feature_arr[0]) if feature_arr.shape[0] > 0 else 0.0
        perm = float(feature_arr[1]) if feature_arr.shape[0] > 1 else 0.0
        results.append({'model': name, 'sample': i+1, 'porosity': poro, 'permeability': perm,
                        'MAE': mae, 'RMSE': rmse, 'R2': r2})

        # Final-step error heatmap (labeled colorbar)
        plt.figure()
        hm = plt.imshow(abs_err_seq[-1], cmap='hot')
        cb = plt.colorbar(hm); cb.set_label('|Error|', rotation=270, labelpad=12)
        plt.title(f'{name} Abs Error Heatmap (final step)')
        plt.savefig(f"{name}_abs_err_heatmap_{i+1}.png", dpi=160)
        plt.close()

        # Histogram
        plt.figure()
        plt.hist(abs_err_seq[-1].flatten(), bins=50)
        plt.title(f'{name} Abs Error Histogram (final step)')
        plt.xlabel('Error'); plt.ylabel('Frequency')
        plt.savefig(f"{name}_abs_err_hist_{i+1}.png")
        plt.close()

        if i >= 2:  # limit plotting samples per model for speed
            break

    pd.DataFrame(results).to_csv(f"{out_csv_prefix}_{name}_results.csv", index=False)
    return results


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # Data
    root_dirs = ['./het_1', './het_2', './het_3', './het_4', './het_5']
    target_size = 256
    forecast_steps = 10
    in_channels = 6
    epochs = 200
    val_ratio = 0.2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    dataset = CO2DatasetNoRelPerm(root_dirs, target_size=target_size, forecast_steps=forecast_steps)
    if len(dataset) == 0:
        raise RuntimeError("No .hdf5 files found under provided root_dirs.")

    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = max(1, len(dataset) - val_size)
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=1)

    # Models
    models = {
        'UNet':     UNetForecast(in_channels, forecast_steps).to(device),
        'ResUNet':  ResUNetForecast(in_channels, forecast_steps).to(device),
        'DeepCNN':  DeepCNN(in_channels, forecast_steps).to(device),
        'FNO':      FNO2d(modes1=16, modes2=16, width=32, in_ch=in_channels, out_ch=forecast_steps).to(device)
    }

    # Train + Evaluate
    all_results = []
    for model_name, model in models.items():
        print(f"\n===== Training {model_name} =====\n")
        model_results = train_and_evaluate(model, model_name, train_loader, val_loader,
                                           forecast_steps, device, out_csv_prefix='dl_results',
                                           epochs=epochs)
        all_results.extend(model_results)

    # Aggregate results and plots
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("dl_all_results.csv", index=False)

    # Bar charts
    for metric in ['MAE', 'RMSE', 'R2']:
        plt.figure(figsize=(7,4))
        results_df.groupby('model')[metric].mean().plot(kind='bar')
        plt.title(f'Model Comparison: {metric}')
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(f'model_comparison_{metric.lower()}.png', dpi=160)
        plt.close()

    # Stratified scatterplots with legends
    plt.figure(figsize=(7,5))
    models_order = results_df['model'].unique().tolist()
    colors = plt.cm.tab10(np.arange(len(models_order)))
    for m, c in zip(models_order, colors):
        sel = results_df['model'] == m
        plt.scatter(results_df.loc[sel, 'porosity'],
                    results_df.loc[sel, 'MAE'],
                    label=m, color=c, alpha=0.85)
    plt.xlabel('Porosity'); plt.ylabel('MAE'); plt.title('MAE vs Porosity (all models)')
    plt.legend(title='Model', ncol=2, frameon=True)
    plt.tight_layout(); plt.savefig('mae_vs_porosity.png', dpi=160); plt.close()

    plt.figure(figsize=(7,5))
    for m, c in zip(models_order, colors):
        sel = results_df['model'] == m
        plt.scatter(results_df.loc[sel, 'permeability'],
                    results_df.loc[sel, 'MAE'],
                    label=m, color=c, alpha=0.85)
    plt.xlabel('Permeability'); plt.ylabel('MAE'); plt.title('MAE vs Permeability (all models)')
    plt.legend(title='Model', ncol=2, frameon=True)
    plt.tight_layout(); plt.savefig('mae_vs_permeability.png', dpi=160); plt.close()

    print("Done. Outputs: *_val_sample_*.png, *_animation.gif, *_heatmap.png, *_hist.png, model_comparison_*.png, mae_vs_*.png, and CSVs.")
