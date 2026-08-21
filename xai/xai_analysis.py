# ═══════════════════════════════════════════════════════════════════════════════
# XAI ANALYSIS CELL — STANDALONE (paste as a NEW cell, no prior cells needed)
# Generates: GradCAM, GradCAM++, t-SNE, Attention Maps, YOLO Visualizations
# Outputs saved to: /kaggle/working/xai_outputs/
# ═══════════════════════════════════════════════════════════════════════════════

import os, random, cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from PIL import Image
from torchvision import transforms
from collections import defaultdict

os.system("pip install grad-cam scikit-learn timm -q")

import timm
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight',
                     'savefig.facecolor': 'white'})

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 204
CKPT_PATH   = "/kaggle/working/best_lightpestformer.pth"
XAI_DIR     = Path("/kaggle/working/xai_outputs")
XAI_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("XAI ANALYSIS — LightPestFormer + YOLO11n")
print(f"Device: {DEVICE}")
print("=" * 60)

# ─── STEP 1: LightPestFormer Architecture ────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class MSSA(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ) for c in in_channels_list
        ])
        self.attn = nn.ModuleList([ChannelAttention(out_channels) for _ in in_channels_list])
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels_list), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        target_size = features[0].shape[2:]
        outs = []
        for feat, proj, attn in zip(features, self.projs, self.attn):
            x = proj(feat)
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            x = x * attn(x)
            outs.append(x)
        return self.fuse(torch.cat(outs, dim=1))


class LightPestFormer(nn.Module):
    def __init__(self, num_classes=204):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b2', pretrained=False, features_only=True, out_indices=(2, 3, 4))
        # Probe actual output channels — feature_info can report wrong values
        with torch.no_grad():
            _dummy = torch.zeros(1, 3, 224, 224)
            _feats = self.backbone(_dummy)
            in_chs = [f.shape[1] for f in _feats]
        print(f"    backbone output channels: {in_chs}")
        self.mssa = MSSA(in_chs, out_channels=256)
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.head  = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        feats = self.backbone(x)
        x = self.mssa(feats)
        x = self.pool(x).flatten(1)
        return self.head(x)

# ─── STEP 2: Build & load model ───────────────────────────────────────────────
print("\n[Setup] Building LightPestFormer...")
model = LightPestFormer(num_classes=NUM_CLASSES).to(DEVICE)

# Search for checkpoint in multiple locations
_ckpt_candidates = [
    CKPT_PATH,
    "/kaggle/working/best_lightpestformer.pth",
    "/kaggle/working/lightpestformer_best.pth",
    "/kaggle/working/best_model.pth",
    "/kaggle/working/checkpoint.pth",
]
# Also scan /kaggle/working for any .pth file
for _f in Path("/kaggle/working").rglob("*.pth"):
    if str(_f) not in _ckpt_candidates:
        _ckpt_candidates.append(str(_f))

_found_ckpt = next((p for p in _ckpt_candidates if os.path.exists(p)), None)
if _found_ckpt:
    ckpt = torch.load(_found_ckpt, map_location=DEVICE)
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  ✓ Checkpoint loaded: {_found_ckpt}")
    print(f"    (missing={len(missing)}, unexpected={len(unexpected)})")
else:
    print(f"  ⚠ No .pth checkpoint found in /kaggle/working — using random weights")
    print(f"    XAI visuals will still run; patterns show architecture behaviour")

model.eval()
print(f"  ✓ Model ready [{sum(p.numel() for p in model.parameters())/1e6:.2f}M params]")

# ─── STEP 3: Find INSECT204 images (robust recursive search) ─────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def find_image_files(search_roots, max_images=500):
    """
    Walk search_roots in order. Collect (img_path, class_name) pairs.
    class_name = immediate parent folder of the image file.
    Stops at max_images. Skips dirs that only contain .txt files (YOLO labels).
    """
    collected = []
    seen_roots = set()
    for root in search_roots:
        if not os.path.isdir(root) or root in seen_roots:
            continue
        seen_roots.add(root)
        for dirpath, subdirs, filenames in os.walk(root):
            imgs = [f for f in filenames if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not imgs:
                continue
            cls_name = os.path.basename(dirpath)
            # Skip YOLO-format label dirs (parent is 'labels')
            if os.path.basename(os.path.dirname(dirpath)) == 'labels':
                continue
            for f in imgs:
                collected.append((os.path.join(dirpath, f), cls_name))
            if len(collected) >= max_images:
                break
        if len(collected) >= max_images:
            break
    return collected

# Priority order: insect204-preprocessed first (the Kaggle dataset name)
SEARCH_ROOTS = [
    "/kaggle/input/insect204-preprocessed/test",
    "/kaggle/input/insect204-preprocessed/val",
    "/kaggle/input/insect204-preprocessed/train",
    "/kaggle/input/insect204-preprocessed",
    "/kaggle/input/insect204/test",
    "/kaggle/input/insect204/val",
    "/kaggle/input/insect-204/test",
    "/kaggle/input/insect-204/val",
    "/kaggle/working/data/test",
    "/kaggle/working/data/val",
]

# Auto-discover any insect-related dataset not in the list above
for entry in os.listdir("/kaggle/input"):
    if 'insect' in entry.lower() or 'pest' in entry.lower():
        for split in ['test', 'val', 'train', '']:
            p = os.path.join("/kaggle/input", entry, split)
            if p not in SEARCH_ROOTS:
                SEARCH_ROOTS.append(p)

print(f"\n[Setup] Searching for images...")
all_image_files = find_image_files(SEARCH_ROOTS, max_images=500)
print(f"  Found {len(all_image_files)} images")

if not all_image_files:
    # Hard fallback: scan ALL of /kaggle/input for any jpg/png
    print("  ⚠ No insect images found — scanning all of /kaggle/input (slow but thorough)...")
    all_image_files = find_image_files(["/kaggle/input"], max_images=300)
    print(f"  Found {len(all_image_files)} images total")

def make_samples(image_files, n=60):
    """Convert raw file list → list of (tensor, np_img, class_name, label_idx)."""
    # Group by class, sample evenly
    by_class = defaultdict(list)
    for path, cls in image_files:
        by_class[cls].append(path)
    random.seed(42)
    classes = sorted(by_class.keys())
    result = []
    n_per = max(1, n // max(len(classes), 1))
    for lbl_idx, cls in enumerate(classes):
        imgs = by_class[cls]
        random.shuffle(imgs)
        for p in imgs[:n_per]:
            try:
                pil    = Image.open(p).convert("RGB").resize((224, 224))
                np_img = np.array(pil).astype(np.float32) / 255.0
                tensor = TRANSFORM(pil).unsqueeze(0).to(DEVICE)
                result.append((tensor, np_img, cls, lbl_idx))
            except Exception:
                pass
    return result

samples = make_samples(all_image_files, n=60)
print(f"  ✓ {len(samples)} sample tensors ready from {len(set(c for _,_,c,_ in samples))} classes")

if not samples:
    print("\n  ✗ FATAL: No usable images found in /kaggle/input")
    print("  Please verify the INSECT204 dataset is attached to this notebook.")
    raise RuntimeError("No images found — attach the insect204-preprocessed dataset")

# ─── STEP 4: GradCAM + GradCAM++ ─────────────────────────────────────────────
print("\n[1/5] Generating GradCAM heatmaps...")

class ModelWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        out = self.m(x)
        return out[0] if isinstance(out, (tuple, list)) else out

wrapped = ModelWrapper(model)
wrapped.eval()

target_layer = None
_last_conv   = "unknown"
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d) and 'backbone' in name:
        target_layer = module
        _last_conv   = name

print(f"  → GradCAM target layer: {_last_conv}")

viz_samples = samples[:8]
cam_methods = {
    'GradCAM':   GradCAM(model=wrapped, target_layers=[target_layer]),
    'GradCAM++': GradCAMPlusPlus(model=wrapped, target_layers=[target_layer]),
}

for method_name, cam in cam_methods.items():
    ncols = 4
    fig, axes = plt.subplots(2, ncols * 2, figsize=(ncols * 6, 7))
    axes = axes.flatten()
    fig.suptitle(f'{method_name} — LightPestFormer on INSECT204\n(Original | Heatmap)',
                 fontsize=13, fontweight='bold')

    for i, (tensor, np_img, cls_name, lbl) in enumerate(viz_samples):
        with torch.no_grad():
            logits   = wrapped(tensor)
            pred_cls = logits.argmax(dim=1).item()

        targets       = [ClassifierOutputTarget(pred_cls)]
        grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
        cam_image     = show_cam_on_image(np_img, grayscale_cam, use_rgb=True)
        conf          = torch.softmax(logits, dim=1)[0, pred_cls].item()

        axes[i*2].imshow(np_img);    axes[i*2].set_title(f'GT: {cls_name[:15]}', fontsize=8);      axes[i*2].axis('off')
        axes[i*2+1].imshow(cam_image); axes[i*2+1].set_title(f'{method_name} conf={conf:.2f}', fontsize=8); axes[i*2+1].axis('off')

    for ax in axes[len(viz_samples)*2:]:
        ax.axis('off')

    fig.tight_layout()
    fname = f"XAI_{'gradcam' if 'Plus' not in method_name else 'gradcampp'}.png"
    fig.savefig(XAI_DIR / fname);  plt.close()
    print(f"  ✓ {fname}")

# ─── STEP 5: t-SNE Feature Visualization ─────────────────────────────────────
print("\n[2/5] Generating t-SNE feature embeddings...")

tsne_samples = make_samples(all_image_files, n=200)
if not tsne_samples:
    tsne_samples = samples
print(f"  → t-SNE pool: {len(tsne_samples)} images")

features_list = []
labels_list   = []

def hook_fn(module, input, output):
    feat = input[0] if isinstance(input, tuple) else input
    features_list.append(feat.detach().cpu().numpy())

hook = model.head[-1].register_forward_hook(hook_fn)
model.eval()
with torch.no_grad():
    for tensor, _, cls_name, _ in tsne_samples:
        try:
            model(tensor)
            labels_list.append(cls_name)
        except Exception:
            pass
hook.remove()

if not features_list:
    print("  ⚠ Hook captured nothing — skipping t-SNE")
else:
    features_arr = np.vstack(features_list)
    print(f"  → Feature shape: {features_arr.shape}")

    n_pts  = len(features_arr)
    perp   = min(30, max(5, n_pts // 4))
    tsne   = TSNE(n_components=2, perplexity=perp, n_iter=1000,
                  random_state=42, learning_rate='auto', init='pca')
    coords = tsne.fit_transform(features_arr)

    le         = LabelEncoder()
    label_ints = le.fit_transform(labels_list)
    n_cls_t    = len(le.classes_)
    cmap       = plt.cm.get_cmap('tab20', n_cls_t)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(coords[:, 0], coords[:, 1], c=label_ints, cmap=cmap,
               s=60, alpha=0.8, edgecolors='white', linewidths=0.4)
    ax.set_title(f't-SNE Feature Embedding — LightPestFormer on INSECT204\n'
                 f'({n_pts} images, {n_cls_t} classes)', fontsize=13, fontweight='bold')
    ax.set_xlabel('t-SNE Dim 1');  ax.set_ylabel('t-SNE Dim 2')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False);  ax.spines['right'].set_visible(False)

    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], marker='o', color='w',
                      markerfacecolor=cmap(i/n_cls_t), markersize=8, label=c[:20])
               for i, c in enumerate(le.classes_[:20])]
    ax.legend(handles=handles, loc='upper right', fontsize=7, ncol=2,
              framealpha=0.8, title='Classes (top 20)')
    fig.tight_layout()
    fig.savefig(XAI_DIR / "XAI_tsne_features.png");  plt.close()
    print("  ✓ XAI_tsne_features.png")

# ─── STEP 6: MSSA Attention Map Visualization ────────────────────────────────
print("\n[3/5] Generating MSSA attention maps...")

attn_outputs = {}

def make_attn_hook(name):
    def h(module, inp, output):
        if isinstance(output, torch.Tensor):
            attn_outputs[name] = output.detach().cpu()
    return h

attn_hooks = []
for name, module in model.named_modules():
    if any(k in name.lower() for k in ['attn', 'mssa', 'attention', 'channel']):
        attn_hooks.append(module.register_forward_hook(make_attn_hook(name)))

# Guard: need at least one sample
if samples:
    sample_tensor, sample_np, sample_cls, _ = samples[0]
    with torch.no_grad():
        model(sample_tensor)
    for h in attn_hooks:
        h.remove()

    if attn_outputs:
        n_maps = min(6, len(attn_outputs))
        fig, axes = plt.subplots(2, n_maps, figsize=(n_maps * 3.5, 7))
        if n_maps == 1:
            axes = np.array(axes).reshape(2, 1)
        fig.suptitle(f'MSSA Attention Maps — {sample_cls}\n'
                     f'({len(attn_outputs)} layers)', fontsize=12, fontweight='bold')
        for col, (lname, attn_t) in enumerate(list(attn_outputs.items())[:n_maps]):
            axes[0, col].imshow(sample_np)
            axes[0, col].set_title(f'Input\n{lname.split(".")[-1]}', fontsize=7)
            axes[0, col].axis('off')
            a = attn_t[0]
            if a.dim() == 3: a = a.mean(0)
            a = a.numpy()
            a = (a - a.min()) / (a.max() - a.min() + 1e-8)
            overlay = (0.6 * sample_np + 0.4 * cm.jet(cv2.resize(a, (224,224)))[:,:,:3]).clip(0,1)
            axes[1, col].imshow(overlay)
            axes[1, col].set_title(f'Attn\n{list(attn_t.shape[1:])}', fontsize=7)
            axes[1, col].axis('off')
        fig.tight_layout()
        fig.savefig(XAI_DIR / "XAI_attention_maps.png");  plt.close()
        print(f"  ✓ XAI_attention_maps.png ({len(attn_outputs)} layers)")
    else:
        print("  ⚠ No attention layers captured")
else:
    for h in attn_hooks: h.remove()
    print("  ⚠ No samples available for attention maps")

# ─── STEP 7: YOLO XAI Visualizations ─────────────────────────────────────────
print("\n[4/5] Generating YOLO detection visualizations...")

YOLO_PT = None
for p in [
    "/kaggle/working/yolo11n_agropest12/weights/best.pt",
    "/kaggle/working/runs/detect/train/weights/best.pt",
    "/kaggle/working/best.pt",
]:
    if os.path.exists(p):
        YOLO_PT = p
        break

if YOLO_PT:
    from ultralytics import YOLO
    yolo_model = YOLO(YOLO_PT)

    AGROPEST_TEST = None
    for p in [
        "/kaggle/input/agropest12/test/images",
        "/kaggle/input/agropest-12/test/images",
        "/kaggle/input/datasets/mageshkumarnagappan/agropest12/AgroPest-12/test/images",
        "/kaggle/input/datasets/mageshkumarnagappan/agropest12/AgroPest-12/valid/images",
    ]:
        if os.path.isdir(p):
            AGROPEST_TEST = p
            break

    # Auto-discover agropest dataset
    if not AGROPEST_TEST:
        for entry in os.listdir("/kaggle/input"):
            if 'agropest' in entry.lower() or 'agro' in entry.lower():
                for sub in ['test/images', 'valid/images', 'val/images']:
                    p = f"/kaggle/input/{entry}/{sub}"
                    if os.path.isdir(p):
                        AGROPEST_TEST = p
                        break
                if AGROPEST_TEST:
                    break

    if AGROPEST_TEST:
        test_imgs = list(Path(AGROPEST_TEST).glob("*.jpg"))
        random.seed(42);  random.shuffle(test_imgs)
        test_imgs = test_imgs[:12]

        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle('YOLO11n Inference — AgroPest-12\n(Bounding Boxes + Labels + Confidence)',
                     fontsize=13, fontweight='bold')
        for ax, img_path in zip(axes.flatten(), test_imgs):
            results = yolo_model.predict(str(img_path), conf=0.25, verbose=False)[0]
            ax.imshow(cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB))
            ax.set_title(f'{img_path.stem[:18]}\n{len(results.boxes or [])} det', fontsize=7)
            ax.axis('off')
        for ax in axes.flatten()[len(test_imgs):]:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(XAI_DIR / "XAI_yolo_detections.png");  plt.close()
        print("  ✓ XAI_yolo_detections.png")
    else:
        print("  ⚠ AgroPest images not found — skipping YOLO viz")
else:
    print(f"  ⚠ YOLO best.pt not found at expected paths — skipping")

# ─── STEP 8: Confidence Calibration ──────────────────────────────────────────
print("\n[5/5] Generating confidence calibration...")

all_confs, all_correct = [], []
model.eval()
with torch.no_grad():
    for tensor, _, cls_name, lbl in samples:
        probs = torch.softmax(wrapped(tensor), dim=1)[0]
        all_confs.append(probs.max().item())
        all_correct.append(probs.argmax().item() == lbl)

correct_confs   = [c for c, ok in zip(all_confs, all_correct) if ok]
incorrect_confs = [c for c, ok in zip(all_confs, all_correct) if not ok]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('LightPestFormer — Prediction Confidence Analysis', fontsize=13, fontweight='bold')
bins = np.linspace(0, 1, 21)
ax1.hist(correct_confs,   bins=bins, color='#16A34A', alpha=0.75, label=f'Correct ({len(correct_confs)})')
ax1.hist(incorrect_confs, bins=bins, color='#DC2626', alpha=0.75, label=f'Incorrect ({len(incorrect_confs)})')
ax1.set_xlabel('Confidence');  ax1.set_ylabel('Count');  ax1.set_title('Confidence Distribution')
ax1.legend();  ax1.grid(True, alpha=0.3)

bin_accs, bin_mids = [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    in_bin = [ok for c, ok in zip(all_confs, all_correct) if lo <= c < hi]
    if in_bin:
        bin_accs.append(np.mean(in_bin));  bin_mids.append((lo+hi)/2)
ax2.plot([0,1],[0,1],'k--',lw=1.5,label='Perfect')
ax2.plot(bin_mids, bin_accs, 'o-', color='#2563EB', lw=2, ms=6, label='LightPestFormer')
ax2.fill_between(bin_mids, bin_accs, bin_mids[:len(bin_accs)], alpha=0.15, color='#2563EB')
ax2.set_xlabel('Mean Confidence');  ax2.set_ylabel('Accuracy')
ax2.set_title('Reliability Diagram');  ax2.legend()
ax2.set_xlim(0,1);  ax2.set_ylim(0,1);  ax2.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(XAI_DIR / "XAI_confidence_calibration.png");  plt.close()
print("  ✓ XAI_confidence_calibration.png")

# ─── SUMMARY ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ XAI ANALYSIS COMPLETE")
saved = sorted(XAI_DIR.glob("*.png"))
for f in saved:
    print(f"   {f.name:<45} {f.stat().st_size//1024:>5} KB")
print(f"\nTotal: {len(saved)} files")
print("👉 Kaggle Output panel → xai_outputs/ → Download all")
