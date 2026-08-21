"""
LightPestFormer – Simple Clean Architecture Diagram
Single top-to-bottom flow with clear arrows
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(14, 18))
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FFFFFF')
ax.set_xlim(0, 14)
ax.set_ylim(0, 18)
ax.axis('off')

ax.set_title('LightPestFormer: Architecture Overview', fontsize=18, fontweight='bold',
             color='#1A237E', pad=16)

# ─── helpers ────────────────────────────────────────────────────────────────
def box(cx, cy, w, h, title, subtitle='', color='#1565C0', title_size=12, sub_size=10, radius=0.3):
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle=f'round,pad=0.08,rounding_size={radius}',
                           facecolor=color, edgecolor='white', linewidth=2, zorder=3)
    ax.add_patch(rect)
    ty = cy + (h * 0.14 if subtitle else 0)
    ax.text(cx, ty, title, ha='center', va='center', color='white',
            fontsize=title_size, fontweight='bold', zorder=4)
    if subtitle:
        ax.text(cx, cy - h * 0.20, subtitle, ha='center', va='center',
                color='#E3F2FD', fontsize=sub_size, zorder=4)

def arrow(x1, y1, x2, y2, color='#37474F', lw=2.2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle='arc3,rad=0.0'), zorder=2)

def label(x, y, txt, color='#455A64', fs=9.5, bold=False):
    ax.text(x, y, txt, ha='center', va='center', fontsize=fs,
            color=color, fontweight='bold' if bold else 'normal')

CX = 7      # centre x for main flow
W  = 6.5    # default box width

# ── 1. INPUT ────────────────────────────────────────────────────────────────
box(CX, 17.0, W, 0.90, 'Input Image',  '224 × 224 × 3 RGB', '#1565C0', 13, 10)

# ── 2. EFFICIENTNET-B2 BACKBONE ─────────────────────────────────────────────
# outer dashed region
bkg = FancyBboxPatch((1.5, 13.5), 11.0, 2.8,
                     boxstyle='round,pad=0.1,rounding_size=0.4',
                     facecolor='#E8F5E9', edgecolor='#2E7D32',
                     linewidth=2, linestyle='--', zorder=1)
ax.add_patch(bkg)
ax.text(CX, 16.55, 'EfficientNet-B2 Backbone  (ImageNet pre-trained)',
        ha='center', fontsize=11, fontweight='bold', color='#1B5E20')

# three stage boxes side-by-side
stage_xs  = [3.5, 7.0, 10.5]
stage_lbl = ['Stage C₂', 'Stage C₃', 'Stage C₄']
stage_sub = ['28×28 | 48 ch', '14×14 | 120 ch', '7×7 | 352 ch']
for sx, sl, ss in zip(stage_xs, stage_lbl, stage_sub):
    box(sx, 15.0, 2.8, 1.1, sl, ss, '#2E7D32', 11, 9.5)

# arrows between stages
for i in range(len(stage_xs) - 1):
    arrow(stage_xs[i] + 1.4, 15.0, stage_xs[i+1] - 1.4, 15.0, '#2E7D32', 1.8)

# input → backbone entry
arrow(CX, 16.55, CX, 16.35, '#37474F')   # short: title row → top of bkg
arrow(CX, 17.0 - 0.45, CX, 16.55, '#37474F')

# ── 3. MSSA MODULE ──────────────────────────────────────────────────────────
# Outer region
mssa_bkg = FancyBboxPatch((1.5, 10.0), 11.0, 3.0,
                           boxstyle='round,pad=0.1,rounding_size=0.4',
                           facecolor='#EDE7F6', edgecolor='#6A1B9A',
                           linewidth=2, linestyle='--', zorder=1)
ax.add_patch(mssa_bkg)
ax.text(CX, 13.25, 'MSSA — Multi-Scale Semantic Attention',
        ha='center', fontsize=11, fontweight='bold', color='#4A148C')

# three branches under MSSA
mssa_xs   = [3.5, 7.0, 10.5]
mssa_proj = ['Conv 1×1 → 256\nBN-ReLU + Pool', 'Conv 1×1 → 256\nBN-ReLU + Pool', 'Conv 1×1 → 256\nBN-ReLU + Pool']
for mx in mssa_xs:
    box(mx, 12.45, 2.8, 0.95, 'Projection', 'Conv 1×1→256 + Pool', '#6A1B9A', 10, 8.5)

# arrows from backbone stages → MSSA projections
for sx, mx in zip(stage_xs, mssa_xs):
    arrow(sx, 14.45, mx, 12.93, '#455A64', 1.6)

# SE attention boxes
for mx in mssa_xs:
    box(mx, 11.25, 2.8, 0.80, 'SE Channel Attention', 'GAP → FC → Sigmoid ⊗', '#AD1457', 10, 8.5)
    arrow(mx, 11.97, mx, 11.65, '#6A1B9A', 1.6)

# converge → concat
for mx in mssa_xs:
    ax.annotate('', xy=(CX, 10.35), xytext=(mx, 10.85),
                arrowprops=dict(arrowstyle='->', color='#455A64', lw=1.6,
                                connectionstyle='arc3,rad=0.0'), zorder=2)

box(CX, 10.1, 8.0, 0.55, 'Concat  [C₂ ⊕ C₃ ⊕ C₄]  →  768 ch  →  Conv 1×1  →  256 ch',
    '', '#E65100', 10.5, 9)

# ── 4. HEAD ─────────────────────────────────────────────────────────────────
arrow(CX, 9.83, CX, 9.35, '#37474F')
box(CX, 9.0, W, 0.75, 'Global Average Pool', '256-dim Feature Vector', '#00695C', 12, 10)

arrow(CX, 8.63, CX, 8.15, '#37474F')
box(CX, 7.8, W, 0.75, 'Dropout  (p = 0.30)', '', '#546E7A', 12, 10)

arrow(CX, 7.43, CX, 6.95, '#37474F')
box(CX, 6.6, W, 0.75, 'Fully-Connected  256 → 204', '204-class Logits', '#00695C', 12, 10)

# ── 5. HYBRID LOSS ──────────────────────────────────────────────────────────
arrow(CX, 6.23, CX, 5.65, '#37474F')
box(CX, 5.3, 9.5, 0.95, 'HybridLoss  =  α · L_LSCE  +  (1−α) · L_SCL',
    'α annealed 0.70 → 0.85  |  ε=0.10  |  τ=0.07', '#C62828', 12, 10)

# ── 6. OUTPUT ───────────────────────────────────────────────────────────────
arrow(CX, 4.83, CX, 4.35, '#37474F')
box(CX, 4.0, W, 0.80, 'Predicted Class', 'INSECT204 — 204 insect species', '#1565C0', 12, 10)

# ── 7. TWO-PHASE TRAINING (side box, right) ─────────────────────────────────
train_bkg = FancyBboxPatch((11.0, 6.8), 2.8, 4.5,
                            boxstyle='round,pad=0.1,rounding_size=0.3',
                            facecolor='#E8EAF6', edgecolor='#3949AB',
                            linewidth=1.8, zorder=1)
ax.add_patch(train_bkg)
ax.text(12.4, 11.05, 'Two-Phase\nTraining', ha='center', fontsize=10,
        fontweight='bold', color='#1A237E')

# Phase 1
ph1 = FancyBboxPatch((11.1, 9.15), 2.6, 1.6,
                      boxstyle='round,pad=0.08', facecolor='#FFF3E0',
                      edgecolor='#E65100', lw=1.5, zorder=2)
ax.add_patch(ph1)
ax.text(12.4, 10.6, 'Phase 1', ha='center', fontsize=9.5, fontweight='bold', color='#BF360C')
ax.text(12.4, 10.25, '5 epochs', ha='center', fontsize=8.5, color='#37474F')
ax.text(12.4, 9.95, 'Backbone frozen', ha='center', fontsize=8.5, color='#37474F')
ax.text(12.4, 9.65, 'lr = 1e-3', ha='center', fontsize=8.5, color='#37474F')
ax.text(12.4, 9.35, 'α = 0.70', ha='center', fontsize=8.5, color='#37474F')

# arrow Phase1 → Phase2
ax.annotate('', xy=(12.4, 9.1), xytext=(12.4, 9.15),
            arrowprops=dict(arrowstyle='->', color='#3949AB', lw=1.6), zorder=2)

# Phase 2
ph2 = FancyBboxPatch((11.1, 7.15), 2.6, 1.85,
                      boxstyle='round,pad=0.08', facecolor='#E8F5E9',
                      edgecolor='#2E7D32', lw=1.5, zorder=2)
ax.add_patch(ph2)
ax.text(12.4, 8.75, 'Phase 2', ha='center', fontsize=9.5, fontweight='bold', color='#1B5E20')
ax.text(12.4, 8.42, '40 epochs', ha='center', fontsize=8.5, color='#37474F')
ax.text(12.4, 8.12, 'All params unfrozen', ha='center', fontsize=8.5, color='#37474F')
ax.text(12.4, 7.82, 'BB lr=1e-4 | lr=5e-4', ha='center', fontsize=8.5, color='#37474F')
ax.text(12.4, 7.52, 'Cosine LR + Early stop', ha='center', fontsize=8.5, color='#37474F')
ax.text(12.4, 7.22, 'α → 0.85', ha='center', fontsize=8.5, color='#37474F')

arrow(12.4, 9.15, 12.4, 9.0, '#3949AB', 1.5)

# ── 8. YOLO11n side branch (left) ───────────────────────────────────────────
yolo_bkg = FancyBboxPatch((0.2, 8.2), 2.8, 3.0,
                           boxstyle='round,pad=0.1,rounding_size=0.3',
                           facecolor='#E3F2FD', edgecolor='#0277BD',
                           linewidth=1.8, zorder=1)
ax.add_patch(yolo_bkg)
ax.text(1.6, 11.0, 'YOLO11n', ha='center', fontsize=11, fontweight='bold', color='#01579B')
ax.text(1.6, 10.55, 'Detection Branch', ha='center', fontsize=9, color='#0277BD', style='italic')
ax.text(1.6, 10.1, 'AgroPest-12', ha='center', fontsize=9, color='#37474F')
ax.text(1.6, 9.75, '12 insect classes', ha='center', fontsize=9, color='#37474F')
ax.text(1.6, 9.4, 'mAP@50 = 79.4%', ha='center', fontsize=9, fontweight='bold', color='#01579B')
ax.text(1.6, 9.05, '85 epochs trained', ha='center', fontsize=8.5, color='#37474F')
ax.text(1.6, 8.72, 'Best epoch = 77', ha='center', fontsize=8.5, color='#37474F')
ax.text(1.6, 8.4, 'EarlyStopping p=30', ha='center', fontsize=8.5, color='#37474F')

# dashed connector from backbone to YOLO
ax.annotate('', xy=(3.0, 10.0), xytext=(1.6, 9.5),
            arrowprops=dict(arrowstyle='<->', color='#0277BD', lw=1.5,
                            linestyle='dashed'), zorder=2)
ax.text(2.2, 9.1, 'Parallel\nbranch', ha='center', fontsize=7.5,
        color='#0277BD', style='italic')

# ── 9. STATS box ────────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((0.3, 0.4), 13.4, 3.2,
                             boxstyle='round,pad=0.15,rounding_size=0.35',
                             facecolor='#E3F2FD', edgecolor='#1565C0',
                             linewidth=2, zorder=1))
ax.text(CX, 3.38, 'LightPestFormer — System Performance Summary',
        ha='center', fontsize=12, fontweight='bold', color='#0D47A1')

stats = [
    ('INSECT204 Top-1 Acc', '89.73%', '#2E7D32'),
    ('INSECT204 Top-5 Acc', '98.21%', '#2E7D32'),
    ('AgroPest-12 mAP@50',  '79.4%',  '#0277BD'),
    ('Parameters',          '8.21 M',  '#6A1B9A'),
    ('FLOPs',              '634.5 M', '#6A1B9A'),
    ('Inference Latency',  '12.6 ms', '#C62828'),
    ('Throughput',         '79.2 FPS','#C62828'),
]
col_positions = [1.2, 3.2, 5.1, 7.1, 9.0, 10.9, 12.8]
for (stat_name, stat_val, scolor), cx2 in zip(stats, col_positions):
    box(cx2, 2.3, 1.7, 0.85, stat_val, stat_name, scolor, 12.5, 8)

arrow(CX, 3.6, CX, 3.58, '#37474F')  # tiny connector

plt.tight_layout(pad=1.2)
plt.savefig('/home/claude/actual_graphs/ARCH_architecture_diagram.png',
            dpi=160, bbox_inches='tight', facecolor='white')
print("Saved architecture diagram")
plt.close()
