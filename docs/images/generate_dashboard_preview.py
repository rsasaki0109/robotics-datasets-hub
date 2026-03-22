"""Generate a matplotlib-based mock screenshot of the Streamlit dashboard."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Figure setup: 1200x700 at 150dpi
fig = plt.figure(figsize=(1200 / 150, 700 / 150), dpi=150)
fig.patch.set_facecolor('white')

# Colors
SIDEBAR_BG = '#262730'
SIDEBAR_TEXT = '#FAFAFA'
SIDEBAR_DIM = '#A0A0A0'
CARD_BG = '#1a1a2e'
CARD_BORDER = '#333333'
TITLE_COLOR = '#262730'
BODY_TEXT = '#AAAAAA'

MODALITY_COLORS = {
    'vision': '#4CAF50', 'language': '#FF9800', 'action': '#9C27B0',
    'RGB-D': '#2196F3', '3D-mesh': '#3F51B5',
    'RGB': '#4CAF50', 'thermal-infrared': '#F44336', 'radar': '#795548', 'LiDAR': '#2196F3',
}

# Layout constants
sidebar_w = 0.22
main_x = sidebar_w + 0.01
main_w = 1.0 - main_x - 0.02

# ---- SIDEBAR ----
sidebar = fig.add_axes([0, 0, sidebar_w, 1])
sidebar.set_xlim(0, 1)
sidebar.set_ylim(0, 1)
sidebar.set_facecolor(SIDEBAR_BG)
sidebar.set_xticks([])
sidebar.set_yticks([])
for spine in sidebar.spines.values():
    spine.set_visible(False)

y = 0.92
sidebar.text(0.1, y, 'Filters', color=SIDEBAR_TEXT, fontsize=11, fontweight='bold')

y -= 0.07
sidebar.text(0.1, y, 'Search datasets', color=SIDEBAR_DIM, fontsize=7)
y -= 0.04
search_box = FancyBboxPatch(
    (0.08, y - 0.02), 0.82, 0.035,
    boxstyle="round,pad=0.005",
    facecolor='#3a3a4a', edgecolor='#555555', linewidth=0.8,
)
sidebar.add_patch(search_box)
sidebar.text(0.14, y - 0.003, 'e.g. SLAM, lidar, driving', color='#666666', fontsize=5.5)

y -= 0.09
sidebar.text(0.1, y, 'Modalities', color=SIDEBAR_DIM, fontsize=7)
y -= 0.04
modalities_list = [
    '3D-mesh', 'LiDAR', 'RGB', 'RGB-D', 'action',
    'language', 'radar', 'thermal-infrared', 'vision',
]
for mod in modalities_list:
    cb = FancyBboxPatch(
        (0.1, y - 0.008), 0.04, 0.022,
        boxstyle="round,pad=0.002",
        facecolor='#3a3a4a', edgecolor='#555555', linewidth=0.5,
    )
    sidebar.add_patch(cb)
    sidebar.text(0.18, y, mod, color=SIDEBAR_TEXT, fontsize=5.5, va='center')
    y -= 0.035

y -= 0.02
sidebar.text(0.1, y, 'Tasks', color=SIDEBAR_DIM, fontsize=7)
y -= 0.04
tasks_list = [
    'autonomous-driving', 'embodied-AI', 'end-to-end-driving', 'maritime',
    'object-detection', 'object-goal-nav', 'object-tracking', 'open-vocabulary',
    'vision-language-action',
]
for task in tasks_list:
    cb = FancyBboxPatch(
        (0.1, y - 0.008), 0.04, 0.022,
        boxstyle="round,pad=0.002",
        facecolor='#3a3a4a', edgecolor='#555555', linewidth=0.5,
    )
    sidebar.add_patch(cb)
    sidebar.text(0.18, y, task, color=SIDEBAR_TEXT, fontsize=5, va='center')
    y -= 0.035

y -= 0.01
sidebar.plot([0.08, 0.9], [y, y], color='#444444', linewidth=0.5)
y -= 0.03
sidebar.text(0.1, y, '8 datasets registered', color=SIDEBAR_DIM, fontsize=6, fontweight='bold')
y -= 0.03
sidebar.text(
    0.1, y, 'Install: pip install robotics-datasets-hub',
    color=SIDEBAR_DIM, fontsize=4.5, family='monospace',
)

# ---- MAIN AREA ----
main = fig.add_axes([main_x, 0, main_w, 1])
main.set_xlim(0, 1)
main.set_ylim(0, 1)
main.set_facecolor('#FFFFFF')
main.set_xticks([])
main.set_yticks([])
for spine in main.spines.values():
    spine.set_visible(False)

main.text(0.03, 0.92, 'Robotics Datasets Hub', color=TITLE_COLOR, fontsize=16, fontweight='bold')
main.text(
    0.03, 0.87,
    'Browse, search, and download curated AI/Robotics datasets with a single command.',
    color='#666666', fontsize=7,
)
main.plot([0.03, 0.97], [0.845, 0.845], color='#E0E0E0', linewidth=0.5)
main.text(0.03, 0.82, 'Showing 8 dataset(s)', color='#999999', fontsize=6)

# ---- DATASET CARDS ----
cards_data = [
    {
        'name': 'CoVLA: Comprehensive\nVision-Language-Action Dataset',
        'desc': 'Large-scale vision-language-action dataset\nfor autonomous driving with 80+ hours of\ndriving data, 10k scenes, paired with...',
        'modalities': ['vision', 'language', 'action'],
        'tasks': 'autonomous-driving,\nvision-language-action,\nend-to-end-driving',
        'license': 'Non-commercial',
        'size': 'Mini: 50 scenes (~75GB full)',
    },
    {
        'name': 'HM3D-OVON: Open-Vocabulary\nObject Goal Navigation',
        'desc': 'Benchmark for open-vocabulary object goal\nnavigation using HM3D 3D scans. Agents\nmust navigate to objects described by...',
        'modalities': ['RGB-D', '3D-mesh', 'language'],
        'tasks': 'object-goal-navigation,\nembodied-AI,\nopen-vocabulary',
        'license': 'Matterport ToU',
        'size': '~130 GB',
    },
    {
        'name': 'PoLaRIS: Pohang Canal\nMaritime Detection & Tracking',
        'desc': 'Multi-sensor maritime dataset with RGB,\nthermal infrared, radar, and LiDAR for\nobject detection and tracking in canal...',
        'modalities': ['RGB', 'thermal-infrared', 'radar', 'LiDAR'],
        'tasks': 'object-detection,\nobject-tracking,\nmaritime',
        'license': 'Not specified',
        'size': '~360k labeled images',
    },
]

card_w = 0.30
card_h = 0.57
card_gap = 0.02
card_y = 0.20
start_x = 0.03

for i, card in enumerate(cards_data):
    cx = start_x + i * (card_w + card_gap)

    card_rect = FancyBboxPatch(
        (cx, card_y), card_w, card_h,
        boxstyle="round,pad=0.008",
        facecolor=CARD_BG, edgecolor=CARD_BORDER, linewidth=0.8,
    )
    main.add_patch(card_rect)

    ty = card_y + card_h - 0.04
    main.text(cx + 0.015, ty, card['name'], color='#FFFFFF', fontsize=6, fontweight='bold',
              verticalalignment='top', linespacing=1.3)

    ty -= 0.10
    main.text(cx + 0.015, ty, card['desc'], color=BODY_TEXT, fontsize=5,
              verticalalignment='top', linespacing=1.3)

    ty -= 0.10
    tag_x = cx + 0.015
    for mod in card['modalities']:
        color = MODALITY_COLORS.get(mod, '#607D8B')
        tag_w_px = len(mod) * 0.008 + 0.015
        tag = FancyBboxPatch(
            (tag_x, ty - 0.005), tag_w_px, 0.028,
            boxstyle="round,pad=0.004",
            facecolor=color, edgecolor='none', alpha=0.9,
        )
        main.add_patch(tag)
        main.text(tag_x + tag_w_px / 2, ty + 0.009, mod, color='white', fontsize=4.5,
                  ha='center', va='center', fontweight='bold')
        tag_x += tag_w_px + 0.005

    ty -= 0.05
    main.text(cx + 0.015, ty, f'Tasks: {card["tasks"]}', color='#CCCCCC', fontsize=4.5,
              verticalalignment='top', linespacing=1.3)

    ty -= 0.08
    main.text(cx + 0.015, ty, f'License: {card["license"]}', color='#CCCCCC', fontsize=4.8)

    ty -= 0.035
    main.text(cx + 0.015, ty, f'Size: {card["size"]}', color='#CCCCCC', fontsize=4.8)

    # Link buttons
    ty -= 0.05
    btn_x = cx + 0.015
    for label in ['Paper', 'GitHub', 'HuggingFace']:
        btn_w = len(label) * 0.007 + 0.015
        btn = FancyBboxPatch(
            (btn_x, ty - 0.005), btn_w, 0.028,
            boxstyle="round,pad=0.003",
            facecolor='#1a1a2e', edgecolor='#444444', linewidth=0.5,
        )
        main.add_patch(btn)
        main.text(btn_x + btn_w / 2, ty + 0.009, label, color='#E0E0E0', fontsize=4,
                  ha='center', va='center')
        btn_x += btn_w + 0.005

    # Download command expander
    ty -= 0.045
    main.text(cx + 0.015, ty, '> Download Command', color='#FF4B4B', fontsize=4.5)

    # Details button
    ty -= 0.035
    btn = FancyBboxPatch(
        (cx + 0.015, ty - 0.005), 0.06, 0.028,
        boxstyle="round,pad=0.003",
        facecolor='#FFFFFF', edgecolor='#FF4B4B', linewidth=0.8,
    )
    main.add_patch(btn)
    main.text(cx + 0.015 + 0.03, ty + 0.009, 'Details', color='#FF4B4B', fontsize=4.5,
              ha='center', va='center')

# Streamlit branding bar
brand_bar = fig.add_axes([0, 0, 1, 0.03])
brand_bar.set_xlim(0, 1)
brand_bar.set_ylim(0, 1)
brand_bar.set_facecolor('#F0F2F6')
brand_bar.set_xticks([])
brand_bar.set_yticks([])
for spine in brand_bar.spines.values():
    spine.set_visible(False)
brand_bar.text(0.5, 0.5, 'Made with Streamlit', color='#999999', fontsize=4,
               ha='center', va='center')

out_path = '/media/autoware/aa/ai_coding_ws/opendata_ws/robotics-datasets-hub/docs/images/dashboard_preview.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'Saved to {out_path}')
