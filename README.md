# robotics-datasets-hub

One-command download, convert, and visualize curated AI/Robotics datasets.

AI/ロボティクス研究で使える厳選データセットを、1コマンドでダウンロード・可視化・デモ再生できるCLIツール。

## Install / インストール

```bash
pip install -e .
```

For 3D visualization (Open3D, Plotly):
```bash
pip install -e ".[viz3d]"
```

## Quick Start / クイックスタート

```bash
# List all datasets / データセット一覧
rdh list

# Search datasets / 検索
rdh list navigation

# Show dataset details / 詳細表示
rdh info covla

# Download a dataset / ダウンロード
rdh download covla --split mini --output ./data/

# Visualize samples / 可視化
rdh viz covla --samples 5
```

## Supported Datasets / 対応データセット

| Name | Task | Modalities | Paper |
|------|------|------------|-------|
| **CoVLA** | Autonomous Driving VLA | Vision, Language, Action | [arXiv:2408.10845](https://arxiv.org/abs/2408.10845) |
| **HM3D-OVON** | Open-Vocab Object Goal Nav | RGB-D, 3D mesh, Language | [arXiv:2409.14296](https://arxiv.org/abs/2409.14296) |
| **PoLaRIS** | Maritime Detection/Tracking | RGB, TIR, Radar, LiDAR | [arXiv:2412.06192](https://arxiv.org/abs/2412.06192) |
| **MCD** | Multi-modal SLAM | Multi-LiDAR, Camera, IMU, UWB | [arXiv:2403.11496](https://arxiv.org/abs/2403.11496) |
| **GGRt** | Pose-free 3D Gaussian Splatting | RGB | [arXiv:2403.10147](https://arxiv.org/abs/2403.10147) |
| **SLABIM** | SLAM + BIM | LiDAR, Camera, IMU, BIM | [arXiv:2502.16856](https://arxiv.org/abs/2502.16856) |
| **HK_MEMS** | LiDAR SLAM (Extreme) | MEMS LiDAR, Camera, GNSS, INS | [JFR](https://onlinelibrary.wiley.com/doi/10.1002/rob.70136) |
| **GEODE** | LiDAR SLAM (Degeneracy) | Multi-LiDAR, Stereo, IMU | [arXiv:2409.04961](https://arxiv.org/abs/2409.04961) |

## Web Dashboard / ダッシュボード

```bash
pip install -e ".[dashboard]"
rdh dashboard
```

ブラウザでデータセットの検索・フィルタリング・詳細閲覧ができます。

## Jupyter Notebook

`notebooks/01_quickstart.ipynb` でインタラクティブにデータセットを探索できます。

## Adding a Dataset / データセットの追加

`registry/` にYAMLファイルを追加するだけで新しいデータセットを登録できます:

```yaml
name: my_dataset
display_name: "My Dataset"
description: "Description of the dataset"
paper_url: "https://arxiv.org/abs/..."
project_url: "https://..."
github_url: "https://github.com/..."
huggingface_id: ""
modalities:
  - RGB
  - LiDAR
tasks:
  - SLAM
license: "CC BY 4.0"
size_hint: "10GB"
tags:
  - outdoor
download:
  method: huggingface  # or: gdown, wget, git
  url: "org/dataset-name"
```

## Development / 開発

```bash
pip install -e ".[dev]"
ruff check src/ tests/
pytest tests/ -v
```

## Citations / 引用

各データセットの論文を引用してください。詳細は `rdh info <dataset_name>` で確認できます。

## License

Apache License 2.0
