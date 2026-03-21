"""Tests for dataset-specific visualizers (import checks)."""

import importlib


def test_import_covla():
    mod = importlib.import_module("rdh.datasets.covla")
    assert hasattr(mod, "viz_scene")
    assert hasattr(mod, "viz_multi_scene_trajectories")
    assert hasattr(mod, "load_scene")


def test_import_polaris():
    mod = importlib.import_module("rdh.datasets.polaris")
    assert hasattr(mod, "viz_sensor_comparison")
    assert hasattr(mod, "viz_detection_annotations")


def test_import_mcd():
    mod = importlib.import_module("rdh.datasets.mcd")
    assert hasattr(mod, "viz_trajectories")
    assert hasattr(mod, "viz_sequence_stats")


def test_import_hm3d_ovon():
    mod = importlib.import_module("rdh.datasets.hm3d_ovon")
    assert hasattr(mod, "viz_episode_overview")
    assert hasattr(mod, "viz_episode_detail")


def test_import_pohang_canal():
    mod = importlib.import_module("rdh.datasets.pohang_canal")
    assert hasattr(mod, "viz_pohang_canal")
    assert hasattr(mod, "load_gps")
    assert hasattr(mod, "load_ahrs")
    assert hasattr(mod, "latlon_to_local")


def test_covla_load_scene_synthetic():
    """Test loading synthetic CoVLA scene data."""
    from pathlib import Path

    from rdh.datasets.covla import list_scenes, load_scene

    demo_dir = Path(__file__).resolve().parents[1] / "data" / "covla_demo"
    if not demo_dir.exists():
        return  # Skip if no synthetic data
    scenes = list_scenes(demo_dir)
    assert len(scenes) > 0
    scene = load_scene(demo_dir, scenes[0])
    assert scene["n_frames"] > 0
    assert "frames" in scene


def test_pohang_canal_load_gps():
    """Test loading real Pohang Canal GPS data."""
    from pathlib import Path

    from rdh.datasets.pohang_canal import load_gps

    base = Path(__file__).resolve().parents[1] / "data" / "polaris_pohang00"
    gps_path = base / "navigation" / "gps.txt"
    if not gps_path.exists():
        return  # Skip if not downloaded
    gps = load_gps(gps_path)
    assert len(gps) > 0
    assert gps.shape[1] == 4  # timestamp, lat, lon, alt
