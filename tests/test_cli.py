"""Tests for the CLI interface."""

from typer.testing import CliRunner

from rdh.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_list():
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "covla" in result.output


def test_list_search():
    result = runner.invoke(app, ["list", "navigation"])
    assert result.exit_code == 0
    assert "hm3d_ovon" in result.output


def test_info():
    result = runner.invoke(app, ["info", "covla"])
    assert result.exit_code == 0
    assert "CoVLA" in result.output
    assert "arxiv" in result.output


def test_info_not_found():
    result = runner.invoke(app, ["info", "nonexistent"])
    assert result.exit_code == 1
