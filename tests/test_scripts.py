"""Test the run_corr.py script functionality."""

import subprocess
import sys


def test_run_corr_script_help():
    """Test that the script shows help without errors."""
    result = subprocess.run(
        [sys.executable, 'scripts/run_corr.py', '--help'],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert 'Simple parcel-based correlation analysis' in result.stdout
    assert '--subjects' in result.stdout
    assert '--exclusions-file' in result.stdout


def test_run_corr_script_missing_exclusions():
    """Test that script fails gracefully when exclusions file is missing."""
    result = subprocess.run(
        [
            sys.executable,
            'scripts/run_corr.py',
            '--exclusions-file',
            'nonexistent.json',
        ],
        capture_output=True,
        text=True,
    )

    # Should exit with error code due to missing file
    assert result.returncode != 0
