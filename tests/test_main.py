"""Test main pipeline functionality."""

from network_parcel_corr.main import main


def test_main_smoke():
    """Smoke test for main function."""
    # Just verify that main runs without error
    main()
