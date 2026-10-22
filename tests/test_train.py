from src.train import optimizer_update_succeeded


def test_scaler_skip_does_not_advance_optimizer_update():
    assert optimizer_update_succeeded(1024.0, 1024.0)
    assert optimizer_update_succeeded(1024.0, 2048.0)
    assert not optimizer_update_succeeded(1024.0, 512.0)
