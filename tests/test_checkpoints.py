"""Checkpoint writing: atomic saves, model-only best checkpoints, failure handling."""

from __future__ import annotations

import os

import torch

from src.train import Trainer


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 6)

    def forward(self, batch):
        return {"logits": self.linear(batch["features"])}


def _trainer(tmp_path):
    return Trainer(
        _TinyModel(),
        device=torch.device("cpu"),
        loss_computer=None,
        out_dir=tmp_path,
    )


def test_save_checkpoint_is_atomic_and_loadable(tmp_path):
    trainer = _trainer(tmp_path)
    path = tmp_path / "checkpoint_best.pt"
    assert trainer.save_checkpoint(path, epoch=3, include_optimizer=False)
    assert path.exists()
    assert not (tmp_path / "checkpoint_best.pt.tmp").exists()

    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["epoch"] == 3
    assert state["optimizer_state_dict"] is None
    assert "model_state_dict" in state


def test_save_checkpoint_failure_does_not_leave_partial_file(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path)
    path = tmp_path / "checkpoint_epoch_0.pt"

    def _boom(*args, **kwargs):
        raise RuntimeError("basic_ios::clear: iostream error")

    monkeypatch.setattr(torch, "save", _boom)
    assert trainer.save_checkpoint(path, epoch=0) is False
    assert not path.exists()
    assert not (tmp_path / "checkpoint_epoch_0.pt.tmp").exists()


def test_prune_removes_oldest_epoch_checkpoints(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.save_total_limit = 2
    for epoch in range(3):
        path = tmp_path / f"checkpoint_epoch_{epoch}.pt"
        path.write_bytes(b"x")
        os.utime(path, (1_700_000_000 + epoch, 1_700_000_000 + epoch))
    trainer._prune_checkpoints()
    remaining = sorted(p.name for p in tmp_path.glob("checkpoint_epoch_*.pt"))
    assert remaining == ["checkpoint_epoch_1.pt", "checkpoint_epoch_2.pt"]
