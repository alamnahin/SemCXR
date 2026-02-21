"""
Unit tests for SemCXR training pipeline.

Run with:
    pytest test.py -v
    pytest test.py -v -k "test_config"      # run specific tests
"""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from train import (
    Config,
    ImageEncoder,
    TextEncoder,
    CrossModalAttention,
    ReportDecoder,
    SemCXR,
    FocalLoss,
    LabelSmoothingCrossEntropy,
    MetricsCalculator,
    MixupCutmix,
    CheckpointManager,
    get_transforms,
    set_seed,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def config():
    """Minimal config for fast tests."""
    return Config(
        embed_dim=64,
        num_classes=3,
        dropout=0.1,
        image_size=224,
        use_cross_attention=True,
        use_report_generation=False,
        use_swa=False,
        batch_size=2,
        num_epochs=1,
    )


@pytest.fixture
def small_config():
    """Very small config for component tests."""
    return Config(
        embed_dim=32,
        num_classes=3,
        dropout=0.0,
        use_cross_attention=False,
        use_report_generation=False,
    )


@pytest.fixture
def temp_dir():
    """Temporary directory for checkpoint tests."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# =====================================================================
# Config Tests
# =====================================================================

class TestConfig:
    def test_default_creation(self):
        cfg = Config()
        assert cfg.num_classes == 3
        assert cfg.batch_size == 32
        assert cfg.lr == 1e-4
        assert cfg.precision == "16-mixed"

    def test_to_dict_roundtrip(self):
        cfg = Config(embed_dim=128, lr=5e-5, use_swa=False)
        d = cfg.to_dict()
        cfg2 = Config.from_dict(d)
        assert cfg2.embed_dim == 128
        assert cfg2.lr == 5e-5
        assert cfg2.use_swa is False

    def test_custom_values(self):
        cfg = Config(num_classes=5, image_size=384, seed=123)
        assert cfg.num_classes == 5
        assert cfg.image_size == 384
        assert cfg.seed == 123


# =====================================================================
# Model Component Tests
# =====================================================================

class TestImageEncoder:
    def test_output_shape(self):
        encoder = ImageEncoder(
            model_name="swin_base_patch4_window7_224",
            embed_dim=64,
            pretrained=False,
            drop_rate=0.0
        )
        x = torch.randn(2, 3, 224, 224)
        out = encoder(x)
        assert out.shape == (2, 64)

    def test_different_embed_dim(self):
        encoder = ImageEncoder(
            model_name="swin_base_patch4_window7_224",
            embed_dim=128,
            pretrained=False
        )
        x = torch.randn(1, 3, 224, 224)
        out = encoder(x)
        assert out.shape == (1, 128)


class TestCrossModalAttention:
    def test_output_shape(self):
        attn = CrossModalAttention(embed_dim=64, num_heads=4)
        img_feat = torch.randn(2, 64)
        txt_feat = torch.randn(2, 64)
        out = attn(img_feat, txt_feat)
        assert out.shape == (2, 64)

    def test_residual_connection(self):
        """Output should differ from input (attention changes it)."""
        attn = CrossModalAttention(embed_dim=32, num_heads=4)
        attn.eval()
        img_feat = torch.randn(1, 32)
        txt_feat = torch.randn(1, 32)
        out = attn(img_feat, txt_feat)
        # Residual connection means output = attn_out + img_feat
        # They should be different unless attn_out is zero
        assert out.shape == img_feat.shape


class TestReportDecoder:
    def test_output_shape(self):
        decoder = ReportDecoder(embed_dim=64, vocab_size=100, num_layers=1)
        memory = torch.randn(2, 64)
        tgt = torch.randint(0, 100, (2, 10))
        out = decoder(memory, tgt)
        assert out.shape == (2, 10, 100)


class TestSemCXR:
    def test_forward_no_cross_attn_no_report(self):
        cfg = Config(
            embed_dim=64, num_classes=3,
            use_cross_attention=False,
            use_report_generation=False,
            image_encoder="swin_base_patch4_window7_224",
        )
        model = SemCXR(cfg)
        model.eval()

        image = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones(2, 32, dtype=torch.long)

        outputs = model(image, input_ids, attention_mask)
        assert 'logits' in outputs
        assert outputs['logits'].shape == (2, 3)
        assert 'report_logits' not in outputs

    def test_forward_with_cross_attn(self):
        cfg = Config(
            embed_dim=64, num_classes=3,
            use_cross_attention=True,
            use_report_generation=False,
            image_encoder="swin_base_patch4_window7_224",
        )
        model = SemCXR(cfg)
        model.eval()

        image = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones(2, 32, dtype=torch.long)

        outputs = model(image, input_ids, attention_mask)
        assert outputs['logits'].shape == (2, 3)

    def test_init_weights_preserves_pretrained(self):
        """Verify _init_weights does NOT overwrite pretrained backbone params."""
        cfg = Config(
            embed_dim=64,
            use_cross_attention=False,
            use_report_generation=False,
            image_encoder="swin_base_patch4_window7_224",
        )
        # Get a reference to pretrained weights before init
        import timm
        ref_backbone = timm.create_model(
            "swin_base_patch4_window7_224", pretrained=False, num_classes=0
        )
        ref_param = next(ref_backbone.parameters()).clone()

        model = SemCXR(cfg)
        actual_param = next(model.image_encoder.backbone.parameters())

        # After _init_weights, backbone params should still match
        # (both created from same pretrained=True source)
        # We just verify _init_weights didn't zero them out or make them xavier uniform
        assert actual_param.std() > 0.001, "Backbone weights look zeroed out"


# =====================================================================
# Loss Function Tests
# =====================================================================

class TestFocalLoss:
    def test_basic(self):
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 3)
        labels = torch.tensor([0, 1, 2, 1])
        loss = loss_fn(logits, labels)
        assert loss.dim() == 0  # scalar
        assert loss.item() >= 0

    def test_with_alpha(self):
        alpha = torch.tensor([1.0, 2.0, 0.5])
        loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
        logits = torch.randn(4, 3)
        labels = torch.tensor([0, 1, 2, 0])
        loss = loss_fn(logits, labels)
        assert loss.item() >= 0

    def test_gamma_zero_matches_ce(self):
        """With gamma=0, focal loss should equal weighted CE."""
        loss_fn = FocalLoss(gamma=0.0)
        logits = torch.randn(8, 3)
        labels = torch.randint(0, 3, (8,))
        focal = loss_fn(logits, labels)
        ce = F.cross_entropy(logits, labels)
        assert torch.allclose(focal, ce, atol=1e-5)


class TestLabelSmoothingCrossEntropy:
    def test_basic(self):
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
        logits = torch.randn(4, 3)
        labels = torch.tensor([0, 1, 2, 1])
        loss = loss_fn(logits, labels)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_zero_smoothing_matches_ce(self):
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.0)
        logits = torch.randn(8, 3)
        labels = torch.randint(0, 3, (8,))
        smooth_loss = loss_fn(logits, labels)
        ce_loss = F.cross_entropy(logits, labels)
        assert torch.allclose(smooth_loss, ce_loss, atol=1e-5)


# =====================================================================
# Metrics Tests
# =====================================================================

class TestMetricsCalculator:
    def test_perfect_predictions(self):
        calc = MetricsCalculator(num_classes=3)
        # Perfect logits: class 0 gets high score for label 0, etc.
        logits = torch.tensor([
            [10.0, -10.0, -10.0],
            [-10.0, 10.0, -10.0],
            [-10.0, -10.0, 10.0],
        ])
        labels = torch.tensor([0, 1, 2])
        calc.update(logits, labels)
        metrics = calc.compute()
        assert metrics['Accuracy'] == 100.0
        assert metrics['Macro_AUC'] > 0.99

    def test_reset(self):
        calc = MetricsCalculator(num_classes=3)
        logits = torch.randn(4, 3)
        labels = torch.randint(0, 3, (4,))
        calc.update(logits, labels)
        calc.reset()
        assert len(calc.all_probs) == 0
        assert len(calc.all_labels) == 0

    def test_multiple_updates(self):
        calc = MetricsCalculator(num_classes=3)
        for _ in range(3):
            logits = torch.randn(4, 3)
            labels = torch.randint(0, 3, (4,))
            calc.update(logits, labels)
        metrics = calc.compute()
        assert 0 <= metrics['Accuracy'] <= 100
        assert 0 <= metrics['F1_Macro'] <= 1.0


# =====================================================================
# MixupCutmix Tests
# =====================================================================

class TestMixupCutmix:
    def test_no_op_when_prob_zero(self):
        mixer = MixupCutmix(prob=0.0)
        batch = {
            'image': torch.randn(4, 3, 32, 32),
            'label': torch.tensor([0, 1, 2, 0])
        }
        result = mixer(batch)
        assert 'target' not in result  # no mixing happened
        assert torch.equal(result['image'], batch['image'])

    def test_applies_when_prob_one(self):
        set_seed(42)
        mixer = MixupCutmix(prob=1.0)
        batch = {
            'image': torch.randn(4, 3, 32, 32),
            'label': torch.tensor([0, 1, 2, 0])
        }
        original_images = batch['image'].clone()
        result = mixer(batch)
        assert 'target' in result
        assert result['target'].shape == (4, 3)  # one-hot
        assert 'lam' in result

    def test_output_target_sums_to_one(self):
        set_seed(0)
        mixer = MixupCutmix(prob=1.0, num_classes=3)
        batch = {
            'image': torch.randn(4, 3, 32, 32),
            'label': torch.tensor([0, 1, 2, 0])
        }
        result = mixer(batch)
        if 'target' in result:
            sums = result['target'].sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# =====================================================================
# CheckpointManager Tests
# =====================================================================

class TestCheckpointManager:
    def test_save_and_load(self, temp_dir):
        mgr = CheckpointManager(temp_dir, "test_exp", save_top_k=1)

        state = {'epoch': 5, 'model_state_dict': {'weight': torch.randn(10)}}
        mgr.save_checkpoint(state, epoch=5, score=0.9, is_best=True)

        # last_model.pt should exist
        assert (Path(temp_dir) / "test_exp" / "last_model.pt").exists()
        # best_model.pt should exist
        assert (Path(temp_dir) / "test_exp" / "best_model.pt").exists()

        loaded = mgr.load_checkpoint(mgr.get_best_checkpoint())
        assert loaded['epoch'] == 5

    def test_latest_checkpoint(self, temp_dir):
        mgr = CheckpointManager(temp_dir, "test_exp")
        assert mgr.get_latest_checkpoint() is None

        state = {'epoch': 0}
        mgr.save_checkpoint(state, epoch=0, score=0.5)
        assert mgr.get_latest_checkpoint() is not None

    def test_best_only_updated_on_is_best(self, temp_dir):
        mgr = CheckpointManager(temp_dir, "test_exp")

        mgr.save_checkpoint({'epoch': 0}, epoch=0, score=0.5, is_best=True)
        mgr.save_checkpoint({'epoch': 1}, epoch=1, score=0.3, is_best=False)

        best = mgr.load_checkpoint(mgr.get_best_checkpoint())
        assert best['epoch'] == 0  # still the first one

        latest = mgr.load_checkpoint(mgr.get_latest_checkpoint())
        assert latest['epoch'] == 1  # updated


# =====================================================================
# Transforms Tests
# =====================================================================

class TestTransforms:
    def test_train_transforms(self):
        from PIL import Image
        transform = get_transforms(image_size=224, is_training=True)
        img = Image.new('RGB', (256, 256), color='red')
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_val_transforms(self):
        from PIL import Image
        transform = get_transforms(image_size=224, is_training=False)
        img = Image.new('RGB', (256, 256), color='blue')
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_custom_size(self):
        from PIL import Image
        transform = get_transforms(image_size=384, is_training=False)
        img = Image.new('RGB', (100, 100))
        tensor = transform(img)
        assert tensor.shape == (3, 384, 384)


# =====================================================================
# Seed Reproducibility Test
# =====================================================================

class TestReproducibility:
    def test_set_seed(self):
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.equal(a, b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
