"""
tests/test_safemask.py
----------------------
Run with:  python -m pytest tests/ -v
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
import numpy as np

from src.uncertainty.entropy import compute_entropy, get_warning_regions, compute_entropy_mc_dropout
from src.evaluation.metrics  import compute_iou, compute_pixel_accuracy, compute_uncertainty_metrics


# ============================================================
# entropy.py
# ============================================================

class TestComputeEntropy:
    def test_output_shape(self):
        logits = torch.randn(2, 19, 64, 128)
        out    = compute_entropy(logits)
        assert out.shape == (2, 64, 128), "Entropy shape should be (B, H, W)"

    def test_range(self):
        logits = torch.randn(1, 19, 32, 32)
        out    = compute_entropy(logits)
        assert out.min().item() >= 0.0,  "Entropy must be ≥ 0"
        assert out.max().item() <= 1.01, "Normalised entropy must be ≤ 1"

    def test_uniform_logits_high_entropy(self):
        """Uniform logits → probabilities all equal → maximum entropy (≈1)."""
        logits = torch.zeros(1, 19, 8, 8)   # all-zero logits → uniform softmax
        out    = compute_entropy(logits)
        assert out.mean().item() > 0.99, "Uniform logits should give entropy ≈ 1"

    def test_peaked_logits_low_entropy(self):
        """One class has a very large logit → near-zero entropy."""
        logits        = torch.zeros(1, 19, 8, 8)
        logits[:, 0]  = 100.0                # class 0 dominates
        out           = compute_entropy(logits)
        assert out.mean().item() < 0.01, "Peaked logits should give entropy ≈ 0"


class TestGetWarningRegions:
    def test_output_dtype(self):
        entropy = torch.rand(2, 64, 128)
        mask    = get_warning_regions(entropy, threshold=0.5)
        assert mask.dtype == torch.float32

    def test_threshold_respected(self):
        entropy          = torch.zeros(1, 4, 4)
        entropy[0, 0, 0] = 0.9   # above threshold
        entropy[0, 1, 1] = 0.3   # below threshold
        mask = get_warning_regions(entropy, threshold=0.5)
        assert mask[0, 0, 0].item() == 1.0
        assert mask[0, 1, 1].item() == 0.0

    def test_all_below_threshold(self):
        entropy = torch.full((1, 8, 8), 0.2)
        mask    = get_warning_regions(entropy, threshold=0.5)
        assert mask.sum().item() == 0.0

    def test_all_above_threshold(self):
        entropy = torch.full((1, 8, 8), 0.9)
        mask    = get_warning_regions(entropy, threshold=0.5)
        assert mask.sum().item() == 8 * 8


# ============================================================
# metrics.py
# ============================================================

class TestComputeIoU:
    def test_perfect_prediction(self):
        pred   = torch.zeros(1, 4, 4, dtype=torch.long)
        target = torch.zeros(1, 4, 4, dtype=torch.long)
        ious   = compute_iou(pred, target, num_classes=2)
        assert ious[0] == pytest.approx(1.0), "Perfect overlap → IoU = 1"

    def test_no_overlap(self):
        pred   = torch.zeros(1, 4, 4, dtype=torch.long)
        target = torch.ones(1,  4, 4, dtype=torch.long)
        ious   = compute_iou(pred, target, num_classes=2)
        assert ious[0] == pytest.approx(0.0), "No overlap → IoU = 0"

    def test_absent_class_is_nan(self):
        pred   = torch.zeros(1, 4, 4, dtype=torch.long)
        target = torch.zeros(1, 4, 4, dtype=torch.long)
        ious   = compute_iou(pred, target, num_classes=2)
        assert np.isnan(ious[1]), "Absent class should return NaN"

    def test_output_length(self):
        pred   = torch.zeros(1, 4, 4, dtype=torch.long)
        target = torch.zeros(1, 4, 4, dtype=torch.long)
        ious   = compute_iou(pred, target, num_classes=5)
        assert len(ious) == 5


class TestComputePixelAccuracy:
    def test_all_correct(self):
        pred   = torch.zeros(1, 8, 8, dtype=torch.long)
        target = torch.zeros(1, 8, 8, dtype=torch.long)
        acc    = compute_pixel_accuracy(pred, target)
        assert acc == pytest.approx(1.0)

    def test_half_correct(self):
        pred   = torch.zeros(2, 8, 8, dtype=torch.long)
        target = torch.zeros(2, 8, 8, dtype=torch.long)
        target[:, :, 4:] = 1          # right half wrong
        acc = compute_pixel_accuracy(pred, target)
        assert acc == pytest.approx(0.5)

    def test_all_wrong(self):
        pred   = torch.zeros(1, 4, 4, dtype=torch.long)
        target = torch.ones(1,  4, 4, dtype=torch.long)
        acc    = compute_pixel_accuracy(pred, target)
        assert acc == pytest.approx(0.0)


class TestComputeUncertaintyMetrics:
    def _make_inputs(self):
        B, H, W  = 1, 8, 8
        pred     = torch.zeros(B, H, W, dtype=torch.long)
        target   = torch.zeros(B, H, W, dtype=torch.long)
        target[:, :, 4:] = 1         # right half is incorrect

        entropy  = torch.rand(B, H, W)
        # warning mask flags the right half
        warning  = torch.zeros(B, H, W)
        warning[:, :, 4:] = 1.0
        return pred, target, entropy, warning

    def test_returns_expected_keys(self):
        pred, target, entropy, warning = self._make_inputs()
        result = compute_uncertainty_metrics(pred, target, entropy, warning)
        for key in ['avg_entropy_correct', 'avg_entropy_incorrect',
                    'warning_precision', 'warning_recall']:
            assert key in result

    def test_precision_and_recall_range(self):
        pred, target, entropy, warning = self._make_inputs()
        result = compute_uncertainty_metrics(pred, target, entropy, warning)
        assert 0.0 <= result['warning_precision'] <= 1.0
        assert 0.0 <= result['warning_recall']    <= 1.0

    def test_perfect_warning_region(self):
        """Warning mask exactly matches incorrect pixels → P=R=1."""
        B, H, W  = 1, 4, 4
        pred     = torch.zeros(B, H, W, dtype=torch.long)
        target   = torch.zeros(B, H, W, dtype=torch.long)
        target[0, 0, 0] = 1

        entropy = torch.rand(B, H, W)
        warning = torch.zeros(B, H, W)
        warning[0, 0, 0] = 1.0

        result = compute_uncertainty_metrics(pred, target, entropy, warning)
        assert result['warning_precision'] == pytest.approx(1.0)
        assert result['warning_recall']    == pytest.approx(1.0)