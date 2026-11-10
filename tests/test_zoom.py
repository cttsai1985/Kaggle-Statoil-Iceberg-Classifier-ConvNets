import numpy as np
import pytest

from iceberg.data import augment, keras_zoom
from iceberg.runtime import resolve_config


def test_keras_zoom_reproduces_extreme_coordinate_crop_and_identity():
    y, x = np.indices((75, 75), dtype=np.float32)
    image = np.stack([x, y])
    original = image.copy()
    np.testing.assert_array_equal(keras_zoom(image, np.ones(2)), image)
    result = keras_zoom(image, np.array([0.1, 0.1]))
    np.testing.assert_array_equal(np.unique(result[0]), np.arange(34, 43))
    np.testing.assert_array_equal(np.unique(result[1]), np.arange(34, 43))
    np.testing.assert_array_equal(image, original)


def test_study_policies_keep_random_schedule_and_channel_alignment():
    image = np.stack([np.arange(75 * 75, dtype=np.float32).reshape(75, 75)] * 2)
    image[1] += 100
    original = image.copy()
    for seed in range(60):
        states = []
        for policy in ["none", "mild_5pct", "extreme_5pct"]:
            rng = np.random.default_rng(seed)
            result = augment(image, rng, zoom_policy=policy)
            np.testing.assert_allclose(result[1] - result[0], 100, atol=0.002)
            np.testing.assert_array_equal(image, original)
            assert result.shape == image.shape and np.isfinite(result).all()
            states.append(rng.bit_generator.state)
        assert states[0] == states[1] == states[2]


def test_default_policy_and_rejection_are_explicit():
    assert resolve_config({})["zoom_policy"] == "current"
    with pytest.raises(ValueError, match="zoom policy"):
        resolve_config({"zoom_policy": "typo"})
    with pytest.raises(ValueError, match="zoom policy"):
        augment(np.zeros((2, 75, 75)), np.random.default_rng(1), zoom_policy="typo")
