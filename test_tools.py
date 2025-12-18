import os
import tempfile
import numpy as np
import pytest
import tools
from unittest.mock import Mock

def test_flat_2d_list():
    input_2d = [[1, 2, 3], [4, 5, 6]]
    expected = [1, 2, 3, 4, 5, 6]
    result = tools.flat(input_2d)
    assert result == expected


def test_flat_3d_list():
    input_3d = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    expected = [1, 2, 3, 4, 5, 6, 7, 8]
    result = tools.flat(input_3d)
    assert result == expected

def test_flat_empty_list():
    input_empty = []
    result = tools.flat(input_empty)
    assert result == []

def test_flat_single_element():
    input_single = [42]
    result = tools.flat(input_single)
    assert result == [42]

def test_flat_numpy_array():
    input_array = np.array([[1, 2], [3, 4]])
    expected = [1, 2, 3, 4]
    result = tools.flat(input_array)
    assert result == expected

def test_ensure_directory_exists_new():
    with tempfile.TemporaryDirectory() as tmpdir:
        new_dir = os.path.join(tmpdir, "new", "deep", "path")
        filepath = os.path.join(new_dir, "test.txt")

        assert not os.path.exists(new_dir)
        tools.ensure_directory_exists(filepath)
        assert os.path.exists(new_dir)

def test_ensure_directory_exists_existing():
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_dir = os.path.join(tmpdir, "existing")
        os.makedirs(existing_dir)
        filepath = os.path.join(existing_dir, "test.txt")

        tools.ensure_directory_exists(filepath)
        assert os.path.exists(existing_dir)

def test_export_import_h5_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test_model.h5")
        model_data = [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]
        ]

        tools.export_h5_model(test_file, model_data)
        assert os.path.exists(test_file)
        imported_data = tools.import_h5_model(test_file)
        assert len(imported_data) == 2
        assert len(imported_data[0]) == 2
        assert len(imported_data[1]) == 3
        assert imported_data[0][0][0] == pytest.approx(0.1)
        assert imported_data[0][1][2] == pytest.approx(0.6)
        assert imported_data[1][2][1] == pytest.approx(1.2)

def test_export_h5_model_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "empty_model.h5")
        model_data = []
        tools.export_h5_model(test_file, model_data)
        assert os.path.exists(test_file)

def test_import_h5_model_nonexistent():
    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent_file = os.path.join(tmpdir, "nonexistent.h5")

        with pytest.raises(FileNotFoundError, match="HDF5 file not found"):
            tools.import_h5_model(non_existent_file)

def test_save_load_model_h5_integration():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "integrated_model.h5")
        mock_network = Mock()
        mock_network.export.return_value = [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0], [6.0], [7.0]]
        ]

        tools.save_model_h5(mock_network, test_file)

        assert os.path.exists(test_file)

        mock_network.export.assert_called_once()

        mock_network2 = Mock()
        tools.load_model_h5(mock_network2, test_file)

        mock_network2.import_.assert_called_once()

        call_args = mock_network2.import_.call_args[0][0]

        assert len(call_args) == 2
        assert len(call_args[0]) == 2
        assert len(call_args[1]) == 3

def test_load_model_h5_nonexistent():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "nonexistent.h5")
        mock_network = Mock()

        result = tools.load_model_h5(mock_network, test_file)

        assert result is False
        mock_network.import_.assert_not_called()

def test_save_load_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test_config.config")
        config = {
            'layers': [
                {'type': 'dense', 'size': 32, 'activation': 'relu', 'use_bias': True, 'random_radius': 0.1},
                {'type': 'dense', 'size': 10, 'activation': 'softmax', 'use_bias': False, 'random_radius': 0.05}
            ],
            'training': {
                'learning_rate': 0.01,
                'epochs': 100,
                'clip_value': 3.0,
                'use_cross_entropy': True
            }
        }

        tools.save_config(config, test_file)
        assert os.path.exists(test_file)

        loaded_config = tools.load_config(test_file)

        assert len(loaded_config['layers']) == 2

        assert loaded_config['layers'][0]['type'] == 'dense'
        assert loaded_config['layers'][0]['size'] == 32
        assert loaded_config['layers'][0]['activation'] == 'relu'
        assert loaded_config['layers'][0]['use_bias'] is True
        assert loaded_config['layers'][0]['random_radius'] == pytest.approx(0.1)

        assert loaded_config['layers'][1]['use_bias'] is False

        assert loaded_config['training']['learning_rate'] == pytest.approx(0.01)
        assert loaded_config['training']['epochs'] == 100
        assert loaded_config['training']['clip_value'] == pytest.approx(3.0)
        assert loaded_config['training']['use_cross_entropy'] is True


def test_save_config_defaults():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "minimal.config")

        config = {
            'layers': [
                {'type': 'dense', 'size': 16, 'activation': 'sigmoid', 'use_bias': False, 'random_radius': 0.5}
            ],
            'training': {}
        }

        tools.save_config(config, test_file)
        loaded_config = tools.load_config(test_file)

        assert loaded_config['training']['learning_rate'] == pytest.approx(0.1)
        assert loaded_config['training']['epochs'] == 10
        assert loaded_config['training']['clip_value'] == pytest.approx(5.0)
        assert loaded_config['training']['use_cross_entropy'] is True


def test_load_config_nonexistent():
    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent_file = os.path.join(tmpdir, "nonexistent.config")

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            tools.load_config(non_existent_file)


def test_load_config_invalid_format():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "invalid.config")
        with open(test_file, 'w') as f:
            f.write("invalid:format:only:three:parts\n")

        with pytest.raises(ValueError):
            tools.load_config(test_file)


def test_save_config_creates_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        deep_path = os.path.join(tmpdir, "deep", "nested", "path")
        test_file = os.path.join(deep_path, "config.config")

        config = {
            'layers': [{'type': 'dense', 'size': 8, 'activation': 'relu', 'use_bias': True, 'random_radius': 0.2}],
            'training': {'learning_rate': 0.05}
        }
        assert not os.path.exists(deep_path)

        tools.save_config(config, test_file)
        assert os.path.exists(deep_path)
        assert os.path.exists(test_file)

def test_import_h5_model_file_not_found_error():
    with pytest.raises(FileNotFoundError, match="HDF5 file not found"):
        tools.import_h5_model("nonexistent.h5")

def test_load_config_file_not_found_error():
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        tools.load_config("nonexistent.config")