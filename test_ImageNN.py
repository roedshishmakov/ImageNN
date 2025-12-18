import pytest
import os
import sys
import tempfile
from PIL import Image
from main import NeuralNetwork


sys.argv = ['test_script.py', '--help']

from ImageNN import (
        create_standart_nn,
        validate_arguments,
        sredpix,
    )
from exceptions import (
        ArgumentError,
        ValidationError,
        PathError,
        EpochError,
    )

def test_creates_nn_instance():
    nn = create_standart_nn()
    assert isinstance(nn, NeuralNetwork)
    assert len(nn.layers) == 4

def test_has_correct_architecture():
    nn = create_standart_nn()
    assert len(nn.layers) == 4
    assert len(nn.layers[0]) == 256
    assert len(nn.layers[1]) == 32
    assert len(nn.layers[2]) == 32
    assert len(nn.layers[3]) == 10

def test_empty_list_raises_error():
    with pytest.raises(ZeroDivisionError, match="Error in image"):
        sredpix([])

def test_single_pixel():
    pixels = [255, 255, 255]
    result = sredpix(pixels)
    assert result == 255.0

def test_no_command_raises_error():
    with pytest.raises(ArgumentError, match="No command specified"):
        validate_arguments([], [])


def test_unknown_command():
    with pytest.raises(ValidationError, match="Unknown command flag"):
        validate_arguments(['unknown'], [])


def test_load_missing_arguments():
    with pytest.raises(ArgumentError, match="requires 2 arguments"):
        validate_arguments(['--load'], ['model_name'])


def test_train_missing_arguments():
    with pytest.raises(ArgumentError, match="requires 3 arguments"):
        validate_arguments(['--train'], ['model', 'config'])


def test_simple_train_invalid_epochs():
    with pytest.raises(EpochError, match="must be an integer"):
        validate_arguments(['-s'], ['model', 'not_a_number', 'path'])


def test_simple_train_negative_epochs():
    with pytest.raises(EpochError, match="must be positive"):
        validate_arguments(['--simple-train'], ['model', '-5', 'path'])

def test_simple_train_too_many_epochs():
    with pytest.raises(EpochError, match="too high"):
        validate_arguments(['--simple-train'], ['model', '20000', 'path'])

def test_no_command_raises_error():
    with pytest.raises(ArgumentError):
        validate_arguments([], [])


def test_unknown_command():
    with pytest.raises(ValidationError):
        validate_arguments(['--unknown'], [])


def test_load_missing_arguments():
    with pytest.raises(ArgumentError):
        validate_arguments(['--load'], ['model_name'])


def test_train_missing_arguments():
    with pytest.raises(ArgumentError):
        validate_arguments(['--train'], ['model', 'config'])


def test_simple_train_invalid_epochs():
    with pytest.raises(EpochError):
        validate_arguments(['-s'], ['model', 'not_a_number', 'path'])


def test_simple_train_negative_epochs():
    with pytest.raises(EpochError):
        validate_arguments(['--simple-train'], ['model', '-5', 'path'])


def test_simple_train_too_many_epochs():
    with pytest.raises(EpochError):
        validate_arguments(['--simple-train'], ['model', '20000', 'path'])


def test_conflicting_flags():
    with pytest.raises(PathError):
        validate_arguments(['--load', '--train'], ['model', 'arg1', 'arg2'])

def test_valid_arguments_help():
    try:
        validate_arguments(['--help'], [])
    except Exception as e:
        pytest.fail(f"validate_arguments raised {type(e).__name__} unexpectedly: {e}")

def test_valid_arguments_create_config():
    with pytest.raises(ArgumentError):
        validate_arguments(['--create-config'], [])


def test_valid_load_arguments():
    with pytest.raises(ArgumentError):
        validate_arguments(['--load'], ['default'])


def test_valid_train_arguments():
    with pytest.raises(ArgumentError):
        validate_arguments(['--train'], ['only_one_arg'])

def test_valid_simple_train_arguments():
    with pytest.raises(ArgumentError):
        validate_arguments(['--simple-train'], ['only_one_arg'])
