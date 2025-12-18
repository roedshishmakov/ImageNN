import pytest
import os
import sys
import tempfile
from PIL import Image
from main import NeuralNetwork


original_argv = sys.argv
sys.argv = ['test_script.py', '--help']

from ImageNN import (
        create_standart_nn,
        create_config_template,
        parse_config_file,
        create_network_from_config,
        save_config_to_file,
        validate_arguments,
        sredpix,
        load_examples,
        show_logo,
        show_help_info,
        create_network_from_config,
        save_config_to_file
    )
from exceptions import (
        ArgumentError,
        ValidationError,
        PathError,
        EpochError,
        IncorrectCommand
    )
# try:
#     sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#
#     from ImageNN import (
#         create_standart_nn,
#         create_config_template,
#         parse_config_file,
#         create_network_from_config,
#         save_config_to_file,
#         validate_arguments,
#         sredpix,
#         load_examples,
#         show_logo,
#         show_help_info,
#         create_network_from_config,
#         save_config_to_file
#     )
#
#     # Импортируем исключения отдельно
#     from exceptions import (
#         ArgumentError,
#         ValidationError,
#         PathError,
#         EpochError,
#         IncorrectCommand
#     )
#
#     from main import NeuralNetwork
#
# finally:
#     # Восстанавливаем оригинальные аргументы
#     sys.argv = original_argv
#
#
# # ==================== ТЕСТЫ ====================

def test_creates_nn_instance():
    """Тест создания нейронной сети"""
    nn = create_standart_nn()
    assert isinstance(nn, NeuralNetwork)


def test_has_correct_architecture():
    """Тест архитектуры сети"""
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
    """Тест отсутствия команды"""
    with pytest.raises(ArgumentError, match="No command specified"):
        validate_arguments([], [])


def test_unknown_command():
    """Тест неизвестной команды"""
    with pytest.raises(ValidationError, match="Unknown command flag"):
        validate_arguments(['--unknown'], [])


def test_load_missing_arguments():
    """Тест недостающих аргументов для --load"""
    with pytest.raises(ArgumentError, match="requires 2 arguments"):
        validate_arguments(['--load'], ['model_name'])


def test_train_missing_arguments():
    """Тест недостающих аргументов для --train"""
    with pytest.raises(ArgumentError, match="requires 3 arguments"):
        validate_arguments(['--train'], ['model', 'config'])


def test_simple_train_invalid_epochs():
    """Тест некорректного числа эпох"""
    with pytest.raises(EpochError, match="must be an integer"):
        validate_arguments(['-s'], ['model', 'not_a_number', 'path'])


def test_simple_train_negative_epochs():
    """Тест отрицательного числа эпох"""
    with pytest.raises(EpochError, match="must be positive"):
        validate_arguments(['--simple-train'], ['model', '-5', 'path'])

def test_simple_train_too_many_epochs():
    """Тест слишком большого числа эпох"""
    with pytest.raises(EpochError, match="too high"):
        validate_arguments(['--simple-train'], ['model', '20000', 'path'])

def test_no_command_raises_error():
    """Тест отсутствия команды"""
    with pytest.raises(ArgumentError):
        validate_arguments([], [])


def test_unknown_command():
    """Тест неизвестной команды"""
    with pytest.raises(ValidationError):
        validate_arguments(['--unknown'], [])


def test_load_missing_arguments():
    """Тест недостающих аргументов для --load"""
    with pytest.raises(ArgumentError):
        validate_arguments(['--load'], ['model_name'])


def test_train_missing_arguments():
    """Тест недостающих аргументов для --train"""
    with pytest.raises(ArgumentError):
        validate_arguments(['--train'], ['model', 'config'])


def test_simple_train_invalid_epochs():
    """Тест некорректного числа эпох"""
    with pytest.raises(EpochError):
        validate_arguments(['-s'], ['model', 'not_a_number', 'path'])


def test_simple_train_negative_epochs():
    """Тест отрицательного числа эпох"""
    with pytest.raises(EpochError):
        validate_arguments(['--simple-train'], ['model', '-5', 'path'])


def test_simple_train_too_many_epochs():
    """Тест слишком большого числа эпох"""
    with pytest.raises(EpochError):
        validate_arguments(['--simple-train'], ['model', '20000', 'path'])


def test_conflicting_flags():
    """Тест конфликтующих флагов"""
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
