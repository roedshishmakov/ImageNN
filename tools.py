import os
import h5py

import numpy as np
from matplotlib import pyplot as plt

def show_loss_save(LOSS_FILE_NAME):
    """
    Функция выводит на экран график с потерями в процессе обучения сохраненной модели

    :param LOSS_FILE_NAME: название файла с сохранением
    :type LOSS_FILE_NAME: str
    """

    tls = []
    i = 0
    try:
        with open(LOSS_FILE_NAME) as f:
            for s in f:
                tls.append([i, float(s[:-1])])
                i += 1
        plt.plot([x[0] for x in tls], [y[1] for y in tls])
        plt.title("Total loss")
        plt.xlabel("Number of epochs")
        plt.ylabel("Loss")
        plt.show()
    except Exception: raise FileNotFoundError("Cannot show loss from file {}".format(LOSS_FILE_NAME))

def flat(inp):
    """
    Функция преобразует матрицу значений в вектор

    :param inp: матрица на входе
    :type inp: list
    :return: выходной вектор
    :rtype: list
    """
    inp = np.array(inp)
    output_dim = 1
    for dim in inp.shape:
        output_dim *= dim

    output = inp.reshape(output_dim)
    return output.tolist()

def ensure_directory_exists(filepath):
    """
    Проверяет существование директории для файла и создает ее если нужно

    :param filepath: путь к файлу
    :type filepath: str
    """

    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def import_h5_model(filename):
    """
    Импорт модели из HDF5 файла

    :param filename: путь к файлу .h5
    :return: данные модели в формате списка слоев
    :rtype: list
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(f"HDF5 file not found: {filename}")

    with h5py.File(filename, 'r') as f:
        layers_data = []

        if 'layers' in f:
            layer_groups = []
            for key in f['layers'].keys():
                if key.startswith('layer_'):
                    layer_groups.append(key)

            layer_groups.sort(key=lambda x: int(x.split('_')[1]))

            for layer_key in layer_groups:
                layer_grp = f[f'layers/{layer_key}']
                layer_weights = []

                neuron_keys = []
                for key in layer_grp.keys():
                    if key.startswith('neuron_'):
                        neuron_keys.append(key)

                neuron_keys.sort(key=lambda x: int(x.split('_')[1]))

                for neuron_key in neuron_keys:
                    weights = layer_grp[neuron_key][:].tolist()
                    layer_weights.append(weights)

                layers_data.append(layer_weights)

        return layers_data

def export_h5_model(filename, model_data):
    """
    Экспорт модели в HDF5 файл

    :param filename: путь к файлу .h5
    :type filename: str
    :param model_data: данные модели
    :type model_data: list
    """

    with h5py.File(filename, 'w') as f:
        layers_grp = f.create_group('layers')

        for i, layer_weights in enumerate(model_data):
            layer_grp = layers_grp.create_group(f'layer_{i}')

            for j, neuron_weights in enumerate(layer_weights):
                dataset_name = f'neuron_{j}'
                layer_grp.create_dataset(dataset_name, data=np.array(neuron_weights))

def save_model_h5(network, filename):
    """
    Сохраняет модель в формате HDF5

    :param network: нейронная сеть
    :type network: NeuralNetwork
    :param filename: путь к файлу .h5
    :type filename: str
    """

    ensure_directory_exists(filename)
    model_data = network.export()

    export_h5_model(filename, model_data)
    print(f"Модель сохранена в {filename}")

def load_model_h5(network, filename):
    """
    Загружает модель из формата HDF5

    :param network: нейронная сеть для загрузки весов
    :type network: NeuralNetwork
    :param filename: путь к файлу .h5
    :type filename: str
    :return: True если загрузка успешна, иначе False
    :rtype: bool
    """

    if not os.path.exists(filename):
        print(f"Файл {filename} не найден")
        return False

    try:
        model_data = import_h5_model(filename)
        if model_data:
            network.import_(model_data)
            print(f"Модель загружена из {filename}")
            return True
        else:
            print(f"Не удалось загрузить модель из {filename}")
            return False
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return False

def save_config(config, filename):
    """
    Сохраняет конфигурацию сети в текстовый файл

    :param config: конфигурация сети (словарь с layers и training)
    :type config: dict
    :param filename: путь к файлу .config
    :type filename: str
    """

    ensure_directory_exists(filename)

    with open(filename, 'w') as f:
        f.write("# Конфигурация нейронной сети\n")
        f.write("# Формат: [layer_type]:[size]:[activation]:[use_bias]:[random_radius]\n\n")

        for layer in config['layers']:
            use_bias_str = 'true' if layer['use_bias'] else 'false'
            f.write(f"dense:{layer['size']}:{layer['activation']}:{use_bias_str}:{layer['random_radius']}\n")

        f.write("\n# Параметры обучения\n")
        for key, value in config['training'].items():
            f.write(f"{key}={value}\n")

    print(f"Конфигурация сохранена в {filename}")

def load_config(filename):
    """
    Загружает конфигурацию сети из текстового файла

    :param filename: путь к файлу .config
    :type filename: str
    :return: конфигурация сети
    :rtype: dict
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Config file not found: {filename}")

    config = {
        'layers': [],
        'training': {}
    }

    with open(filename, 'r') as f:
        lines = f.readlines()

    in_training_section = False

    for line_num, line in enumerate(lines, 1):
        line = line.strip()

        if not line or line.startswith('#'):
            if config['layers'] and not in_training_section:
                in_training_section = True
            continue

        if not in_training_section:
            parts = line.split(':')
            if len(parts) != 5:
                raise ValueError(f"Ошибка в строке {line_num}: неверный формат слоя")

            layer_config = {
                'type': parts[0].strip(),
                'size': int(parts[1].strip()),
                'activation': parts[2].strip(),
                'use_bias': parts[3].strip().lower() == 'true',
                'random_radius': float(parts[4].strip())
            }

            config['layers'].append(layer_config)
        else:
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                if key in ['epochs']:
                    config['training'][key] = int(value)
                elif key in ['learning_rate', 'clip_value']:
                    config['training'][key] = float(value)
                elif key in ['use_cross_entropy']:
                    config['training'][key] = value.lower() == 'true'
                else:
                    config['training'][key] = value

    # Значения по умолчанию
    if 'learning_rate' not in config['training']:
        config['training']['learning_rate'] = 0.1
    if 'epochs' not in config['training']:
        config['training']['epochs'] = 10
    if 'clip_value' not in config['training']:
        config['training']['clip_value'] = 5.0
    if 'use_cross_entropy' not in config['training']:
        config['training']['use_cross_entropy'] = True

    return config