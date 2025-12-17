import os
import h5py
import json

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
    with open(LOSS_FILE_NAME) as f:
        for s in f:
            tls.append([i, float(s[:-1])])
            i += 1
    plt.plot([x[0] for x in tls], [y[1] for y in tls])
    plt.title("Total loss")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.show()

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

def export_h5_model(filename, model_data, metadata=None):
    """
    Экспорт модели в HDF5 файл

    :param filename: путь к файлу .h5
    :type filename: str
    :param model_data: данные модели
    :type model_data: list
    :param metadata: метаданные
    :type metadata: dict
    """

    with h5py.File(filename, 'w') as f:
        if metadata:
            for key, value in metadata.items():
                f.attrs[key] = value

        layers_grp = f.create_group('layers')

        for i, layer_weights in enumerate(model_data):
            layer_grp = layers_grp.create_group(f'layer_{i}')

            for j, neuron_weights in enumerate(layer_weights):
                dataset_name = f'neuron_{j}'
                layer_grp.create_dataset(dataset_name, data=np.array(neuron_weights))

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
