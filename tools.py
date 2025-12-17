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
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(f"HDF5 file not found: {filename}")

    with h5py.File(filename, 'r') as f:
        layers_data = []

        if 'layers' in f:
            for layer_name in sorted(f['layers'].keys()):
                layer_grp = f[f'layers/{layer_name}']
                layer_weights = []

                neuron_datasets = [key for key in layer_grp.keys()
                                   if key.startswith('neuron_')]

                for neuron_key in sorted(neuron_datasets):
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
    :param metadata: метаданные
    :type metadata: dict
    """

    with h5py.File(filename, 'w') as f:
        layers_grp = f.create_group('layers')

        for i, layer_weights in enumerate(model_data):
            layer_grp = layers_grp.create_group(f'layer_{i}')

            for j, neuron_weights in enumerate(layer_weights):
                if neuron_weights:
                    layer_grp.create_dataset(f'neuron_{j}_weights',
                                             data=np.array(neuron_weights))

def import_json(filename):
    if not os.path.exists(filename):
        return False

    r = ''
    with open(filename, 'r') as f:
        for line in f:
            r += line
    try:
        return json.loads(r)
    except json.decoder.JSONDecodeError:
        raise Exception(f'Could not parse {filename}')

def export_json(filename, data):
    r = json.dumps(data, sort_keys=True,indent=4)
    with open(filename, 'w') as f:
        f.write(r)

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
