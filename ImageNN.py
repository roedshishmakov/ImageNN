import sys
import os
import tools
import activations

import numpy as np

from PIL import Image
from main import *
from exceptions import *

def create_standart_nn():
    """
    Функция создания классической архитектуры нейросети
    """

    nn = NeuralNetwork()
    nn.add_input_layer(256)
    nn.add_layer(32, activation_class=activations.ActivationRelu, random_radius=0.1, use_bias=True)
    nn.add_layer(32, activation_class=activations.ActivationRelu, random_radius=0.1, use_bias=True)
    nn.add_layer(10, activation_class=activations.ActivationSoftmax, random_radius=0.1)
    return nn

def create_config_template(config_name):
    """
    Создает шаблон конфигурационного файла для нейронной сети
    :param config_name: Название конфиг файла
    :type config_name: str
    """

    config_content = """# Конфигурация нейронной сети
# Формат: [layer_type]:[size]:[activation]:[use_bias]:[random_radius]
# layer_type: dense (всегда dense в текущей реализации)
# size: количество нейронов в слое (целое число)
# activation: relu, sigmoid, softmax, transparent
# use_bias: true или false
# random_radius: радиус случайной инициализации (float)

# Входной слой (автоматически добавляется)
# Скрытые и выходные слои:
dense:32:relu:true:0.1
dense:32:relu:true:0.1
dense:10:softmax:false:0.1

# Параметры обучения
# Формат: [параметр]=[значение]
learning_rate=0.1
epochs=10
clip_value=5.0
use_cross_entropy=true
"""

    config_dir = "configs"
    os.makedirs(config_dir, exist_ok=True)
    config_filename = f"{config_dir}/{config_name}.config"

    with open(config_filename, 'w') as f:
        f.write(config_content)

    print(f"Конфигурационный файл создан: {config_filename}")
    print("Отредактируйте его для настройки архитектуры нейронной сети.")

def parse_config_file(config_file):
    """
    Парсит конфигурационный файл
    Возвращает словарь с параметрами сети
    """

    config = {
        'layers': [],
        'training': {}
    }

    with open(config_file, 'r') as f:
        lines = f.readlines()

    # Флаг для переключения между секциями
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
                raise ValidationError(
                    f"Ошибка в строке {line_num}: неверный формат слоя. Ожидается 5 частей, получено {len(parts)}")

            layer_config = {
                'type': parts[0].strip(),
                'size': int(parts[1].strip()),
                'activation': parts[2].strip(),
                'use_bias': parts[3].strip().lower() == 'true',
                'random_radius': float(parts[4].strip())
            }

            if layer_config['type'] != 'dense':
                raise ValidationError(f"Ошибка в строке {line_num}: неизвестный тип слоя '{layer_config['type']}'")

            config['layers'].append(layer_config)
        else:
            # Парсим параметры обучения
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

    if not config['layers']:
        raise ValidationError("Конфигурационный файл не содержит слоев")

    training_defaults = {
        'learning_rate': 0.1,
        'epochs': 10,
        'clip_value': 5.0,
        'use_cross_entropy': True
    }

    for key, default_value in training_defaults.items():
        if key not in config['training']:
            config['training'][key] = default_value

    return config

def create_network_from_config(config):
    """
    Создает нейронную сеть на основе конфигурации

    :param config: конфиг файл
    :type config: dict
    :return: нейросеть
    :rtype: NeuralNetwork
    """

    from main import NeuralNetwork

    nn = NeuralNetwork()

    nn.add_input_layer(256)

    # Маппинг строк активаций на классы
    activation_map = {
        'relu': activations.ActivationRelu,
        'sigmoid': activations.ActivationSigmoid,
        'softmax': activations.ActivationSoftmax,
        'transparent': activations.ActivationTransparent
    }

    # Добавляем слои
    for layer_config in config['layers']:
        activation_name = layer_config['activation']
        activation_class = activation_map.get(activation_name)

        if activation_class is None:
            raise ValidationError(f"Неизвестная функция активации: {activation_name}")

        nn.add_layer(
            size=layer_config['size'],
            activation_class=activation_class,
            random_radius=layer_config['random_radius'],
            use_bias=layer_config['use_bias']
        )

    return nn

def save_config_to_file(config, filename):
    """
    Сохраняет конфигурацию в текстовый файл

    :param config: Конфиг
    :type config: dict
    :param filename: название файла сохранения
    :type filename: str
    """

    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)

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

# def validate_arguments(flags, args):
#     """
#     Функция проверки корректности входных аргументов команды
#
#     :param flags: список флагов
#     :type flags: list[str]
#     :param args: список аргументов
#     :type args: list[str]
#     :raises ArgumentError: при недостаточном количестве аргументов
#     :raises ValidationError: при неправильном формате аргументов
#     :raises PathError: при неверно указанном пути
#     :raises EpochError: при неверно указанных эпохах
#     :raises IncorrectCommand: при неверной комбинации флагов
#     """
#
#     required_args = {
#         '--load': 2,
#         '-l': 2,
#         '--train': 3,
#         '-t': 3,
#         '--graph': 1,
#         '-g': 1,
#         '--help': 0,
#         '-h': 0
#     }
#
#     if not flags:
#         raise ArgumentError("No command specified. Use --help for usage information.")
#
#     main_flag = flags[0]
#
#     if main_flag not in required_args:
#         raise ValidationError(f"Unknown command flag: {main_flag}")
#
#     required_count = required_args[main_flag]
#     if len(args) < required_count:
#         if main_flag in ['--load', '-l']:
#             raise ArgumentError(f"{main_flag} requires {required_count} arguments: model_name and images_path")
#         elif main_flag in ['--train', '-t']:
#             raise ArgumentError(
#                 f"{main_flag} requires {required_count} arguments: model_name, epochs, and train_dataset_path")
#         elif main_flag in ['--graph', '-g']:
#             raise ArgumentError(f"{main_flag} requires {required_count} argument: model_name")
#         else:
#             raise ArgumentError(f"Insufficient arguments for {main_flag}")
#
#     if main_flag in ['--train', '-t']:
#         try:
#             epochs = int(args[1])
#             if epochs <= 0:
#                 raise EpochError(f"Number of epochs must be positive, got {epochs}")
#             if epochs > 10000:
#                 raise EpochError(f"Number of epochs too high: {epochs}. Maximum is 10000")
#         except ValueError:
#             raise EpochError(f"Epochs must be an integer, got '{args[1]}'")
#
#         if not os.path.exists(args[2]):
#             raise PathError(f"Training dataset path does not exist: {args[2]}")
#
#     elif main_flag in ['--load', '-l']:
#         if not os.path.exists(args[1]):
#             raise PathError(f"Test images path does not exist: {args[1]}")
#
#     elif main_flag in ['--graph', '-g']:
#         loss_file = "loss_saves/" + args[0] + ".txt"
#         if not os.path.exists(loss_file):
#             raise PathError(f"Loss file not found: {loss_file}. Train the model first.")
#
#     if len(flags) > 1:
#         for flag in flags[1:]:
#             if flag not in ['--graph', '-g']:
#                 raise ValidationError(f"Unexpected flag: {flag}. Only --graph/-g can be used with other flags")
#
#             if main_flag not in ['--load', '-l', '--train', '-t']:
#                 raise ValidationError("--graph flag can only be used with --load or --train")
#
#     # Проверка конфликтов флагов
#     flag_set = set(flags)
#     conflicting_combinations = [
#         {'--load', '--train'}, {'-l', '-t'}, {'-l', '--train'}, {'--load', '-t'}
#     ]
#
#     for combo in conflicting_combinations:
#         if combo.issubset(flag_set):
#             raise IncorrectCommand(f"Cannot use {combo} together. Choose either loading or training.")

def validate_arguments(flags, args):
    """
    Функция проверки корректности входных аргументов команды
    """
    required_args = {
        '--load': 2,
        '-l': 2,
        '--train': 3,
        '-t': 3,
        '--graph': 1,
        '-g': 1,
        '--help': 0,
        '-h': 0,
        '--create-config': 1,
        '-c': 1,
        '--simple-train': 3,
        '-s': 3
    }

    if not flags:
        raise ArgumentError("No command specified. Use --help for usage information.")

    main_flag = flags[0]

    if main_flag not in required_args:
        raise ValidationError(f"Unknown command flag: {main_flag}")

    required_count = required_args[main_flag]
    if len(args) < required_count:
        if main_flag in ['--load', '-l']:
            raise ArgumentError(f"{main_flag} requires {required_count} arguments: model_name and images_path")
        elif main_flag in ['--train', '-t']:
            raise ArgumentError(
                f"{main_flag} requires {required_count} arguments: model_name, config_file, and train_dataset_path")
        elif main_flag in ['--simple-train', '-s']:
            raise ArgumentError(
                f"{main_flag} requires {required_count} arguments: model_name, epochs, and train_dataset_path")
        elif main_flag in ['--graph', '-g']:
            raise ArgumentError(f"{main_flag} requires {required_count} argument: model_name")
        elif main_flag in ['--create-config', '-c']:
            raise ArgumentError(f"{main_flag} requires {required_count} argument: config_name")
        else:
            raise ArgumentError(f"Insufficient arguments for {main_flag}")

    if main_flag in ['--train', '-t']:
        if not os.path.exists(args[1]):
            raise PathError(f"Config file does not exist: {args[1]}")

        if not os.path.exists(args[2]):
            raise PathError(f"Training dataset path does not exist: {args[2]}")

    elif main_flag in ['--simple-train', '-s']:
        try:
            epochs = int(args[1])
            if epochs <= 0:
                raise EpochError(f"Number of epochs must be positive, got {epochs}")
            if epochs > 10000:
                raise EpochError(f"Number of epochs too high: {epochs}. Maximum is 10000")
        except ValueError:
            raise EpochError(f"Epochs must be an integer, got '{args[1]}'")

        if not os.path.exists(args[2]):
            raise PathError(f"Training dataset path does not exist: {args[2]}")

    elif main_flag in ['--load', '-l']:
        if not os.path.exists(args[1]):
            raise PathError(f"Test images path does not exist: {args[1]}")

    elif main_flag in ['--graph', '-g']:
        loss_file = "loss_saves/" + args[0] + ".txt"
        if not os.path.exists(loss_file):
            raise PathError(f"Loss file not found: {loss_file}. Train the model first.")

    if len(flags) > 1:
        for flag in flags[1:]:
            if flag not in ['--graph', '-g']:
                raise ValidationError(f"Unexpected flag: {flag}. Only --graph/-g can be used with other flags")

            if main_flag not in ['--load', '-l', '--train', '-t', '--simple-train', '-s']:
                raise ValidationError("--graph flag can only be used with --load or --train")

    # Проверка конфликтов флагов
    flag_set = set(flags)
    conflicting_combinations = [
        {'--load', '--train'}, {'-l', '-t'}, {'-l', '--train'}, {'--load', '-t'},
        {'--load', '--simple-train'}, {'-l', '-s'}, {'--train', '--simple-train'},
        {'-t', '-s'}
    ]

    for combo in conflicting_combinations:
        if combo.issubset(flag_set):
            raise IncorrectCommand(f"Cannot use {combo} together. Choose either loading or training.")

def sredpix(a):
    """
    Функция нахождения среднего значения пикселя

    :param a: список с параметрами пикселя
    :type a: list
    :return: среднее значение параметров
    :rtype: float
    """

    if len(a) != 0:
        return sum(a)/len(a)
    else: raise ZeroDivisionError("Error in image")

def show_logo():
    """
    Функция отображения логотипа программы
    """

    s = ['+============================================+',
         '| ___                            _   _ _   _ |',
         '||_ _|_ __ ___   __ _  __ _  ___| \ | | \ | ||',
         '| | || `_ ` _ \ / _` |/ _` |/ _ \  \| |  \| ||',
         '| | || | | | | | (_| | (_| |  __/ |\  | |\  ||',
         '||___|_| |_| |_|\__,_|\__, |\___|_| \_|_| \_||',
         '|                     |___/                  |',
         '+============================================+']
    for i in s:
        print(i)
    print()

def show_help_info():
    """
    Функция отображения информации при запуске программы без аргументов
    """

    print('Available arguments:')
    print('--load or -l: loading model, example: python3 ImageNN.py --load <model_name> <images_to_recognize_path>')
    print()
    print('--graph or -g: show loss graph, example: python3 ImageNN.py -l --graph "<model_name>"')
    print()
    print(
        '--train or -t: train model with config, example: python3 ImageNN.py --train <model_name> <config_file> <train_dataset_path>')
    print()
    print(
        '--simple-train or -s: train with default architecture (backward compatibility), example: python3 ImageNN.py --simple-train <model_name> <epochs> <train_dataset_path>')
    print()
    print('--create-config or -c: create config template, example: python3 ImageNN.py --create-config <config_name>')

def load_examples(path, purpose = 0):
    """
    Функция загрузки своего датасета

    :param path: путь до папки
    :type path: str
    :param purpose: цель использования датасета(0 -
    :type purpose: int
    :return: список с преобразованными данными из фото и правильными ответами
    :rtype: list
    """

    names = os.listdir(path)
    dataset = []
    if purpose == 0:
        for name in names:
            name = path + '/' + name
            if ('.png' in str(name)) or ('.jpg' in str(name)):
                arr = []

                img = (Image.open(name).resize((16, 16)))
                np_img = np.array(img)

                for i in range(len(np_img)):
                    new_row = []
                    for j in np_img[i]:
                        sr = sredpix(j)
                        new_row.append(1) if sr < 170 else new_row.append(0)
                    arr.append(new_row)

                arr = tools.flat(arr)
                dataset.append(arr)
        return dataset
    elif purpose == 1:
        for name in names:
            o = [0.0]*10
            o[int(name.split('_')[0])] = 1.0
            name = path + '/' + name
            if ('.png' in str(name)) or ('.jpg' in str(name)):
                arr = []

                img = (Image.open(name).resize((16, 16)))
                np_img = np.array(img)

                for i in range(len(np_img)):
                    new_row = []
                    for j in np_img[i]:
                        sr = sredpix(j)
                        new_row.append(1) if sr < 170 else new_row.append(0)
                    arr.append(new_row)

                arr = tools.flat(arr)
                dataset.append((arr, o))
        return dataset

inpargs = sys.argv[1:]
flags = []
args = []
if len(inpargs) < 1:
    raise ArgumentError("You need to use at least one argument.")
else:
    show_logo()
    for a in inpargs:
        if '-' in a:
            flags.append(a)
        else: args.append(a)
    validate_arguments(flags, args)
    # if '--load' in flags or '-l' in flags:
    #     tools.load_model_h5(nn, "weight_saves/" + args[0] + ".h5")
    #     if '--graph' in flags or '-g' in flags:
    #         tools.show_loss_save("loss_saves/" + args[0] + ".txt")
    #
    #     ds = load_examples(args[1])
    #     for i in range(len(ds)):
    #         print('#---------------------------#')
    #         nn.run(ds[i])
    #         ansnn = nn.get_best_index()
    #         print(f'Test number {i + 1}')
    #         print("Answer: ", ansnn)
    #
    # elif '--train' in flags or '-t' in flags:
    #     z = args[0]
    #     DATA_FILE_NAME = "weight_saves/" + z + ".h5"
    #     LOSS_FILE_NAME = "loss_saves/" + z + ".txt"
    #     ep = int(args[1])
    #     total_loss_statistics = []
    #     train_data = load_examples(args[2], 1)
    #     for i in range(ep):
    #         print('EPOCH #{}'.format(i+1))
    #         print(len(train_data))
    #         loss_total = nn.train(train_data, 0.1)
    #         print('LOSS: {:.4f}'.format(loss_total))
    #         total_loss_statistics.append(loss_total)
    #
    #     tools.save_model_h5(nn, DATA_FILE_NAME)
    #     f = open(LOSS_FILE_NAME, 'w')
    #     for s in total_loss_statistics:
    #         f.write(str(s) + '\n')
    #     f.close()
    #
    # elif '--help' in flags or '-h' in flags:
    #     show_help_info()
    # else: raise IncorrectCommand("Arguments are incorrect")
    if '--load' in flags or '-l' in flags:
        model_name = args[0]
        images_path = args[1]

        # Пытаемся загрузить конфигурацию
        config_file = f"configs/{model_name}.config"
        if os.path.exists(config_file):
            try:
                config = parse_config_file(config_file)
                nn = create_network_from_config(config)
                print(f"Загружена конфигурация сети из: {config_file}")
            except Exception as e:
                print(f"Ошибка загрузки конфигурации: {e}")
                print("Используется стандартная архитектура")
                create_standart_nn()
        else:
            print("Используется стандартная архитектура")
            create_standart_nn()
        tools.load_model_h5(nn, "weight_saves/" + model_name + ".h5")

        if '--graph' in flags or '-g' in flags:
            tools.show_loss_save("loss_saves/" + model_name + ".txt")

        ds = load_examples(images_path)
        for i in range(len(ds)):
            print('#---------------------------#')
            nn.run(ds[i])
            ansnn = nn.get_best_index()
            print(f'Test number {i + 1}')
            print("Answer: ", ansnn)

    elif '--train' in flags or '-t' in flags:
        model_name = args[0]
        config_file = args[1]
        train_path = args[2]

        config = parse_config_file("configs/" + config_file + ".config")

        nn = create_network_from_config(config)

        config_save_path = f"configs/{model_name}.config"
        save_config_to_file(config, config_save_path)

        DATA_FILE_NAME = "weight_saves/" + model_name + ".h5"
        LOSS_FILE_NAME = "loss_saves/" + model_name + ".txt"

        training_config = config['training']
        epochs = training_config['epochs']
        learning_rate = training_config['learning_rate']
        clip_value = training_config['clip_value']
        use_cross_entropy = training_config['use_cross_entropy']

        total_loss_statistics = []
        train_data = load_examples(train_path, 1)

        for i in range(epochs):
            print(f'EPOCH #{i + 1}/{epochs}')
            loss_total = nn.train(train_data, learning_rate,
                                  clip_value=clip_value,
                                  use_cross_entropy=use_cross_entropy)
            print(f'LOSS: {loss_total:.4f}')
            total_loss_statistics.append(loss_total)

        tools.save_model_h5(nn, DATA_FILE_NAME)

        with open(LOSS_FILE_NAME, 'w') as f:
            for s in total_loss_statistics:
                f.write(str(s) + '\n')

        print(f"Модель сохранена: {DATA_FILE_NAME}")
        print(f"Конфигурация сохранена: {config_save_path}")

    elif '--simple-train' in flags or '-s' in flags:
        z = args[0]
        epochs = int(args[1])
        train_path = args[2]

        DATA_FILE_NAME = "weight_saves/" + z + ".h5"
        LOSS_FILE_NAME = "loss_saves/" + z + ".txt"

        # Создаем стандартную сеть
        create_standart_nn()

        total_loss_statistics = []
        train_data = load_examples(train_path, 1)

        for i in range(epochs):
            print('EPOCH #{}'.format(i + 1))
            loss_total = nn.train(train_data, 0.1)
            print('LOSS: {:.4f}'.format(loss_total))
            total_loss_statistics.append(loss_total)

        tools.save_model_h5(nn, DATA_FILE_NAME)

        with open(LOSS_FILE_NAME, 'w') as f:
            for s in total_loss_statistics:
                f.write(str(s) + '\n')

        std_config = {
            'layers': [
                {'type': 'dense', 'size': 32, 'activation': 'relu', 'use_bias': True, 'random_radius': 0.1},
                {'type': 'dense', 'size': 32, 'activation': 'relu', 'use_bias': True, 'random_radius': 0.1},
                {'type': 'dense', 'size': 10, 'activation': 'softmax', 'use_bias': False, 'random_radius': 0.1}
            ],
            'training': {
                'learning_rate': 0.1,
                'epochs': epochs,
                'clip_value': 5.0,
                'use_cross_entropy': True
            }
        }

        save_config_to_file(std_config, f"configs/{z}.config")

    elif '--create-config' in flags or '-c' in flags:
        config_name = args[0]
        create_config_template(config_name)

    elif '--help' in flags or '-h' in flags:
        show_help_info()

    else:
        raise IncorrectCommand("Arguments are incorrect")