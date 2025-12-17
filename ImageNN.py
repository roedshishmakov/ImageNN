import sys
import os
import tools
import activations

import numpy as np

from PIL import Image
from main import *
from exceptions import *

def validate_arguments(flags, args):
    """
    Функция проверки корректности входных аргументов команды

    :param flags: список флагов
    :type flags: list[str]
    :param args: список аргументов
    :type args: list[str]
    :raises ArgumentError: при недостаточном количестве аргументов
    :raises ValidationError: при неправильном формате аргументов
    :raises PathError: при неверно указанном пути
    :raises EpochError: при неверно указанных эпохах
    :raises IncorrectCommand: при неверной комбинации флагов
    """

    required_args = {
        '--load': 2,
        '-l': 2,
        '--train': 3,
        '-t': 3,
        '--graph': 1,
        '-g': 1,
        '--help': 0,
        '-h': 0
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
                f"{main_flag} requires {required_count} arguments: model_name, epochs, and train_dataset_path")
        elif main_flag in ['--graph', '-g']:
            raise ArgumentError(f"{main_flag} requires {required_count} argument: model_name")
        else:
            raise ArgumentError(f"Insufficient arguments for {main_flag}")

    if main_flag in ['--train', '-t']:
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

            if main_flag not in ['--load', '-l', '--train', '-t']:
                raise ValidationError("--graph flag can only be used with --load or --train")

    # Проверка конфликтов флагов
    flag_set = set(flags)
    conflicting_combinations = [
        {'--load', '--train'}, {'-l', '-t'}, {'-l', '--train'}, {'--load', '-t'}
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
    print('--load or -l: loading model, example: python3 cli.py --load <model_name> <images_to_recognize_path>')
    print()
    print('--graph or -g: show loss graph, example: python3 cli.py -l --graph "<model_name>"')
    print()
    print('--train or -t: train model, example: python3 cli.py --train <model_name> <epochs> <train_dataset_path>')

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

nn = NeuralNetwork()
nn.add_input_layer(256)
nn.add_layer(32, activation_class = activations.ActivationRelu, random_radius=0.1, use_bias = True)
nn.add_layer(32, activation_class = activations.ActivationRelu, random_radius=0.1, use_bias = True)
nn.add_layer(10, activation_class = activations.ActivationSoftmax, random_radius=0.1)

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
    if '--load' in flags or '-l' in flags:
        tools.load_model_h5(nn, "weight_saves/" + args[0] + ".h5")
        if '--graph' in flags or '-g' in flags:
            tools.show_loss_save("loss_saves/" + args[0] + ".txt")

        ds = load_examples(args[1])
        for i in range(len(ds)):
            print('#---------------------------#')
            nn.run(ds[i])
            ansnn = nn.get_best_index()
            print(f'Test number {i + 1}')
            print("Answer: ", ansnn)

    elif '--train' in flags or '-t' in flags:
        z = args[0]
        DATA_FILE_NAME = "weight_saves/" + z + ".h5"
        LOSS_FILE_NAME = "loss_saves/" + z + ".txt"
        ep = int(args[1])
        total_loss_statistics = []
        train_data = load_examples(args[2], 1)
        for i in range(ep):
            print('EPOCH #{}'.format(i+1))
            print(len(train_data))
            loss_total = nn.train(train_data, 0.1)
            print('LOSS: {:.4f}'.format(loss_total))
            total_loss_statistics.append(loss_total)

        tools.save_model_h5(nn, DATA_FILE_NAME)
        f = open(LOSS_FILE_NAME, 'w')
        for s in total_loss_statistics:
            f.write(str(s) + '\n')
        f.close()

    elif '--help' in flags or '-h' in flags:
        show_help_info()
    else: raise IncorrectCommand("Arguments are incorrect")