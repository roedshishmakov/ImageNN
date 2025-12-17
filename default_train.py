import tools
import activations
import keras
import datetime

from main import *
from tools import *
from PIL import Image

def edit_x_data(traindata, p):
    """
    Функция подготовки входа датасета

    :param traindata: датасет
    :type traindata: list
    :param p: количество примеров
    :type p: int
    :return: список входных значений
    :rtype: list
    """

    resarr = []
    size = 16
    i = 0
    for x in traindata[:p]:
        arr = flat(Image.fromarray(x).resize((size, size)))
        rr = []
        for z in arr:
            if z > 150:
                rr.append(1)
            else: rr.append(0)
        resarr.append(rr)
        i += 1
    return resarr

def edit_y_data(traindata, p):
    """
    Функция подготовки выходов для обучения

    :param traindata: датасет
    :type traindata: list
    :param p: число ответов
    :type p: int
    :return: список ответов
    :rtype: list
    """

    resarr = []
    for x in traindata[:p]:
        a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        a[x] = 1.0
        resarr.append(a)
    return resarr

def load_mnist():
    """
    Функция загрузки датасета mnist

    :return: список входов и выходов
    :rtype: list
    """

    train_data = []
    test_data = []

    train_num = 150
    test_num = 100

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train_data = edit_x_data(x_train, train_num)
    x_test_data = edit_x_data(x_test, test_num)

    y_train_data = edit_y_data(y_train, train_num)
    y_test_data = edit_y_data(y_test, test_num)

    for i in range(len(x_train_data)):
        train_data.append((x_train_data[i], y_train_data[i]))
    for i in range(len(x_test_data)):
        test_data.append((x_test_data[i], y_test_data[i]))
    return [train_data, test_data]

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
        model_data = tools.import_h5_model(filename)
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

nn = NeuralNetwork()
nn.add_input_layer(256)
nn.add_layer(32, activation_class = activations.ActivationRelu, random_radius=0.1, use_bias = True)
nn.add_layer(32, activation_class = activations.ActivationRelu, random_radius=0.1, use_bias = True)
nn.add_layer(10, activation_class = activations.ActivationSoftmax, random_radius=0.1)

mnist = load_mnist()
train_data, test_data = mnist[0], mnist[1]
print("MNIST loaded!")

print("Do you want to load saved model?(y/n):", end = ' ')
z = input()
if z == 'y':
    print("Enter name of save:", end = ' ')
    z = input()
    DATA_FILE_NAME = "weight_saves/" + z + ".h5"
    LOSS_FILE_NAME = "loss_saves/" + z + ".txt"
    load_model_h5(nn, DATA_FILE_NAME)
else:
    print("Enter new save name:", end = ' ')
    z = input()
    DATA_FILE_NAME = "weight_saves/" + z + ".h5"
    LOSS_FILE_NAME = "loss_saves/" + z + ".txt"
    total_loss_statistics = []
    print("Enter number of Epochs:", end = ' ')
    ep = int(input())
    for i in range(ep):
        print('EPOCH #{}'.format(i))
        loss_total = nn.train(train_data, 0.01, verbose = True)
        print('LOSS: {:.4f}'.format(loss_total))
        total_loss_statistics.append(loss_total)

    save_model_h5(nn, DATA_FILE_NAME)
    f = open(LOSS_FILE_NAME, 'w')
    for s in total_loss_statistics:
        f.write(str(s) + '\n')
    f.close()

print("Show loss graph?(y/n):", end = ' ')
ans = input()
if ans == 'y':
    show_loss_save(LOSS_FILE_NAME)
else: print()

print("Enter number of tests:", end = ' ')
test_len = int(input())
for i in range(test_len):
    nn.run(test_data[i][0])
    print("Answer:", nn.get_best_index())
    print("Correct answer:", test_data[i][1].index(1))
    print()