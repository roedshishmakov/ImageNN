import random
import types
import math

import activations as act

class Neuron:
    """
    Класс нейрона

    :ivar activation_class: класс активационной функции
    :type activation_class: ActivationBase
    :ivar input: вход нейрона
    :type input: float
    :ivar output: выход нейрона
    :type output: float
    :ivar link_input: список линков на вход
    :type link_input: list
    :ivar link_output: список линков на выход
    :type link_output: list
    """

    def __init__(self, activation_class):
        """
        Конструктор класса

        :param activation_class:  Активационная функция
        :type activation_class: ActivationBase
        """
        self.activation_class = activation_class
        self.input = 0
        self.output = 0
        self.link_input = []
        self.link_output = []

    def add_link_input(self, link):
        """
        Добавление линка на вход

        :param link: добавляемый линк
        :type link: Link
        :raises TypeError: при добавлении в список линков не линка
        """

        if not isinstance(link, Link):
            raise TypeError("Link must be of type Link")

        self.link_input.append(link)
        return self

    def add_link_output(self, link):
        """
        Добавление линка на выход

        :param link: добавляемый линк
        :type link: Link
        :raises TypeError: при добавлении не линка в список выходных линков
        """

        if not isinstance(link, Link):
            raise TypeError(f"Link {link} must be of type Link")

        self.link_output.append(link)
        return self

    def add(self, signal):
        """
        Подача сигнала на вход нейрона

        :param signal: подаваемый сигнал
        """

        if not isinstance(signal, float):
            raise TypeError(f"Signal {signal} must be of type float")

        self.input += signal
        return self

    def reset(self, input = 0):
        """
        Сброс входа нейрона

        :param input: новый вход, по умолчанию None
        :type input: int
        """

        if not(isinstance(input, int) or isinstance(input, float)):
            raise TypeError(f"Input {input} must be of type float")

        self.input = input

    def activation(self):
        """
        Запуск нейрона
        """

        try:
            self.output = self.get_activation(self.input)
            return self
        except TypeError: raise TypeError(f"Input {self.input} must be of type float")

    def get_activation(self, x):
        """
        Выполняет активационную функцию нейрона

        :param x: вход
        :type x: float
        :return: результат активации
        :rtype: float
        """

        try:
            return self.activation_class.calc(x)
        except: raise (f"Input {x} must be of type float")

    def send(self):
        """
        Отправка данных с выхода на линки
        """

        for link in self.link_output:
            link.send(self.output)
        return self

    def back_propagation_output_l(self, ref, speed):
        """
        Алгоритм обратного распространения ошибки для выходного уровня

        :param ref: тренировочный пример
        :type ref: float
        :param speed: скорость обучения
        :type speed: float
        """

        for link in self.link_input:
            link.back_propagation_output_l(ref, speed)
        return self

    def back_propagation_hidden_l(self, speed):
        """
        Алгоритм обратного распространения ошибки для скрытых уровней

        :param speed: скорость обучения
        :type speed: float
        """

        for link in self.link_input:
            link.back_propagation_hidden_l(speed)
        return self

    def back_propagation_apply(self):
        """
        Применение алгоритма обратного распространения ошибки к каждому линку
        """

        for link in self.link_output:
            link.apply_weight_delta()
        return self

    def import_(self, links_data):
        """
        Импорт сохраненных данных

        :param links_data: данные линков
        :type links_data: list[Link]
        """

        for i in range(len(links_data)):
            link = self.link_input[i]
            link.import_(links_data[i])

    def export(self):
        """
        Экспорт данных

        :return: список линков
        :rtype: list[Link]
        """

        res = []
        for link in self.link_input:
            res.append(link.export())
        return res

    def get_activation_derivative(self, x):
        """
        Метод получения производной функции активации

        :param x: значение на вход функции
        :type x: float
        :return: выход с функции производной
        :rtype: float
        """

        try:
            return self.activation_class.derivative(x)
        except TypeError: raise TypeError(f"Input {x} must be of type float")

    @staticmethod
    def get_loss(output, ref):
        """
        Метод поиска потерь

        :param output: выходное значение
        :type output: float
        :param ref: референс
        :type ref: float
        :return: потеря
        :rtype: float
        """

        if not(isinstance(output, float) or isinstance(ref, float)):
            raise TypeError(f"Output {output} and reference {ref} must be of type float")
        return 0.5 * (output-ref)**2

    def __repr__(self):
        """
        Служебный метод вывода значения

        :return: строка со значениями входа и выхода
        :rtype: str
        """
        return "[{:.4f} => {:.4f}]".format(self.input, self.output)

class TransparentNeuron(Neuron):
    """
    Класс 'прозрачного' нейрона
    """
    def activation(self):
        """
        Применение активационной функции
        """
        self.output = self.input
        return self

class BiasNeuron(Neuron):
    """
    Нейрон смещения
    """
    def __init__(self, activation_class):
        """
        Конструктор класса

        :param activation_class: тип активационной функции
        :type activation_class: ActivationBase
        """
        super().__init__(activation_class)
        self.output = 1

    def reset(self, input = None):
        """
        Сброс входа нейрона

        :param input: входные данные
        :type input: float
        """
        return self

class NeuronSoftmax(Neuron):
    """
    Нейрон с функцией Softmax
    """

    def __init__(self, activation_class = act.ActivationSoftmax):
        """
        Конструктор класса

        :param activation_class: тип активационной функции
        :type activation_class: ActivationSoftmax
        """

        super().__init__(activation_class)
        self.layer_outputs = None  # Для хранения выходов всего слоя
        self.softmax_output = 0  # Результат после применения softmax

    def activation(self):
        """
        Применение функции активации
        """

        # Softmax требует вычисления для всего слоя,
        # поэтому этот метод будет вызываться из LayerSoftmax
        try:
            self.output = self.get_activation(self.input)
            return self
        except TypeError: raise TypeError(f"Input {self.input} must be of type float")

    def set_softmax_output(self, value):
        """
        Устанавливает выход после применения softmax

        :param value: значение с выхода
        :type value: float
        """

        self.softmax_output = value
        self.output = value  # Также обновляем обычный output

    def get_activation(self, x):
        """
        Возвращает само значение

        :param x: значение на входе функции
        :type x: float
        :return: само значение
        :rtype: float
        """
        if not isinstance(x, float):
            raise TypeError(f"Input {x} must be of type float")

        return x

    def get_activation_derivative(self, x):
        """
        Расчет производной активационной функции

        :param x: значение на входе функции
        :type x: float
        :return: возвращает 1
        :rtype: int
        """
        # Производная будет обрабатываться на уровне слоя
        return 1

class Link:
    """
    Класс 'линка'

    :ivar n_from: нейрон, из которого получают данные
    :type n_from: Neuron
    :ivar n_to: нейрон, в который передаются данные
    :type n_to: Neuron
    :ivar weight: вес
    :type weight: float
    """

    def __init__(self, n_from, n_to, weight):
        """
        Инициализация объекта

        :param n_from: нейрон, из которого получают данные
        :type n_from: Neuron
        :param n_to: нейрон, в который передаются данные
        :type n_to: Neuron
        :param weight: вес
        :type weight: float
        """

        self.n_from = n_from
        self.n_to = n_to
        self.weight = weight
        self.weight_delta = 0
        self.weight_delta_param = 0

    def send(self, signal):
        """
        Метод передачи данных из одного нейрона в другой

        :param signal: данные для передачи
        """

        try:
            self.n_to.add(signal*self.weight)
            return self
        except TypeError: raise TypeError(f"Input {signal} must be of type float")

    def back_propagation_output_l(self, ref, speed = 1):
        """
        Метод обратного распространения ошибки для выходного слоя

        :param ref: пример
        :type ref: float
        :param speed: скорость обучения
        :type speed: float
        """

        if not isinstance(ref, float):
            raise TypeError(f"Reference {ref} must be of type float")
        if not isinstance(speed, float):
            raise TypeError(f"Speed {speed} must be of type float")

        dl_dy = self.n_to.get_activation(self.n_to.input) - ref
        dy_dz = self.n_to.get_activation_derivative(self.n_to.input)
        dz_dw = self.n_from.output

        self.weight_delta_param = dl_dy * dy_dz
        self.weight_delta = -dl_dy * dy_dz * dz_dw * speed

        return self

    def back_propagation_hidden_l(self, speed=1):
        """
        Метод обратного распространения ошибки для скрытых уровней

        :param speed: скорость обучения
        :type speed: float
        """

        self.weight_delta = 0
        self.weight_delta_param = 0
        dl_dy = 0

        if not isinstance(speed, float):
            raise TypeError(f"Speed {speed} must be of type float")

        for link in self.n_to.link_output:
            dz_dy = link.weight
            dl_dy += link.weight_delta_param * dz_dy

        dy_dz = self.n_to.get_activation_derivative(self.n_to.input)
        dz_dw = self.n_from.output

        self.weight_delta_param = dl_dy * dy_dz
        self.weight_delta = -dl_dy * dy_dz * dz_dw * speed

        return self

    def apply_weight_delta(self):
        """
        Применение изменения веса
        """

        self.weight += self.weight_delta
        self.weight_delta = 0
        self.weight_delta_param = 0
        return self

    def export(self):
        """
        Экспорт данных

        :return: вес линка
        :rtype: float
        """
        return self.weight

    def import_(self, value):
        """
        Импорт сохраненных данных

        :param value: сохраненное значение
        :type value: float
        """

        self.weight = value
        return self

class Layer:
    """
    Класс слоя

    :ivar neuron_class: тип нейронов в уровне
    :type neuron_class: Neuron
    :ivar size: количество нейронов в классе
    :type size: int
    :ivar activation_class: тип активационной функции нейронов
    :type activation_class: ActivationBase
    :ivar use_bias: ипользование нейрона смещения
    :type use_bias: bool
    """

    def __init__(self, neuron_class, size, activation_class, use_bias = False):
        """
        Конструктор класса

        :param neuron_class: тип нейронов в уровне
        :type neuron_class: Neuron
        :param size: количество нейронов в классе
        :type size: int
        :param activation_class: тип активационной функции нейронов
        :type activation_class: ActivationBase
        :param use_bias: ипользование нейрона смещения
        :type use_bias: bool
        """

        self.neurons = []
        self.bias = None
        self.activation_class = activation_class

        for _ in range(size):
            self.neurons.append(neuron_class(activation_class()))

        if use_bias:
            self.bias = BiasNeuron(act.ActivationTransparent)
            self.bias.input=1

            for neuron in self.neurons:
                link = Link(self.bias, neuron, random.uniform(-1, 1))

                self.bias.add_link_input(link)

                neuron.add_link_input(link)

    def reset(self, input = None):
        """
        Сброс входов всех нейронов

        :param input: данные на вход
        :type input: list
        """

        if input is None:
            for neuron in self.neurons:
                neuron.reset()
        else:
            if not isinstance(input, list):
                raise TypeError(f"Input {input} must be of type list")
            for i in range(len(self.neurons)):
                self.neurons[i].reset(input[i])

        return self

    def calc(self):
        """
        Применение активационной функции каждого нейрона
        """

        if self.bias is not None:
            self.bias.send()

        for neuron in self.neurons:
            neuron.activation()

        return self

    def get_loss(self, refs):
        """
        Подсчет потерь каждого нейрона

        :param refs: референсы
        :type refs: list
        :return: потери
        :rtype: float
        """

        loss = 0

        for i in range(len(self.neurons)):
            neuron = self.neurons[i]
            ref = refs[i]
            loss += neuron.get_loss(neuron.output, ref)

        return loss

    def send(self):
        """
        Отправка данных со всех нейронов слоя
        """

        for neuron in self.neurons:
            neuron.send()

        return self

    def back_propagation_output_l(self, refs, speed):
        """
        Расчет обратного распространения ошибки на уровне выходного слоя

        :param refs: референсы
        :type refs: list
        :param speed: скорость обучения
        :type speed: float
        """

        for i in range(len(self.neurons)):
            neuron = self.neurons[i]
            neuron.back_propagation_output_l(refs[i], speed)

        return self

    def back_propagation_hidden_l(self, speed):
        """
        Расчет обратного распространения ошибки на уровне скрытого слоя

        :param speed: скорость обучения
        :type speed: float
        """

        for i in range(len(self.neurons)):
            neuron = self.neurons[i]
            neuron.back_propagation_hidden_l(speed)

        return self

    def back_propagation_apply(self):
        """
        Применение обратного распространения ошибки
        """

        for neuron in self.neurons:
            neuron.back_propagation_apply()

        return self

    def export(self):
        """
        Экспорт данных с каждого нейрона
        :return: данные для сохранения
        :rtype: list
        """

        res = []
        for neuron in self.neurons:
            res.append(neuron.export())

        return res

    def import_(self, neurons_data):
        """
        Импорт данных в каждый нейрон

        :param neurons_data: сохраненные данные
        :type neurons_data: list
        """

        for i in range(len(self.neurons)):
            self.neurons[i].import_(neurons_data[i])

        return self

    def __len__(self):
        """
        Метод нахождения размера слоя

        :return: число нейронов в слое
        :rtype: int
        """

        return len(self.neurons)

    def __str__(self):
        """
        Метод преобразования в строку

        :return: строка со значениями нейронов
        :rtype: str
        """

        return f'Dense(neurons={self.neurons}, bias={self.bias})'

    def __repr__(self):
        """
        Служебный метод вывода данных слоя

        :return: строка с данными слоя
        :rtype: str
        """

        return ', '.join([repr(x) for x in self.neurons])

class LayerSoftmax(Layer):
    """
    Слой с функцией Softmax
    """

    def __init__(self, size):
        """
        Инициализация

        :param size: количество нейронов в слое
        :type size: int
        """

        super().__init__(NeuronSoftmax, size, act.ActivationTransparent, False)
        self.activation_class = act.ActivationSoftmax

    def calc(self):
        """
        Расчет функции активации для слоя
        """

        for neuron in self.neurons:
            neuron.output = neuron.input

        softmax_outputs = act.ActivationSoftmax.calc_layer([neuron.output for neuron in self.neurons])

        for i, neuron in enumerate(self.neurons):
            neuron.set_softmax_output(softmax_outputs[i])

        return self

    def back_propagation_output_l(self, refs, speed):
        """
        Расчет обратного распространения ошибки на уровне слоя

        :param refs: референсы
        :type refs: list
        :param speed: скорость обучения
        :type speed: float
        """

        for i in range(len(self.neurons)):
            neuron = self.neurons[i]
            ref = refs[i]

            grad = neuron.output - ref

            for link in neuron.link_input:
                dz_dw = link.n_from.output
                link.weight_delta = -grad * dz_dw * speed
                link.weight_delta_param = grad

        return self

    def get_loss(self, refs):
        """
        Метод расчета ошибки методом кросс-энтропии

        :param refs: референсы
        :type refs: list
        :return: значение ошибки
        :rtype: float
        """

        loss = 0
        epsilon = 1e-15  # Для численной стабильности

        for i in range(len(self.neurons)):
            neuron = self.neurons[i]
            ref = refs[i]

            # Кросс-энтропийная потеря для одного класса
            # L = -Σ y_i * log(ŷ_i)
            output_clipped = max(min(neuron.output, 1 - epsilon), epsilon)
            loss += -ref * math.log(output_clipped)

        return loss

class NeuralNetwork:
    """
    Класс нейронной сети

    :ivar layers: список слоев сети
    :type layers: list
    """

    def __init__(self):
        """
        Инициализация
        """
        self.layers = []

    def _add_layer(self, neuron_class, size, activation_class, random_radius, use_bias):
        """
        Метод добавления слоя в сеть

        :param neuron_class: класс нейронов в слое
        :type neuron_class: Neuron
        :param size: размер слоя
        :type size: int
        :param activation_class: активационная функция нейронов в слое
        :type activation_class: ActivationBase
        :param random_radius: радиус случайного начального распределения весов
        :type random_radius: float
        :param use_bias: использовать смещение
        :type use_bias: bool
        """

        if type(neuron_class) == NeuronSoftmax:
            layer = LayerSoftmax(size)
        else: layer = Layer(neuron_class, size, activation_class, use_bias)
        self.layers.append(layer)

        if len(self.layers) > 1:
            previous_layer = self.layers[-2]

            for n_from in previous_layer.neurons:
                for n_to in layer.neurons:
                    link = Link(n_from, n_to, random.uniform(-random_radius, random_radius))
                    n_from.add_link_output(link)
                    n_to.add_link_input(link)

        return self

    def add_layer(self, size, activation_class = act.ActivationSigmoid, random_radius = 0.5,  use_bias = False):
        """
        Добавление слоя в модель

        :param size: размер слоя
        :type size: int
        :param activation_class: активационная функция нейронов слоя
        :type activation_class: ActivationBase
        :param random_radius: радиус случайного распределения весов
        :type random_radius: float
        :param use_bias: использовать смещение
        :type use_bias: bool
        """
        if activation_class == act.ActivationSoftmax:
            # Для softmax используем специальный нейрон
            self._add_layer(NeuronSoftmax, size, activation_class, random_radius, use_bias)
        else:
            self._add_layer(Neuron, size, activation_class, random_radius, use_bias)
        return self

    def add_input_layer(self, size):
        """
        Добавление входного слоя

        :param size: размер слоя
        :type size: int
        """
        self._add_layer(TransparentNeuron, size, act.ActivationTransparent, 0, False)
        return self

    def get_output_layer(self):
        """
        Метод возвращает выходной слой модели

        :return: выходной слой
        :rtype: Layer
        """

        return self.layers[-1]

    def run(self, input):
        """
        Метод вычисления результатов

        :param input: данные на входе модели
        :type input: list
        :return: выходные значения с последнего слоя модели
        :rtype: list
        """

        if len(self.layers) == 0:
            raise Exception('EmptyNetwork')

        if isinstance(input, types.GeneratorType):
            input = next(input)

        if len(input) != len(self.layers[0]):
            raise Exception(f'IncorrectInput({len(self.layers[0])} signals required, {len(input)} given)')

        self.layers[0].reset(input)

        for i in range(len(self.layers) - 1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            next_layer.reset()
            current_layer.calc()
            current_layer.send()

        self.layers[-1].calc()

        return self

    def train(self, data, speed, verbose=False, use_cross_entropy=True, clip_value=5.0):
        """
        Метод обучения модели

        :param data: тренировочный датасет
        :type data: list
        :param speed: скорость обучения модели
        :type speed: float
        :param verbose: отображение подробной информации во время обучения
        :type verbose: bool
        :param use_cross_entropy: использовать кросс=энтропию в качестве критерия ошибки
        :type use_cross_entropy: bool
        :return: суммарная ошибка
        :rtype: float
        """

        i = 0
        loss_total = 0

        for input, refs in data:
            if isinstance(input, types.GeneratorType):
                input = next(input)

            if isinstance(refs, types.GeneratorType):
                refs = next(refs)

            self.run(input)

            output_layer = self.layers[-1]
            # Просто вызываем get_loss, который уже переопределен в LayerSoftmax
            loss_sum = output_layer.get_loss(refs)
            loss_total += loss_sum

            if verbose:
                print(f"item #{i}, loss: {loss_sum}")
                # Добавьте отладочную информацию:
                print(f"  Output: {[round(o, 3) for o in self.get_output()]}")
                print(f"  Target: {[round(r, 3) for r in refs]}")
                print(f"  Predicted: {self.get_best_index()}, Actual: {refs.index(1.0) if 1.0 in refs else 'N/A'}")

            output_layer.back_propagation_output_l(refs, speed)

            other_layers = self.layers[:-1][::-1]

            for layer in other_layers:
                layer.back_propagation_hidden_l(speed)

            if clip_value is not None:
                for layer in self.layers:
                    for neuron in layer.neurons:
                        for link in neuron.link_input:
                            if abs(link.weight_delta) > clip_value:
                                link.weight_delta = clip_value if link.weight_delta > 0 else -clip_value

            for layer in self.layers:
                layer.back_propagation_apply()

            i += 1

        return loss_total

    def export(self):
        """
        Функция экспортирования обученной модели

        :return: массив весов
        :rtype: list
        """

        res = []
        for layer in self.layers:
            res.append(layer.export())

        return res

    def import_(self, layers_data):
        """
        Функция импортирования сохраненной модели

        :param layers_data: сохраненные данные
        :type layers_data: list
        """

        if len(self.layers) != len(layers_data):
            raise ValueError(f"Несоответствие количества слоев: "
                             f"сеть имеет {len(self.layers)} слоев, "
                             f"данные содержат {len(layers_data)} слоев")

        for i in range(len(self.layers)):
            if len(self.layers[i]) != len(layers_data[i]):
                raise ValueError(f"Несоответствие размера слоя {i}: "
                                 f"сеть имеет {len(self.layers[i])} нейронов, "
                                 f"данные содержат {len(layers_data[i])} нейронов")

        # Импортируем веса
        for i in range(len(layers_data)):
            layer_data = layers_data[i]
            layer = self.layers[i]
            layer.import_(layer_data)

        return self

    def get_output(self):
        """
        Метод получения значений выходного слоя

        :return: список выходных значений
        :rtype: list
        """

        return [neuron.output for neuron in self.layers[-1].neurons]

    def get_best_index(self):
        """
        Метод получения большего индекса

        :return: индекс большего выхода
        :rtype: int
        """

        output = self.get_output()

        best_index = -1
        best_value = float('-inf')

        for i in range(len(output)):
            val = output[i]
            if val > best_value:
                best_index = i
                best_value = val

        return best_index

    def __str__(self):
        """
        Метод преобразования в строку

        :return: строка всех весов модели
        :rtype: str
        """

        return '\n'.join([str(x) for x in self.layers])

    def __repr__(self):
        """
        Служебный метод вывода данных

        :return: строка с данными модели
        :rtype: str
        """

        return "\n".join([repr(x) for x in self.layers])
