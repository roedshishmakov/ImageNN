from abc import abstractmethod
import math

class ActivationBase:
    """
    Родительский класс
    """
    @abstractmethod
    def calc(cls, x):
        pass

    @abstractmethod
    def derivative(cls, x):
        pass

class ActivationTransparent(ActivationBase):
    """
    Прозрачная функция активации. На выходе то же значение, что и на входе.
    """
    @classmethod
    def calc(cls, x):
        """
        Метод расчета 'прозрачной' активационной функции

        :param x: входное значение
        :type x: float
        :return: выходное значение
        :rtype: float
        """
        if not isinstance(x, float):
            raise TypeError('X must be float')
        else:
            return x

    @classmethod
    def derivative(cls, x):
        """
        Метод получения значения производной 'прозрачной' функции

        :param x: входное значение
        :type x: int
        :return: выходное значение
        :rtype: int
        """

        return 1

class ActivationRelu(ActivationBase):
    """
    Функция ReLu: y = max(0, x)
    """

    @classmethod
    def calc(cls, x):
        """
        Метод расчета функции ReLu

        :param x: входное значение
        :type x: float
        :return: выходное значение
        :rtype: float
        """

        if not isinstance(x, float):
            raise TypeError('X must be float')
        else:
            return max(0, x)

    @classmethod
    def derivative(cls, x):
        """
        Метод получения значения производной функции ReLu

        :param x: входное значение
        :type x: int
        :return: выходное значение
        :rtype: int
        """

        if not isinstance(x, float):
            raise TypeError('X must be float')
        return 0 if x < 0 else 1

class ActivationSigmoid(ActivationBase):
    """
    Сигмоидальная функция активации
    """

    @classmethod
    def calc(cls, x):
        """
        Метод расчета функции Sigmoid

        :param x: входное значение
        :type x: float
        :return: выходное значение
        :rtype: float
        """

        if not isinstance(x, float):
            raise TypeError('X must be float')

        return 1 / (math.exp(-x) + 1)

    @classmethod
    def derivative(cls, x):
        """
        Метод получения значения производной функции Sigmoid

        :param x: входное значение
        :type x: int
        :return: выходное значение
        :rtype: int
        """

        if not isinstance(x, float):
            raise TypeError('X must be float')

        return cls.calc(x) * (1 - cls.calc(x))

class ActivationSoftmax(ActivationBase):
    @classmethod
    def calc(cls, x):
        """
        Функция для обработки исключения в случае вызова calc

        :param x: входное значение
        :raises NotImplementedError: если метод calc использован для функции Softmax
        """

        raise NotImplementedError("Softmax требует вычисления для всего слоя")

    @classmethod
    def derivative(cls, x):
        """
        Метод для обработки исключения в случае вызова метода derivative

        :param x: входное значение
        :raises NotImplementedError: в случае, если вызван метод derivative к функции Softmax
        """

        raise NotImplementedError("Softmax требует специальной обработки производной")

    @classmethod
    def calc_layer(cls, layer_outputs):
        """
        Метод расчета функции Softmax для всего слоя

        :param layer_outputs: данные с выходного слоя
        :type layer_outputs: list
        :return: выходные значения
        :rtype: list
        """

        if not isinstance(layer_outputs, list):
            raise TypeError('layer_outputs must be list')

        max_val = max(layer_outputs)

        if max_val > 100:
            scale_factor = max_val / 100
            layer_outputs = [x / scale_factor for x in layer_outputs]
            max_val = max(layer_outputs)

        exp_values = [math.exp(x - max_val) for x in layer_outputs]
        exp_sum = sum(exp_values)

        if exp_sum == 0:
            return [1.0 / len(layer_outputs) for _ in layer_outputs]

        softmax_outputs = [exp_val / exp_sum for exp_val in exp_values]
        return softmax_outputs
