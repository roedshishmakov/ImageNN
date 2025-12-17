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
    def derviative(cls, x):
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