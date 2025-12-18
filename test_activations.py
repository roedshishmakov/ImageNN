import pytest
import activations as act

#Transparent
def test_trans_calc_positive():
    assert act.ActivationTransparent.calc(5.0) == 5.0
    assert act.ActivationTransparent.calc(-3.14) == -3.14
    assert act.ActivationTransparent.calc(0.0) == 0.0

def test_trans_calc_negative():
    with pytest.raises(TypeError):
        act.ActivationTransparent.calc('15')

def test_trans_derivative():
    assert act.ActivationTransparent.derivative(5.0) == 1
    assert act.ActivationTransparent.derivative(-3.14) == 1
    assert act.ActivationTransparent.derivative(0.0) == 1

#ReLu
def test_relu_calc_positive():
    assert act.ActivationRelu.calc(3.0) == 3.0
    assert act.ActivationRelu.calc(0.0) == 0.0
    assert act.ActivationRelu.calc(-2.0) == 0.0

def test_relu_calc_negative():
    with pytest.raises(TypeError):
        act.ActivationRelu.calc('15')

def test_relu_derivative_positive():
    assert act.ActivationRelu.derivative(5.0) == 1
    assert act.ActivationRelu.derivative(-3.0) == 0
    assert act.ActivationRelu.derivative(0.0) == 1

def test_relu_derivative_negative():
    with pytest.raises(TypeError):
        act.ActivationRelu.derivative('15')

#Softmax
def test_softmax_calc_negative():
    with pytest.raises(NotImplementedError):
        act.ActivationSoftmax.calc(1.0)

def test_softmax_derivative_negative():
    with pytest.raises(NotImplementedError):
        act.ActivationSoftmax.derivative(1.0)

def test_softmax_calc_layer_positive():
    outputs = [1.0, 2.0, 3.0]
    result = act.ActivationSoftmax.calc_layer(outputs)

    assert len(result) == 3
    assert sum(result) == pytest.approx(1.0, rel=1e-6)
    assert all(r > 0 for r in result)

    max_index = outputs.index(max(outputs))
    assert result[max_index] == max(result)

def test_softmax_calc_layer_negative():
    with pytest.raises(TypeError):
        act.ActivationSoftmax.calc_layer(5)
    with pytest.raises(TypeError):
        act.ActivationSoftmax.calc_layer("string")