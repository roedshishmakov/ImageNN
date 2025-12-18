import pytest
import math

import activations as act

from main import Neuron, Link, Layer, NeuralNetwork, LayerSoftmax, NeuronSoftmax

#Neuron
def test_neuron_initialization():
    neuron = Neuron(act.ActivationRelu)
    assert neuron.input == 0
    assert neuron.output == 0
    assert neuron.link_input == []
    assert neuron.link_output == []
    assert neuron.activation_class == act.ActivationRelu

def test_neuron_add_positive():
    neuron = Neuron(act.ActivationTransparent)
    neuron.add(2.0)
    assert neuron.input == 2.0
    neuron.add(3.0)
    assert neuron.input == 5.0

def test_neuron_add_negative():
    neuron = Neuron(act.ActivationTransparent)
    with pytest.raises(TypeError):
        neuron.add(5)  # int вместо float
    with pytest.raises(TypeError):
        neuron.add("string")

def test_neuron_reset():
    neuron = Neuron(act.ActivationTransparent)
    neuron.add(5.0)
    neuron.reset(10)
    assert neuron.input == 10

def test_neuron_activation_relu():
    neuron = Neuron(act.ActivationRelu)
    neuron.add(3.0)
    neuron.activation()
    assert neuron.output == 3.0

    neuron.reset()
    neuron.add(-3.0)
    neuron.activation()
    assert neuron.output == 0.0

def test_neuron_get_loss():
    assert Neuron.get_loss(1.0, 0.5) == pytest.approx(0.125, rel=1e-6)
    assert Neuron.get_loss(0.0, 0.0) == 0.0
    assert Neuron.get_loss(2.0, 1.0) == 0.5

def test_neuron_get_loss_negative():
    with pytest.raises(TypeError):
        Neuron.get_loss(1.0, "string")

def test_neuron_get_activation_derivative():
    neuron = Neuron(act.ActivationRelu)
    assert neuron.get_activation_derivative(2.0) == 1
    assert neuron.get_activation_derivative(-2.0) == 0

#Neuron Softmax
def test_softmax_neuron_initialization():
    neuron = NeuronSoftmax()
    assert neuron.activation_class == act.ActivationSoftmax
    assert neuron.layer_outputs is None
    assert neuron.softmax_output == 0
    assert neuron.output == 0

def test_softmax_neuron_output():
    neuron = NeuronSoftmax()
    neuron.set_softmax_output(0.7)
    assert neuron.softmax_output == 0.7
    assert neuron.output == 0.7

def test_softmax_neuron_get_activation():
    neuron = NeuronSoftmax()
    assert neuron.get_activation(5.0) == 5.0

def test_softmax_neuron_get_activation_negative():
    neuron = NeuronSoftmax()
    with pytest.raises(TypeError):
        neuron.get_activation(5)

def test_softmax_neuron_get_activation_derivative():
    neuron = NeuronSoftmax()
    assert neuron.get_activation_derivative(2.0) == 1
    assert neuron.get_activation_derivative(-3.0) == 1

#Link
def test_link_initialization():
    n1 = Neuron(act.ActivationTransparent)
    n2 = Neuron(act.ActivationTransparent)
    link = Link(n1, n2, 0.5)

    assert link.n_from == n1
    assert link.n_to == n2
    assert link.weight == 0.5
    assert link.weight_delta == 0

def test_link_send_positive():
    n1 = Neuron(act.ActivationTransparent)
    n2 = Neuron(act.ActivationTransparent)
    link = Link(n1, n2, 2.0)

    n1.output = 3.0
    link.send(3.0)
    assert n2.input == 6.0

def test_link_send_negative():
    n1 = Neuron(act.ActivationTransparent)
    n2 = Neuron(act.ActivationTransparent)
    link = Link(n1, n2, 0.5)

    with pytest.raises(TypeError):
        link.send("string")

def test_link_delta():
    n1 = Neuron(act.ActivationTransparent)
    n2 = Neuron(act.ActivationTransparent)
    link = Link(n1, n2, 1.0)

    link.weight_delta = 0.2
    link.apply_weight_delta()
    assert link.weight == 1.2
    assert link.weight_delta == 0

#Layer
def test_layer_initialization():
    layer = Layer(Neuron, 3, act.ActivationRelu, use_bias=False)
    assert len(layer.neurons) == 3
    assert layer.bias is None
    assert all(isinstance(n, Neuron) for n in layer.neurons)

def test_layer_bias():
    layer = Layer(Neuron, 2, act.ActivationSigmoid, use_bias=True)
    assert layer.bias is not None
    assert len(layer.bias.link_input) == 2

def test_layer_reset_positive():
    layer = Layer(Neuron, 2, act.ActivationTransparent, False)
    layer.neurons[0].add(5.0)
    layer.neurons[1].add(3.0)

    layer.reset([0, 0])
    assert layer.neurons[0].input == 0
    assert layer.neurons[1].input == 0

def test_layer_reset_negative():
    layer = Layer(Neuron, 2, act.ActivationTransparent, False)
    with pytest.raises(TypeError):
        layer.reset("not a list")

def test_layer_calc():
    layer = Layer(Neuron, 2, act.ActivationTransparent, False)
    layer.reset([2, 3.0])
    layer.calc()

    assert layer.neurons[0].input == 2.0
    assert layer.neurons[1].input == 3.0

def test_layer_get_loss():
    layer = Layer(Neuron, 2, act.ActivationTransparent, False)
    layer.reset([1.0, 2.0])
    layer.calc()

    loss = layer.get_loss([0.5, 1.5])
    expected = 0.5 * ((1.0 - 0.5) ** 2 + (2.0 - 1.5) ** 2)
    assert loss == pytest.approx(expected, rel=1e-6)

#Softmax layer

def test_softmax_layer_initialization():
    layer = LayerSoftmax(3)
    assert len(layer.neurons) == 3
    assert all(isinstance(n, NeuronSoftmax) for n in layer.neurons)
    assert layer.bias is None

def test_softmax_layer_calc():
    layer = LayerSoftmax(3)

    layer.neurons[0].input = 1.0
    layer.neurons[1].input = 2.0
    layer.neurons[2].input = 3.0

    layer.calc()

    outputs = [n.output for n in layer.neurons]
    assert sum(outputs) == pytest.approx(1.0, rel=1e-6)
    assert all(o > 0 for o in outputs)
    assert outputs[2] > outputs[1] > outputs[0]

def test_softmax_layer_get_loss():
    layer = LayerSoftmax(3)

    for i, neuron in enumerate(layer.neurons):
        neuron.output = [0.1, 0.7, 0.2][i]

    refs = [0.0, 1.0, 0.0]
    loss = layer.get_loss(refs)

    assert loss == pytest.approx(-math.log(0.7), rel=1e-6)

#NeuralNetwork
def test_NN_initialization():
    nn = NeuralNetwork()
    assert nn.layers == []

    nn.add_input_layer(2)
    nn.add_layer(3, act.ActivationRelu)
    nn.add_softmax_layer(2)

    assert len(nn.layers) == 3
    assert len(nn.layers[0]) == 2
    assert len(nn.layers[1]) == 3
    assert len(nn.layers[2]) == 2

def test_NN_run_positive():
    nn = NeuralNetwork()
    nn.add_input_layer(2)
    nn.add_layer(1, act.ActivationTransparent, random_radius=0.0)

    link = nn.layers[0].neurons[0].link_output[0]
    link.weight = 0.5

    result = nn.run([1.0, 2.0])
    outputs = nn.get_output()

    assert len(outputs) == 1

def test_NN_run_negative():
    nn = NeuralNetwork()
    nn.add_input_layer(2)

    with pytest.raises(Exception) as e:
        nn.run([1.0])  # Неправильный размер входа
    assert "IncorrectInput" in str(e.value)

def test_NN_empty():
    nn = NeuralNetwork()
    with pytest.raises(Exception) as e:
        nn.run([1.0, 2.0])
    assert "EmptyNetwork" in str(e.value)

def test_NN_get_output():
    nn = NeuralNetwork()
    nn.add_input_layer(1)
    nn.add_layer(2, act.ActivationTransparent, random_radius=0.0)

    nn.run([1.0])
    outputs = nn.get_output()

    assert len(outputs) == 2
    assert all(isinstance(o, float) for o in outputs)

def test_NN_get_best_index():
    nn = NeuralNetwork()
    nn.add_input_layer(1)
    nn.add_layer(3, act.ActivationTransparent, random_radius=0.0)

    # Симулируем выходы
    nn.layers[1].neurons[0].output = 0.1
    nn.layers[1].neurons[1].output = 0.7
    nn.layers[1].neurons[2].output = 0.2

    assert nn.get_best_index() == 1