class NeuralNetworkError(Exception):
    pass

class ArgumentError(NeuralNetworkError):
    def __init__(self, message="Invalid arguments provided"):
        self.message = message
        super().__init__(self.message)

class IncorrectCommand(NeuralNetworkError):
    def __init__(self, message="Command is incorrect"):
        self.message = message
        super().__init__(self.message)

class PathError(NeuralNetworkError):
    def __init__(self, message="Path error"):
        self.message = message
        super().__init__(self.message)

class EpochError(NeuralNetworkError):
    def __init__(self, message="Invalid number of epochs"):
        self.message = message
        super().__init__(self.message)

class ValidationError(NeuralNetworkError):
    def __init__(self, message="Validation error"):
        self.message = message
        super().__init__(self.message)