"""Сеть: по нейрону на каждый пиксель изображения."""
from neuron import Neuron


class Network:
    def __init__(self, size=784, correct_coeff=0.5, weights=None):
        self.correct_coeff = correct_coeff  # порог распознавания
        # последний нейрон - смещение (вход всегда 1), без него сеть не может подстроить порог
        self.neurons = ([Neuron() for _ in range(size + 1)] if weights is None
                        else [Neuron(w) for w in weights])

    def set_weights(self, weights):
        for neuron, w in zip(self.neurons, weights):
            neuron.weight = w

    def weights(self):
        return [n.weight for n in self.neurons]

    def calculate(self, image):
        """Сумма выходов нейронов; нулевые пиксели дают 0, поэтому берём только единичные."""
        return (sum(self.neurons[i].calculate(1) for i in image.nonzero()[0])
                + self.neurons[-1].calculate(1))

    def recognize(self, image):
        return self.calculate(image) >= self.correct_coeff

    def update(self, image, expected, study_coeff):
        """Правило перцептрона: веса активных пикселей сдвигаются в сторону эталона."""
        error = int(expected) - int(self.recognize(image))
        if error:
            for i in image.nonzero()[0]:
                self.neurons[i].update(study_coeff * error)
            self.neurons[-1].update(study_coeff * error)
        return error == 0
