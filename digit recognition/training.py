"""Обучение сети до достижения нужного процента верных ответов."""
from preprocessing import shuffled


def accuracy(network, images, expected):
    ok = sum(network.recognize(img) == bool(e) for img, e in zip(images, expected))
    return 100 * ok / len(images)


def train(network, images, is_target, positive, negative, study_coeff,
          target_accuracy=95.0, max_epochs=50):
    """Каждая эпоха - вся обучающая выборка в случайном порядке.

    studyCoeff уменьшается с номером эпохи (иначе веса «прыгают» вокруг решения),
    лучшие найденные веса запоминаются и возвращаются в сеть в конце.
    """
    best_acc, best_weights = 0.0, network.weights()
    for epoch in range(1, max_epochs + 1):
        coeff = study_coeff / epoch
        for i in shuffled(positive, negative):
            network.update(images[i], is_target[i], coeff)
        acc = accuracy(network, images, is_target)
        print(f"Эпоха {epoch}: верно распознано {acc:.2f}%")
        if acc > best_acc:
            best_acc, best_weights = acc, network.weights()
        if acc >= target_accuracy:
            break
    network.set_weights(best_weights)
    return best_acc
