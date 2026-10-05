import random

from network import Network
from preprocessing import binarize, split_by_class
from storage import TEST_PATH, TRAIN_PATH, load_dataset, save_weights
from training import accuracy, train

TARGET = 8            # изучаемый класс (номер команды)
STUDY_COEFF = 0.01    # скорость изменения весов
CORRECT_COEFF = 0.5   # порог распознавания
TARGET_ACCURACY = 95.0
SEED = 1
EXAMPLES = 10


def main():
    random.seed(SEED)
    labels, pixels = load_dataset(TRAIN_PATH)
    images = binarize(pixels)
    is_target = labels == TARGET
    positive, negative = split_by_class(labels, TARGET)
    print(f"Изучаемый класс {TARGET}: {len(positive)} объектов, остальных: {len(negative)}")

    network = Network(images.shape[1], CORRECT_COEFF)
    print("=== Обучение ===")
    best = train(network, images, is_target, positive, negative, STUDY_COEFF, TARGET_ACCURACY)
    save_weights(network.weights(), {"target": TARGET, "correct_coeff": CORRECT_COEFF,
                                     "study_coeff": STUDY_COEFF, "train_accuracy": round(best, 2)})

    print("\n=== Проверка на тестовой выборке ===")
    test_labels, test_pixels = load_dataset(TEST_PATH)
    test_images = binarize(test_pixels)
    test_target = test_labels == TARGET
    print(f"Точность: {accuracy(network, test_images, test_target):.2f}%")

    print("\n=== Примеры ===")
    for i in random.sample(range(len(test_images)), EXAMPLES):
        answer = "относится" if network.recognize(test_images[i]) else "не относится"
        truth = "относится" if test_target[i] else "не относится"
        print(f"Изображение {i} (класс {test_labels[i]}): сеть - {answer}, на самом деле - {truth}")


if __name__ == "__main__":
    main()
