import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import keras
import random
import json

# токенизатор для nltk.word_tokenize (скачивается один раз)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

stemmer = LancasterStemmer()

with open('intents.json', encoding='utf-8') as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []


def stud():
    global words
    global labels
    global docs_x
    global docs_y
    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent['tag'] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    # та же архитектура, что была в tflearn: 8-8-softmax, Adam, categorical crossentropy
    model = keras.Sequential([
        keras.Input(shape=(len(training[0]),)),
        keras.layers.Dense(8),
        keras.layers.Dense(8),
        keras.layers.Dense(len(output[0]), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    print("Training the model (1500 epochs), please wait...")
    model.fit(training, output, epochs=1500, batch_size=8, verbose=0)
    print("Training finished.")
    model.save("model.keras")
    return model

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)


def chat():
    model = stud()
    print("The bot is ready to talk!!(Type 'quit' to exit)")
    while True:
        inp = input("\nYou: ")
        if inp.lower() == 'quit':
            break

        results = model.predict(numpy.array([bag_of_words(inp, words)]), verbose=0)[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if results[results_index] < 0.7:
            print("Bot: Sorry, I didn't understand that.")
            continue

        for tg in data['intents']:
            if tg['tag'] == tag:
                responses = tg['responses']
        print(f"Bot ({tag}, {results[results_index]:.2f}): " + random.choice(responses))


if __name__ == "__main__":
    chat()
