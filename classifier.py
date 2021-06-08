import argparse
import string
import numpy as np
import csv

from network import *

# max word length
maxlen = 21
# LSTM time step (step letters each pass)
step = 1

def train(model, input, output):
    model.fit(input, output, epochs = 10, batch_size = 16)

    model.save("gender_classifier")
    print("saved model")

def evaluate(model, input, output):
    model.evaluate(input, output, batch_size = 4)

def prepare(s):
    s = s.replace("-", "")
    chars = string.ascii_lowercase + "'"
    try:
        a = []
        sequence = []
        last_i = 0
        for i, letter in zip(range(len(s)), s):
            last_i = i
            sequence.append(chars.index(letter) + 1)
            if len(sequence) == step:
                a.append(sequence)
                sequence = []
            if len(sequence) != 0 and len(sequence) < step and i == len(s) - 1:
                sequence.extend([0] * (step - len(sequence)))
                a.append(sequence)

        if len(a) < (maxlen / step):
            n = int(np.floor((maxlen / step))) - len(a)
            for _ in range(n):
                a.append([0] * step)
        return np.array(a)

    except Exception:
        import traceback
        traceback.print_exc()
        print("error processing word: {}".format(s))

def get_data(file):
    with open(file, "r") as csvfile:
        reader = csv.reader((l.replace("\0", "") for l in csvfile), delimiter=";", quotechar="|")
        nouns = []
        gender = []
        for row in reader:
            nouns.append(prepare(row[0]))
            gender.append([1, 0] if row[1] == "M" else [0, 1])
    p = np.random.permutation(len(nouns))
    return np.array(nouns)[p], np.array(gender)[p]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Italian gender classifer for nouns")
    parser.add_argument("--dataset", dest = "dataset", help="path to the italian nouns file", default="dataset/nomi_italiani_gendered.csv", type=str)
    parser.add_argument("--train", dest = "train", help="train", default="False", action="store_true")
    args = parser.parse_args()
    limit_memory()
    nouns, gender = get_data(args.dataset)
    print("loaded csv {}".format(args.dataset))

    try:
        model = load_model("gender_classifier")
        print("loaded model")
    except Exception:
        model = build_model(step = step)
        print("created model")

    if args.train == True:
        train(model, nouns[:-100], gender[:-100])

    evaluate(model, nouns[-100:], gender[-100:])

    while True:
        t = input("test noun - ctrl+c to close -> ")
        pred = model.predict(np.expand_dims(prepare(t), axis = 0))
        id = np.argmax(pred)
        print(["M", "F"][id], pred)
