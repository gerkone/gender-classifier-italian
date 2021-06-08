import italian_dictionary
import argparse
import time
import csv
from tqdm import tqdm

def get_gender(nouns):
    nouns_gendered = {}
    for noun, i in zip(nouns, tqdm(range(len(nouns)))):
        try:
            definition = italian_dictionary.get_definition(noun)
            nouns_gendered[noun] = "M" if "sostantivo maschile" in definition["grammatica"] else "F"
        except Exception:
            pass
        time.sleep(0.3)

    return nouns_gendered

def get_nouns(file):
    with open(file, "r") as csvfile:
        reader = csv.reader((l.replace("\0", "") for l in csvfile), delimiter=";", quotechar="|")
        nouns = []
        for row in reader:
            if row[4] == "\"S\"" and len(row[1]) > 3:
                nouns.append(row[1].replace("\"", ""))
    return nouns

def save_dataset(dataset_out, nouns_gendered):
    with open(dataset_out, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=";",quotechar="|", quoting=csv.QUOTE_MINIMAL)
        for l in nouns_gendered.keys():
            writer.writerow([l, nouns_gendered[l]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Italian gender classifer for nouns")
    parser.add_argument("--dataset-input", dest = "dataset_in", help="path to the italian nouns file", default="dataset/lemmi-export.csv", type=str)
    parser.add_argument("--dataset-output", dest = "dataset_out", help="path to the italian nouns file", default="dataset/nomi_italiani_gendered.csv", type=str)
    args = parser.parse_args()

    nouns_list = get_nouns(args.dataset_in)
    longest = max(nouns_list, key=len)
    print("loaded nouns from csv file: {}. longest word: {} with {} chars".format(args.dataset_in,  longest, len(longest)))
    nouns_gendered = get_gender(nouns_list)
    print("gendered {} nouns".format(len(nouns_gendered)))
    save_dataset(args.dataset_out, nouns_gendered)
    print("saved csv file: {}".format(args.dataset_out))
    print("all done")
