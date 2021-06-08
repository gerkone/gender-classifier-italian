# Italian gender classifer for nouns

## Premise
This is a simple experiment done to end a discussion between me and a friend of mine. The reasoning was to find a functional relation between the syntax of a word and its gender in the Italian language to prove that the gender is tied to the word structure.

It's a couple hours effort, so don't expect very high quality.

## Usage
First install the python requirements with
```
pip install -r requirements.txt
```

Then you can start the training with
```
python classifier.py --train
```

If you simply want to try it out with your words check [this notebook](https://cocalc.com/projects/8c38a9b8-86fa-43d7-81e1-efb5add01b9b/files/gender-classifier-italian.ipynb).

## Dataset
The dataset was created using [this](http://badip.uni-graz.at/it/) corpus.

The nouns were later gendered using [dizionario-italiano](https://www.dizionario-italiano.it/).

It is made up of 6139 italian nouns ([file](dataset/nomi_italiani_gendered.csv)).

## Results
With just this simple LSTM network and small dataset the classifier is able to achive around 80%-85% accuracy on new (real) nouns. It still gets some basic words wrong (eg "mano", which is feminine, is recognized as mostly masculine)

You can also try to get the gender of imaginary words.
