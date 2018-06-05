# AdverserialAttentionModel

Machine Comprehension for Marco using adversarial training examples. Will add support for SQuAD in the future.

## Requirements

1. Tested on Python 3.5 and Tensorflow 1.8
2. NLTK (punkt)
3. Extracted Marco train and dev json files in the datasets folder (or place them anywhere on your filesystem and change corresponding paths in config.yml)
4. Word Embedding file in Glove format in the Embedding folder (or elsewhere, with paths changed similarly)

## Preprocessing

Set hyperparams in config file and run

```
python3 preprocess.py
```

## Training

```
python3 main.py
```