# Named Entity Recognition

This repository implements a Named Entity Recognition system for the Turkish Language. The code will probably work for the other languages as well if the input data format criterion matches.

The model's architecture is Bi-LSTM + CRF. FastText embeddings are used as a pre-trained word embeddings. The code also supports the variable size input.

# Examples
```python
# For the complete example, please see the test.py
sentence = (
    "Aziz Nesin 'in yazmış olduğu Nesin Yayınevi tarafından basılan ' Bir Sürgünün Anıları ' "
    "isimli kitap Nesin 'in sürgün yıllarındaki Bursa anılarını anlatıyor ."
)
# We assume that the sentence is already tokenized.
sentence_tokens = sentence.split()
score, tags = predict_sentence(model, sentence_tokens, w2i, idx2tag)

for token, tag in zip(sentence_tokens, tags):
    print("{:<15}{:<5}".format(token, tag))

# Output:
# Aziz           B-PER
# Nesin          I-PER
# 'in            O
# yazmış         O
# olduğu         O
# Nesin          B-ORG
# Yayınevi       I-ORG
# tarafından     O
# basılan        O
# '              O
# Bir            O
# Sürgünün       O
# Anıları        O
# '              O
# isimli         O
# kitap          O
# Nesin          B-PER
# 'in            O
# sürgün         O
# yıllarındaki   O
# Bursa          B-LOC
# anılarını      O
# anlatıyor      O
# .              O
```


# Data Format

Sentences and the tags should be in separate files. Expected format is:

train_words:

    sent1_token1 sent1_token2 sent1_token3 ...
    sent2_token1 sent_2token2 sent2_token ...
    ...

train_tags:

    sent1_tag1 sent1_tag2 sent1_tag3 ...
    sent2_tag1 sent2_tag2 sent2_tag3 ...
    ...



# Train
```
python train.py -h
-h, --help            show this help message and exit
--train_data TRAIN_DATA [TRAIN_DATA ...]
                    Training data
--valid_data VALID_DATA [VALID_DATA ...]
                    Validation Data
--w2v_file W2V_FILE   Pre-trained Word Embeddings
--hidden_dim HIDDEN_DIM
                    Hidden dimension for the RNN
--num_layers NUM_LAYERS
                    Number of RNN Layers to use
--bidirectional       Option to make the RNNs bidirectional
--dropout_p DROPOUT_P
                    Dropout probability for the embedding layer
--device DEVICE       Device to run the model
--n_epochs N_EPOCHS   Number of epochs to train the model
--model_name MODEL_NAME
                    Model name to save

```

```bash
python train.py --train_data train_words train_tags --valid_data valid_words valid_tags 
--w2v_file <w2v_file> --hidden_dim 64 --num_layers 2 --bidirectional 
--dropout_p 0.3 --n_epochs 10 --device "cuda"
```


# Test



```bash
python test.py -h
usage: test.py [-h] --model_file MODEL_FILE --w2i_file W2I_FILE

Named Entity Recognition Testing

optional arguments:
  -h, --help            show this help message and exit
  --model_file MODEL_FILE
                        Trained model file
  --w2i_file W2I_FILE   Word2Index Vocabulary
```


# Results

<!-- ```
              precision    recall  f1-score   support

       B-LOC       0.95      0.95      0.95      1066
       B-ORG       0.95      0.90      0.92       744
       B-PER       0.96      0.94      0.95      1347
       I-LOC       0.89      0.92      0.90       194
       I-ORG       0.92      0.94      0.93       545
       I-PER       0.95      0.96      0.95       654
           O       1.00      1.00      1.00     38731

    accuracy                           0.99     43281
   macro avg       0.94      0.94      0.94     43281
weighted avg       0.99      0.99      0.99     43281
``` -->

```
              precision    recall  f1-score   support

       B-LOC       0.95      0.95      0.95      1066
       B-ORG       0.90      0.95      0.93       744
       B-PER       0.95      0.94      0.95      1347
       I-LOC       0.94      0.88      0.91       194
       I-ORG       0.88      0.97      0.92       545
       I-PER       0.95      0.96      0.95       654
           O       1.00      0.99      1.00     38731

    accuracy                           0.99     43281
   macro avg       0.94      0.95      0.94     43281
weighted avg       0.99      0.99      0.99     43281

```
