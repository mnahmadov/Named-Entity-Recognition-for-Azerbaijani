import pickle

from utils import load_wv

w2v_model = load_wv("./cc.az.300.vec") # FastText model path
index2word = ["<pad>", "<unk>"] + w2v_model.index2word
word2index = {word: index for index, word in enumerate(index2word)}

with open("/home/frasulov/Desktop/ner/word2index.pkl", "wb") as f:
    pickle.dump(word2index, f)
