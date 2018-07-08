# coding=utf-8

from model.embeds.embeds import load_word2vec_embeddings

# Verify that embeddings are loaded properly
embeddings, index2word, word2index = load_word2vec_embeddings("GoogleNews-vectors-negative300.bin.gz")
print(embeddings.weight.shape)
print(type(index2word), type(word2index))
print(len(index2word), len(word2index))
for i, index in enumerate(index2word):
    if i > 3:
        break
    print(index, index2word[index])
for i, word in enumerate(word2index):
    if i > 3:
        break
    print(word, word2index[word])