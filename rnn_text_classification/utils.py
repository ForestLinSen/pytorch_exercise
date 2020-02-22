import numpy as np
import torch
import torchvision

# Create vocabulary

# Goal: {word1: index1, word2: index2}
def create_vocab(file_path):
    vocab = {}
    all_word = []
    with open(file_path, 'r', encoding = 'UTF-8') as f:
        # Split words
        for line in f:
            # Remove whitespace in the beginning and end
            clean_line = line.strip()
            content = clean_line.split("\t")[0]

            # This method: whenever we meet a new word, we set value of it as 0+1=1; and then for every time we
            # encounter this word, we add 1 to the value of it
            # which means the value of a word is propositional to its frequency
            # Pros: contain information of both frequency and index
            # Cons: two words may have same frequency and index may be too bigger for some word
            #for word in content:
                #vocab[word] = vocab.get(word, 0) + 1

            # This method: using enumerate to keep the index of char smaller
            for char in content:
                all_word.append(char)

        all_word = set(all_word)
        for i, word in enumerate(all_word):
            vocab[word] = i
    return vocab

# Goal: [ [word5, word9, word2, word5], label = 5 ]
def load_dataset(file):
    vocab = create_vocab(file)
    dataset = []
    with open(file, 'r', encoding = 'UTF-8') as f:
        for line in f:
            line = line.strip()
            content = line.split('\t')[0]
            index = line.split('\t')[1]
            dataset.append([[vocab[word_num] for word_num in content], int(index)])

    x = np.zeros(shape = (len(dataset), 40), dtype = int)
    y = np.zeros(shape = (len(dataset), 1), dtype = int)
    for i in range(len(dataset)):
        if len(dataset[i][0]) > 40:
            x[i, :40] = dataset[i][0]
        else:
            x[i, :len(dataset[i][0])] = dataset[i][0]
        y[i] = dataset[i][1]
    return vocab, x, y


def return_loader():

    path = "data/train.txt"
    vocab, x, y = load_dataset(path)
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = 5, shuffle = True)

    return loader


# if __name__ == "__main__":
#     path = "data/train.txt"
#
#     loader = return_loader()
#
#     for step, (x, y) in enumerate(loader):
#         print("X: ", x, "\nY: ", y)
#

