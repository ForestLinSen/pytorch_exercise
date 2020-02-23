import torch
import torch.nn as nn
import numpy as np

# shape = (4762,300)
embedding_pretrained = torch.tensor(np.load('data/embedding_SougouNews.npz')['embeddings'].astype('float32'))   #Load arrays or pickled objects from ``.npy``, ``.npz`` or pickled files.
num_words = embedding_pretrained.shape[0] # 4762
embedding_dim = embedding_pretrained.shape[1] #300
hidden_size = 128
num_classes = 10
path = "data/train.txt"


class Model(nn.Module):
    def __init__(self, pre_trained = True):
        super(Model, self).__init__()

        if pre_trained:
            self.embedding = nn.Embedding.from_pretrained(embeddings = embedding_pretrained, freeze = False) # shape = [num_words, 300]
        else:
            self.embedding = nn.Embedding(num_embeddings = num_words, embedding_dim = embedding_dim)

        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_size, num_layers = 2, bidirectional = True, batch_first=True, dropout = 0.5)
        self.fc = nn.Linear(in_features = hidden_size*2, out_features = num_classes)

    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        return out


if __name__ == '__main__':
    model = Model()
    print('embedding: ', model.embedding)
    print('lstm: ', model.lstm)
    print('out: ', model.fc)

    import utils

    loader = utils.return_loader()

    result = []
    for i, (x, y) in enumerate(loader):
        # x = torch.tensor(x).to(torch.int64)
        if i > 20:
            break
        result.append(model(x))

    print(result)