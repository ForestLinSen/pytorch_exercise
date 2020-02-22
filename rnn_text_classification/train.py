import torch
import numpy as np
from Model_RNN import Model
import utils


def train(data_iter, model):

    learning_rate = 0.01;

    model.train();
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    for index, (x, y) in enumerate(data_iter):
        output = model(x).squeeze()
        model.zero_grad()
        loss = torch.nn.functional.cross_entropy(output, y.squeeze()) # if not y.squeeze(), y.shape will be [5, 1] instead of [5]
        loss.backward()
        optimizer.step()

        if index % 100 == 0:
            print(loss.item())


if __name__ == "__main__":
    model = Model()
    data_iter = utils.return_loader()

    train(data_iter, model)