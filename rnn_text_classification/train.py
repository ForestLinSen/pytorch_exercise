import torch
import numpy as np
from Model_RNN import Model
import utils
from sklearn import metrics


def train(data_iter, model):

    learning_rate = 0.01;

    model.train();
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    for index, (x, y) in enumerate(data_iter):
        y = y.reshape([20])
        output = model(x).squeeze()
        model.zero_grad()
        loss = torch.nn.functional.cross_entropy(output, y) # if not y.squeeze(), y.shape will be [5, 1] instead of [5]
        loss.backward()
        optimizer.step()

        if index % 100 == 0:
            result = torch.max(output.data, 1)[1]
            print("output data: ", output.data)
            print("result: ", result)
            acc = metrics.accuracy_score(y_true = y, y_pred = result)
            print("loss: {}, acc: {}".format(loss.item(), acc))



if __name__ == "__main__":
    model = Model()
    data_iter = utils.return_loader()

    train(data_iter, model)