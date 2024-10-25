import os.path
import torch
from torch import nn
from data_loader import ClientLoader, CSVClientLoader, SPEED_WEIGHT
from model import MLP
import multiprocessing as mp
import matplotlib.pyplot as plt

clients = [ClientLoader(x) for x in range(1, 7)]
clients.append(CSVClientLoader(7))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Devide for torch is : " + str(device))


print(f"Initialized {len(clients)} clients")


CURR_MODEL_NAME = "4_forward_25_back"

LOSS_FN_WEIGHT = torch.tensor([0.4, 2.5, 1, 1])
LOSS_FN_WEIGHT = LOSS_FN_WEIGHT.to(device)
LEARNING_RATE = 0.001
N_EPOCH = 50


def printEpochAcc(epochAcc: list[dict[int, float]]):
    for i, e in enumerate(epochAcc):
        print(f"============ Epoch n°{i + 1} ============")
        for k in e.keys():
            print(f"Client n°{k} : {e[k]}")
        print("============ END ============")


def mean_tensors(tensors):
    res = tensors[0]
    for t in tensors[1:]:
        res += t
    return res / len(tensors)


def trainBaseModel():
    baseModel = MLP(device)
    baseModel = baseModel.to(device)
    optimizer = torch.optim.Adam(baseModel.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(LOSS_FN_WEIGHT)
    loss_fn = loss_fn.to(device)

    # We can take the first train loader of each client to train base model -> this equals to training it on 10% of data for each client
    baseModel.train()

    client_acc = {}

    for c in clients:
        correct = 0
        total = 0
        for X, y in c.trainLoaders.pop(0):
            optimizer.zero_grad()
            y_pred = baseModel(X)
            loss = loss_fn(y_pred, y)
            y_pred = (y_pred > 0.5).float()
            correct += (y_pred == y).sum().item()
            total += y.size(0) * y.size(1)
            loss.backward()
            optimizer.step()
        client_acc[c.clientNumber] = correct / total
    print(f"Base training done, acc was : {client_acc}")
    return baseModel


def doTrainingOnClient(args) -> MLP:
    c, model, stepN = args
    currModel = MLP(device)
    currModel.to(device)
    currModel.load_state_dict(model.state_dict())
    loss_fn = nn.CrossEntropyLoss(LOSS_FN_WEIGHT)
    optimizer = torch.optim.Adam(currModel.parameters(), lr=LEARNING_RATE)
    currModel.train()
    for X, y in c.trainLoaders[stepN]:
        optimizer.zero_grad()
        y_pred = currModel(X)
        loss = loss_fn(y_pred, y)
        y_pred = (y_pred > 0.5).float()
        loss.backward()
        optimizer.step()
    return currModel


def doTrainingStep(model: MLP, stepN) -> MLP:
    models: list[MLP] = []
    for c in clients:
        currModel = MLP(device)
        currModel.load_state_dict(model.state_dict())
        loss_fn = nn.CrossEntropyLoss(LOSS_FN_WEIGHT)
        optimizer = torch.optim.Adam(currModel.parameters(), lr=LEARNING_RATE)
        currModel.train()
        for X, y in c.trainLoaders[stepN]:
            optimizer.zero_grad()
            y_pred = currModel(X)
            loss = loss_fn(y_pred, y)
            y_pred = (y_pred > 0.5).float()
            loss.backward()
            optimizer.step()
        models.append(currModel)
    dicts = [x.state_dict() for x in models]
    new_state_dict = {}
    for k in list(model.state_dict().keys()):
        tensors = []
        for d in dicts:
            tensors.append(d[k])
        new_state_dict[k] = mean_tensors(tensors)
    resModel = MLP(device)
    resModel.load_state_dict(new_state_dict)
    return resModel


def evaluateClients(currModel: MLP):
    currModel.eval()
    trainAcc = {}
    testAcc = {}
    with torch.no_grad():
        for c in clients:
            correct = 0
            total = 0
            for t in c.trainLoaders:
                for X, y in t:
                    y_pred = currModel(X)
                    y_pred = (y_pred > 0.5).float()

                    correct += (y_pred == y).sum().item()
                    total += y.size(0) * y.size(1)
            trainAcc[c.clientNumber] = correct / total
            correct = 0
            total = 0
            for X, y in c.testLoader:
                y_pred = currModel(X)
                y_pred = (y_pred > 0.5).float()

                correct += (y_pred == y).sum().item()
                total += y.size(0) * y.size(1)
            testAcc[c.clientNumber] = correct / total
    return trainAcc, testAcc


# At first, we train a basic model on 10% of all the datas
baseModel = trainBaseModel()

currModel = baseModel
trainAccs = {}
testAccs = {}

for c in clients:
    trainAccs[c.clientNumber] = []
    testAccs[c.clientNumber] = []

for n in range(N_EPOCH):
    for i in range(9):
        currModel = doTrainingStep(currModel, i)
    trainAcc, testAcc = evaluateClients(currModel)
    for c in trainAcc.keys():
        trainAccs[c].append(trainAcc[c])
        testAccs[c].append(testAcc[c])
    if (n + 1) % 10 == 0:
        print(f"Epoch n°{n+1}")


def getGraphs(trainAcc, testAcc, currPath):
    fullPath = currPath + "graphs/"
    os.mkdir(fullPath)
    for c in clients:
        currTrain = trainAcc[c.clientNumber]
        currTest = testAcc[c.clientNumber]
        x = range(len(currTrain))
        yTrain = currTrain
        yTest = currTest
        plt.figure()
        plt.plot(x, yTrain)
        plt.plot(x, yTest)
        plt.legend(["Train", "Test"])
        plt.savefig(f"{fullPath}client{c.clientNumber}.png")


index = 0
BASE_PATH = "./models/"
FILENAME = "model.pickle"
currPath = f"{BASE_PATH}{CURR_MODEL_NAME}/"
while os.path.exists(currPath):
    index += 1
    currPath = f"{BASE_PATH}{CURR_MODEL_NAME}{index}/"
os.mkdir(currPath)
torch.save(currModel.state_dict(), currPath + FILENAME)
getGraphs(trainAccs, testAccs, currPath)
