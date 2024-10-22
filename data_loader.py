import lzma
import os
import pickle
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset

BATCH_SIZE = 64
BASE_FOLDER = "./data/client"
TEST_RATIO = 0.2
PERCENT_BATCH = 0.1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def transformData(data):
    X = [list(x.raycast_distances) + [x.car_speed] for x in data]
    y = [x.current_controls for x in data]
    return X, y


class ClientLoader:

    clientNumber: int
    testLoader: DataLoader
    trainLoaders: list[DataLoader]

    def __init__(self, client_number):
        self.clientNumber = client_number
        files = os.listdir(f"{BASE_FOLDER}{self.clientNumber}")
        X, y = self.loadFile(files)
        self.createLoader(X, y)

    def loadFile(self, filesPath) -> tuple[list[list[float]], list[list[int]]]:
        X = []
        y = []
        for f in filesPath:
            file = lzma.open(f"{BASE_FOLDER}{self.clientNumber}/{f}", "rb")
            data = pickle.load(file)
            currX, currY = transformData(data)
            assert len(currX[0]) == 16
            X += currX
            y += currY
        return X, y

    def createLoader(self, X, y):
        self.trainLoaders = []
        xTensor = torch.tensor(X, dtype=torch.float32)
        xTensor = xTensor.to(device)
        yTensor = torch.tensor(y, dtype=torch.float32)
        yTensor = yTensor.to(device)
        dataset = TensorDataset(xTensor, yTensor)
        train_size = int(1 - TEST_RATIO * len(dataset))
        test_size = len(dataset) - train_size
        trainData, testData = random_split(dataset, [train_size, test_size])
        self.testLoader = DataLoader(testData, batch_size=BATCH_SIZE)
        batchSize = int(PERCENT_BATCH * len(trainData))
        currData = trainData
        for i in range(int(1 / PERCENT_BATCH)):
            restSize = len(currData) - batchSize
            currBatch, currData = random_split(currData, [batchSize, restSize])
            self.trainLoaders.append(DataLoader(currBatch, batch_size=BATCH_SIZE))


class CSVClientLoader(ClientLoader):
    def loadFile(self, filesPath) -> tuple[list[list[float]], list[list[int]]]:
        X = []
        y = []
        for f in filesPath:
            df = pd.read_csv(f"{BASE_FOLDER}{self.clientNumber}/{f}")
            for h, r in df.iterrows():
                l = r.to_list()
                res = []
                res += l[1:16]
                res.append(l[0])
                assert len(res) == 16
                X.append(res)
                y.append(l[16:21])
        return X, y
