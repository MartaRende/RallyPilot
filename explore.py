import torch
from torch import nn
from data_loader import ClientLoader, CSVClientLoader
from model import MLP

model_file = "try.pickle"
model = MLP()
model.load_state_dict(torch.load(f"./models/{model_file}"))


dict1 = model.state_dict()
dict2 = model.state_dict()

dicts = [dict1, dict2]

# print(model.state_dict()["MLP.0.weight"])
# print(list(model.state_dict().keys()))


def mean(tensors):
    res = tensors[0]
    for t in tensors[1:]:
        res += t
    return res / len(tensors)


print(mean([a, b]))
