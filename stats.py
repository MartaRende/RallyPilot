import numpy
import pickle
import lzma


fd_data = pickle.load(lzma.open("./stats/federated_model.npz", "rb"))
lcl_data = pickle.load(lzma.open("./stats/local_model.npz", "rb"))

print(vars(fd_data[0]))
