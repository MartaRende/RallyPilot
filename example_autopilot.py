

from PyQt6 import QtWidgets
import torch
from data_loader import SPEED_WEIGHT
from data_collector import DataCollectionUI
from model import MLP
"""
This file is provided as an example of what a simplistic controller could be done.
It simply uses the DataCollectionUI interface zo receive sensing_messages and send controls.

/!\ Be warned that if the processing time of NNMsgProcessor.process_message is superior to the message reception period, a lag between the images processed and commands sent.
One might want to only process the last sensing_message received, etc. 
Be warned that this could also cause crash on the client side if socket sending buffer overflows

/!\ Do not work directly in this file (make a copy and rename it) to prevent future pull from erasing what you write here.
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
number = 0
BASE_PATH = "./models/"
MODEL_FILENAME = "model.pickle"
MODEL_NAMES = [f"5_forward_23_back", "6_forward_27_back"]

weights = [0.7, 0.3]
models = []
for f in MODEL_NAMES:
    model = MLP(device)
    model.load_state_dict(torch.load(f"{BASE_PATH}{f}/{MODEL_FILENAME}", map_location=device))
    models.append(model)
   
    
class ExampleNNMsgProcessor:
    def __init__(self):
        self.always_forward = True

    def nn_infer(self, message):
        input = list(message.raycast_distances) + [message.car_speed]
        input = torch.tensor(input).to(device)
        input = input.unsqueeze(0)
        preds = [0,0,0,0]
        for i, m in enumerate(models):
            currInput = input.clone()
            currInput[0][15] = currInput[0][15] * m.SPEED_WEIGHT[0]
            for j, a in enumerate(m(currInput)[0]):
                preds[j] += a.item() * weights[i]
        
        if abs(message.car_speed) < 3 and preds[0] < 0.5:
            preds[0] = 1
            preds[1] = 0
        
        return [
            ("forward", preds[0] > 0.5),
            ("back", preds[1] > 0.5),
            ("left", preds[2] > 0.5),
            ("right", preds[3] > 0.5),
            ]

    def process_message(self, message, data_collector):

        commands = self.nn_infer(message)

        for command, start in commands:
            data_collector.onCarControlled(command, start)

if  __name__ == "__main__":
    import sys
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)

    nn_brain = ExampleNNMsgProcessor()
    data_window = DataCollectionUI(nn_brain.process_message)
    data_window.show()

    app.exec()