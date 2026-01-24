import torch
import torch.nn as nn

class SkeletonLSTMModel(nn.Module):
    def __init__(self, input_size=48, hidden_size=512, num_layers=2, num_classes=6, dropout=0.5):
        super(SkeletonLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, skeleton_data):
        lstm_out, _ = self.lstm(skeleton_data)
        lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        return output