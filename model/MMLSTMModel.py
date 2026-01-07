import torch
import torch.nn as nn
from .SharedFeatureExtractor import SharedFeatureExtractor
class MMLSTMModel(nn.Module):
    def __init__(self, seq_len=168, pred_len=24, d_model=128, n_layers=2):
        super().__init__()
        self.feature_extractor = SharedFeatureExtractor(fusion_dim=d_model)
        
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, 
                            num_layers=n_layers, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, pred_len)

    def forward(self, x_load, x_img, x_text):
        x = self.feature_extractor(x_load, x_img, x_text)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc_out(out).unsqueeze(-1)