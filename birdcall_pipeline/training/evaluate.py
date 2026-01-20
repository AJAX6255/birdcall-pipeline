"""
Evaluation helpers: compute basic metrics for classifier outputs.
"""
from typing import Dict
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def evaluate(model, dataloader, device: str = 'cpu') -> Dict[str, float]:
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for (sample, label) in dataloader:
            inputs = sample.get('spectrogram') if isinstance(sample, dict) and 'spectrogram' in sample else sample['audio']
            if isinstance(inputs, np.ndarray):
                inputs = torch.tensor(inputs, dtype=torch.float32)
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)
            inputs = inputs.to(device)
            outputs = model(inputs)
            pred = outputs.argmax(dim=1).cpu().numpy()
            preds.extend(pred.tolist())
            targets.extend(label.numpy().tolist())
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro')
    return {'accuracy': float(acc), 'f1_macro': float(f1)}
