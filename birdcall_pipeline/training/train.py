"""
Minimal training loop utilities.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train(model: nn.Module, dataset, device: str = 'cpu', epochs: int = 1, batch_size: int = 8, lr: float = 1e-3):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for (sample, label) in loader:
            # sample is expected to be a dict with 'audio' or 'spectrogram'
            inputs = sample.get('spectrogram') if isinstance(sample, dict) and 'spectrogram' in sample else sample['audio']
            # convert to tensor if needed
            if isinstance(inputs, list) or isinstance(inputs, tuple) or (isinstance(inputs, np.ndarray) and inputs.dtype != 'float32'):
                inputs = torch.tensor(np.array(inputs), dtype=torch.float32)
            if isinstance(inputs, np.ndarray):
                inputs = torch.tensor(inputs, dtype=torch.float32)
            # ensure shape (B, C, F, T) for model
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)
            inputs = inputs.to(device)
            labels = label.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        print(f"Epoch {epoch+1}/{epochs} loss={{total_loss:.4f}}")
