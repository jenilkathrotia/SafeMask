import os
import torch
import torch.nn as nn
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.save_dir = config['training'].get('save_dir', 'outputs/checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        self.epochs = config['training'].get('epochs', 20)

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        pbar = tqdm(self.train_loader, desc="Training")
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        return epoch_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0
        pbar = tqdm(self.val_loader, desc="Validating")
        with torch.no_grad():
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                val_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        return val_loss / len(self.val_loader)

    def train(self):
        best_val_loss = float('inf')
        for epoch in range(1, self.epochs + 1):
            print(f"Epoch {epoch}/{self.epochs}")
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(self.save_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved best model to {save_path}")

            latest_path = os.path.join(self.save_dir, 'latest_model.pth')
            torch.save(self.model.state_dict(), latest_path)
