import os
import torch
import numpy as np
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, config):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.criterion    = criterion
        self.optimizer    = optimizer
        self.device       = device
        self.config       = config
        self.save_dir     = config['training'].get('save_dir', 'outputs/checkpoints')
        self.epochs       = config['training'].get('epochs', 20)
        self.num_classes  = config['dataset'].get('num_classes', 19)

        # ---- Early stopping ----
        self.patience          = config['training'].get('patience', 5)
        self.epochs_no_improve = 0

        os.makedirs(self.save_dir, exist_ok=True)

        # ---- LR scheduler: reduce on plateau ----
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_miou(self, preds, masks):
        """Batch mIoU — returns scalar."""
        preds  = preds.view(-1).cpu().numpy()
        masks  = masks.view(-1).cpu().numpy()
        ious   = []
        for cls in range(self.num_classes):
            p = preds == cls
            t = masks == cls
            intersection = (p & t).sum()
            union        = (p | t).sum()
            if union == 0:
                continue   # class absent → skip
            ious.append(intersection / union)
        return float(np.mean(ious)) if ious else 0.0

    # ------------------------------------------------------------------
    # Train one epoch
    # ------------------------------------------------------------------
    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        epoch_miou = 0.0
        pbar = tqdm(self.train_loader, desc="Training")

        for images, masks in pbar:
            images = images.to(self.device)
            masks  = masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss    = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            miou  = self._compute_miou(preds, masks)

            epoch_loss += loss.item()
            epoch_miou += miou
            pbar.set_postfix(loss=f"{loss.item():.4f}", mIoU=f"{miou:.4f}")

        n = len(self.train_loader)
        return epoch_loss / n, epoch_miou / n

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        val_miou = 0.0
        pbar = tqdm(self.val_loader, desc="Validating")

        with torch.no_grad():
            for images, masks in pbar:
                images = images.to(self.device)
                masks  = masks.to(self.device)

                outputs = self.model(images)
                loss    = self.criterion(outputs, masks)
                preds   = torch.argmax(outputs, dim=1)
                miou    = self._compute_miou(preds, masks)

                val_loss += loss.item()
                val_miou += miou
                pbar.set_postfix(loss=f"{loss.item():.4f}", mIoU=f"{miou:.4f}")

        n = len(self.val_loader)
        return val_loss / n, val_miou / n

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------
    def train(self):
        best_val_loss = float('inf')

        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch {epoch}/{self.epochs}  "
                  f"(lr={self.optimizer.param_groups[0]['lr']:.2e})")

            train_loss, train_miou = self.train_epoch()
            val_loss,   val_miou   = self.validate()

            print(f"  Train  →  loss: {train_loss:.4f}  mIoU: {train_miou:.4f}")
            print(f"  Val    →  loss: {val_loss:.4f}  mIoU: {val_miou:.4f}")

            # LR scheduler step
            self.scheduler.step(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.epochs_no_improve = 0
                save_path = os.path.join(self.save_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), save_path)
                print(f" Saved best model  (val_loss={val_loss:.4f})")
            else:
                self.epochs_no_improve += 1
                print(f" No improvement for {self.epochs_no_improve}/{self.patience} epochs")

            # Always save latest
            torch.save(self.model.state_dict(),
                       os.path.join(self.save_dir, 'latest_model.pth'))

            # Early stopping
            if self.epochs_no_improve >= self.patience:
                print(f"\n Early stopping triggered after {epoch} epochs.")
                break

        print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
