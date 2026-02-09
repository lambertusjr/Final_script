import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
#from helper_functions import calculate_metrics, calculate_pr_metrics_batched, save_pr_artifacts
from sklearn.metrics import f1_score

#%% Modelwrapper

class ModelWrapper:
    def __init__(self, model, optimiser, criterion):
        self.model = model
        self.optimiser = optimiser
        self.criterion = criterion

    def train_step(self, loader):
        self.model.train()
        total_loss = 0
        device = next(self.model.parameters()).device
        all_preds, all_probs, all_labels = [], [], []
        for batch in loader:
            batch = batch.to(device)
            self.optimiser.zero_grad()
            
            out = self.model(batch)
            # Slice to get only target nodes (first batch_size nodes in NeighborLoader)
            batch_size = batch.batch_size
            out_sliced = out[:batch_size]
            y_sliced = batch.y[:batch_size]
            
            # Validate labels before computing loss
            num_classes = out_sliced.shape[-1]
            if y_sliced.min() < 0 or y_sliced.max() >= num_classes:
                raise ValueError(
                    f"Invalid labels detected: min={y_sliced.min().item()}, "
                    f"max={y_sliced.max().item()}, but model has {num_classes} classes. "
                    f"Labels must be in range [0, {num_classes-1}]"
                )
            
            loss = self.criterion(out_sliced, y_sliced)
            loss.backward()
            self.optimiser.step()
            total_loss += float(loss.detach())

            pred = out_sliced.argmax(dim=1)
            
            all_preds.append(pred.cpu())
            all_labels.append(y_sliced.cpu())
            
            # Clean up batch to prevent memory accumulation
            del batch, out, out_sliced, y_sliced, loss, pred

        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_preds).numpy()

        f1_illicit = f1_score(y_true, y_pred, pos_label=1, average='binary') # illicit is class 1
        

        return total_loss / len(loader), f1_illicit
    
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        all_preds, all_probs, all_labels = [], [], []
        device = next(self.model.parameters()).device

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = self.model(batch)
                
                batch_size = batch.batch_size
                out_sliced = out[:batch_size]
                y_sliced = batch.y[:batch_size]
                
                loss = self.criterion(out_sliced, y_sliced)
                total_loss += float(loss.detach())
                
                probs = torch.softmax(out_sliced, dim=1)
                pred = out_sliced.argmax(dim=1)
                
                all_preds.append(pred.cpu())
                all_probs.append(probs.cpu())
                all_labels.append(y_sliced.cpu())
                
                # Clean up batch to prevent memory accumulation
                del batch, out, out_sliced, y_sliced, loss, probs, pred

        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_preds).numpy()
        y_prob = torch.cat(all_probs).numpy()

        metrics = calculate_metrics(y_true, y_pred, y_prob)

        
        metrics['probs'] = y_prob
        metrics['preds'] = y_pred
        metrics['y_true'] = y_true
        return total_loss / len(loader), metrics