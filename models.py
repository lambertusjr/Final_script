import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
#from helper_functions import calculate_metrics, calculate_pr_metrics_batched, save_pr_artifacts
from sklearn.metrics import f1_score
from torchmetrics.classification import BinaryF1Score

#%% Modelwrapper

class ModelWrapper:
    def __init__(self, model, optimiser, criterion):
        self.model = model
        self.optimiser = optimiser
        self.criterion = criterion

    def train_step_loader(self, loader):
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
    
    def evaluate_loader(self, loader):
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
    
    
    
    def train_step_full(self, data, mask):
        self.model.train()
        device = next(self.model.parameters()).device
        data = data.to(device)
        self.optimiser.zero_grad()
        train_mask = mask[0].to('cuda' if torch.cuda.is_available() else 'cpu')
        out = self.model(data[train_mask])
        
        # Apply mask if provided to select specific nodes (e.g., train_mask)
        if mask is not None:
            out_sliced = out
            y_sliced = data.y[train_mask]
        else:
            out_sliced = out
            y_sliced = data.y
        
        # # Validate labels before computing loss
        # num_classes = out_sliced.shape[-1]
        # if y_sliced.min() < 0 or y_sliced.max() >= num_classes:
        #     raise ValueError(
        #         f"Invalid labels detected: min={y_sliced.min().item()}, "
        #         f"max={y_sliced.max().item()}, but model has {num_classes} classes. "
        #         f"Labels must be in range [0, {num_classes-1}]"
        #     )
        
        loss = self.criterion(out_sliced, y_sliced)
        loss.backward()
        self.optimiser.step()
        
    def mini_eval_full(self, data, masks):
        self.model.eval()
        device = next(self.model.parameters()).device
        eval_mask = masks[0].to('cuda' if torch.cuda.is_available() else 'cpu') if masks is not None else None
        with torch.no_grad():
            data = data.to(device)
            out_sliced = self.model(data[eval_mask])

            if masks is not None:
                y_sliced = data.y[0]
            else:
                y_sliced = data.y

            loss = self.criterion(out_sliced, y_sliced)

            pred = out_sliced.argmax(dim=1)

            f1_metric = BinaryF1Score().to(device)
            f1_illicit = f1_metric(pred, y_sliced).item()

        return float(loss.detach()), f1_illicit


    def evaluate_full(self, data, mask):
        self.model.eval()
        device = next(self.model.parameters()).device

        with torch.no_grad():
            data = data.to(device)
            val_mask = mask[0].to('cuda' if torch.cuda.is_available() else 'cpu')
            out = self.model(data[val_mask])
            
            # Apply mask if provided (e.g., val_mask or test_mask)
            if mask is not None:
                out_sliced = out
                y_sliced = data.y[val_mask]
            else:
                out_sliced = out
                y_sliced = data.y
            
            loss = self.criterion(out_sliced, y_sliced)
            
            probs = torch.softmax(out_sliced, dim=1)
            pred = out_sliced.argmax(dim=1)
            
            y_true = y_sliced.cpu().numpy()
            y_pred = pred.cpu().numpy()
            y_prob = probs.cpu().numpy()

        metrics = calculate_metrics(y_true, y_pred, y_prob)
        
        metrics['probs'] = y_prob
        metrics['preds'] = y_pred
        metrics['y_true'] = y_true
        
        return float(loss.detach()), metrics
    
    def train_step_elliptic(self, data, mask):
        self.model.train()
        device = next(self.model.parameters()).device
        data = data.to(device)
        self.optimiser.zero_grad()
        
        train_mask = mask[0].to('cuda' if torch.cuda.is_available() else 'cpu')
        out = self.model(data[train_mask])
        
        # Apply mask if provided to select specific nodes (e.g., train_mask)
        if mask is not None:
            out_sliced = out
            y_sliced = data.y[train_mask]
        else:
            out_sliced = out
            y_sliced = data.y
        
        # # Validate labels before computing loss
        # num_classes = out_sliced.shape[-1]
        # if y_sliced.min() < 0 or y_sliced.max() >= num_classes:
        #     raise ValueError(
        #         f"Invalid labels detected: min={y_sliced.min().item()}, "
        #         f"max={y_sliced.max().item()}, but model has {num_classes} classes. "
        #         f"Labels must be in range [0, {num_classes-1}]"
        #     )
        #Extracting performance masks and applying to out and y to compute loss and metrics only on the known nodes of the known nodes subset of the training set.
        train_perf_eval_mask = mask[1].to('cuda' if torch.cuda.is_available() else 'cpu')
        out_sliced = out_sliced[train_perf_eval_mask]
        y_sliced = y_sliced[train_perf_eval_mask]
        
        loss = self.criterion(out_sliced, y_sliced)
        loss.backward()
        self.optimiser.step()


    def evaluate_elliptic(self, data, masks):
        self.model.eval()
        device = next(self.model.parameters()).device
        #Extract the full mask, not only the performance nodes, since we want to predict on all known nodes of the known nodes subset of the training set, but we will evaluate performance only on the performance evaluation mask.
        eval_mask = masks[0].to('cuda' if torch.cuda.is_available() else 'cpu') if masks is not None else None
        with torch.no_grad():
            data = data.to(device)
            out_sliced = self.model(data[eval_mask])
            
            
            # Apply mask if provided (e.g., val_mask or test_mask)
            if masks is not None:
                out_sliced = out_sliced
                y_sliced = data.y[0]
            else:
                out_sliced = out_sliced 
                y_sliced = data.y
            
            #Applying performance evaluation mask since we can only predict on known nodes.
            perf_eval_mask = masks[1].to('cuda' if torch.cuda.is_available() else 'cpu') if masks is not None else None
            out_sliced = out_sliced[perf_eval_mask]
            y_sliced = y_sliced[perf_eval_mask]
            
            loss = self.criterion(out_sliced, y_sliced)
            
            probs = torch.softmax(out_sliced, dim=1)
            pred = out_sliced.argmax(dim=1)
            
            y_true = y_sliced.cpu().numpy()
            y_pred = pred.cpu().numpy()
            y_prob = probs.cpu().numpy()

        metrics = calculate_metrics(y_true, y_pred, y_prob)
        
        metrics['probs'] = y_prob
        metrics['preds'] = y_pred
        metrics['y_true'] = y_true
        
        return float(loss.detach()), metrics
    
def mini_eval_elliptic(self, data, masks):
    self.model.eval()
    device = next(self.model.parameters()).device
    eval_mask = masks[0].to('cuda' if torch.cuda.is_available() else 'cpu') if masks is not None else None
    with torch.no_grad():
        data = data.to(device)
        out_sliced = self.model(data[eval_mask])

        if masks is not None:
            y_sliced = data.y[0]
        else:
            y_sliced = data.y

        perf_eval_mask = masks[1].to('cuda' if torch.cuda.is_available() else 'cpu') if masks is not None else None
        out_sliced = out_sliced[perf_eval_mask]
        y_sliced = y_sliced[perf_eval_mask]

        loss = self.criterion(out_sliced, y_sliced)

        pred = out_sliced.argmax(dim=1)

        f1_metric = BinaryF1Score().to(device)
        f1_illicit = f1_metric(pred, y_sliced).item()

    

    return float(loss.detach()), f1_illicit
    