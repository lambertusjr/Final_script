import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
# Bug 2 fix: ensure calculate_metrics is imported (was commented out)
from helper_functions import calculate_metrics
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
        all_preds, all_labels = [], []
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
        # Bug 6 fix: mask is a single tensor, not a list — use directly instead of mask[0]
        self.model.train()
        device = next(self.model.parameters()).device
        data = data.to(device)
        self.optimiser.zero_grad()
        train_mask = mask.to(device)
        out = self.model(data)

        out_sliced = out[train_mask]
        y_sliced = data.y[train_mask]

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
        # Bug 7 fix: masks is a single tensor, not a list — use directly instead of masks[0]
        # Bug 3 fix: data.y[0] was wrong, use data.y[eval_mask] to get correct labels
        self.model.eval()
        device = next(self.model.parameters()).device
        eval_mask = masks.to(device)
        with torch.no_grad():
            data = data.to(device)
            out = self.model(data)
            out_sliced = out[eval_mask]
            y_sliced = data.y[eval_mask]

            loss = self.criterion(out_sliced, y_sliced)

            pred = out_sliced.argmax(dim=1)

            f1_metric = BinaryF1Score().to(device)
            f1_illicit = f1_metric(pred, y_sliced).item()

        return float(loss.detach()), f1_illicit


    def evaluate_full(self, data, mask):
        # Bug 8 fix: mask is a single tensor, not a list — use directly instead of mask[0]
        self.model.eval()
        device = next(self.model.parameters()).device

        with torch.no_grad():
            data = data.to(device)
            val_mask = mask.to(device)
            out = self.model(data)

            out_sliced = out[val_mask]
            y_sliced = data.y[val_mask]
            
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
        
        #train_mask = mask[0].to(device)
        out = self.model(data)


        #Extracting performance masks and applying to out and y to compute loss and metrics only on the known nodes of the known nodes subset of the training set.
        train_perf_eval_mask = mask[1].to(device)
        out_sliced = out[train_perf_eval_mask]
        y_sliced = data.y[train_perf_eval_mask]
        
        loss = self.criterion(out_sliced, y_sliced)
        loss.backward()
        self.optimiser.step()


    def evaluate_elliptic(self, data, masks):
        self.model.eval()
        device = next(self.model.parameters()).device
        #Extract the full mask, not only the performance nodes, since we want to predict on all known nodes of the known nodes subset of the training set, but we will evaluate performance only on the performance evaluation mask.
        eval_mask = masks[0].to(device)
        with torch.no_grad():
            data = data.to(device)
            out = self.model(data)

            #Applying performance evaluation mask since we can only predict on known nodes.
            perf_eval_mask = masks[1].to(device)
            out_sliced = out[perf_eval_mask]
            y_sliced = data.y[perf_eval_mask]
            
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

    # Bug 1 fix: indented into ModelWrapper class (was at module level, so self didn't work)
    # Bug 5 fix: data.y[0] → data.y[eval_mask] to get correct masked labels
    def mini_eval_elliptic(self, data, masks):
        self.model.eval()
        device = next(self.model.parameters()).device
        eval_mask = masks[0].to(device)
        with torch.no_grad():
            data = data.to(device)
            out = self.model(data)

            perf_eval_mask = masks[1].to(device)
            out_sliced = out[perf_eval_mask]
            y_sliced = data.y[perf_eval_mask]

            loss = self.criterion(out_sliced, y_sliced)

            pred = out_sliced.argmax(dim=1)

            f1_metric = BinaryF1Score().to(device)
            f1_illicit = f1_metric(pred, y_sliced).item()

        return float(loss.detach()), f1_illicit

    # Bug 12 fix: lightweight validation method for loader-based training (used by train_and_validate_with_loader)
    def evaluate_loader_mini(self, loader):
        self.model.eval()
        device = next(self.model.parameters()).device
        all_preds, all_labels = [], []
        total_loss = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = self.model(batch)
                batch_size = batch.batch_size
                out_sliced = out[:batch_size]
                y_sliced = batch.y[:batch_size]

                loss = self.criterion(out_sliced, y_sliced)
                total_loss += float(loss.detach())

                pred = out_sliced.argmax(dim=1)
                all_preds.append(pred.cpu())
                all_labels.append(y_sliced.cpu())

                del batch, out, out_sliced, y_sliced, loss, pred

        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_preds).numpy()
        f1_illicit = f1_score(y_true, y_pred, pos_label=1, average='binary')

        return total_loss / len(loader), f1_illicit

#%% GCN
class GCN(torch.nn.Module):
    """
    A simple Graph Convolutional Network model.
    """
    def __init__(self, num_node_features, num_classes, hidden_units, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_units)
        self.bn1 = nn.BatchNorm1d(hidden_units)
        self.conv2 = GCNConv(hidden_units, num_classes)
        self.dropout = dropout

    def forward(self, data):
        # x: Node features [num_nodes, in_channels]
        # edge_index: Graph connectivity [2, num_edges]
        x = data.x
        edge_index = data.adj_t if hasattr(data, 'adj_t') else data.edge_index

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Output raw logits
        x = self.conv2(x, edge_index)
        return x

#%% GAT
class GAT(nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_units, num_heads, dropout_1=0.6, dropout_2=0.5):
        super(GAT, self).__init__()
        # Keep the total latent size roughly equal to hidden_units while limiting per-head width
        per_head_dim = max(1, math.ceil(hidden_units / num_heads))
        total_hidden = per_head_dim * num_heads
        self.conv1 = GATConv(num_node_features, per_head_dim, heads=num_heads, dropout=dropout_1)
        self.bn1 = nn.BatchNorm1d(total_hidden)
        self.conv2 = GATConv(total_hidden, num_classes, heads=1, concat=False, dropout=dropout_2)

    def forward(self, data):
        x = data.x
        edge_index = data.adj_t if hasattr(data, 'adj_t') else data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x
    
#%% GIN
class GIN(nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_units, dropout=0.0):
        super(GIN, self).__init__()
        nn1 = nn.Sequential(
            nn.Linear(num_node_features, hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units)
        )
        self.conv1 = GINConv(nn1, train_eps=True)

        nn2 = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units)
        )
        self.conv2 = GINConv(nn2, train_eps=True)
        self.bn2 = nn.BatchNorm1d(hidden_units)
        self.fc = nn.Linear(hidden_units, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x = data.x
        edge_index = data.adj_t if hasattr(data, 'adj_t') else data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc(x)
        return x

#%% MLP
class MLP(nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_units, dropout_1=0.6, dropout_2=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(num_node_features, hidden_units)
        self.bn1 = nn.BatchNorm1d(hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.bn2 = nn.BatchNorm1d(hidden_units)
        self.fc3 = nn.Linear(hidden_units, num_classes)
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2

    def forward(self, data):
        x = data.x  # only use node features, no graph structure
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=self.dropout_1, training=self.training)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=self.dropout_2, training=self.training)
        x = self.fc3(x)
        return x