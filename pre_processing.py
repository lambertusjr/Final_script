import torch
from torch_geometric.data import InMemoryDataset, Data
import os.path as osp
import os
import pandas as pd
from tqdm import tqdm
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
class EllipticDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(EllipticDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # elliptic_txs_features_raw.csv
        # elliptic_txs_classes_raw.csv
        # elliptic_txs_edgelist_raw.csv
        return ['elliptic_txs_features_raw.csv',
                'elliptic_txs_classes_raw.csv',
                'elliptic_txs_edgelist_raw.csv']

    @property
    def processed_file_names(self):
        # The name of the file where the processed data will be saved.
        return ['data.pt']


    def process(self):
        features_df = pd.read_csv(self.raw_paths[0], header=None)
        features_df.columns = ['txId'] + ['time_step'] + [f'V{i}' for i in range(1, 166)]
        classes_df = pd.read_csv(self.raw_paths[1])
        edgelist_df = pd.read_csv(self.raw_paths[2])
        
        # remap class nodes so labels don't cause weird error
        class_mapping = {
            'unknown': -1,
            '1': 1,
            '2': 0
        }
        classes_df['class'] = classes_df['class'].map(class_mapping)
        
        # Pre-proces node ids
        node_mapping = {old_id: new_id for new_id, old_id in enumerate(features_df['txId'])}
        classes_df['txId'] = classes_df['txId'].map(node_mapping)
        edgelist_df['txId1'] = edgelist_df['txId1'].map(node_mapping)
        edgelist_df['txId2'] = edgelist_df['txId2'].map(node_mapping)
        edgelist_df = edgelist_df.dropna() # Drop edges with missing nodes after mapping
        classes_df = classes_df.sort_values('txId').set_index('txId')
        # 3. Create Tensors
        features_tensor = torch.tensor(features_df.drop(columns=['txId']).values, dtype=torch.float)
        edge_index_tensor = torch.tensor(edgelist_df.values.T, dtype=torch.long)
        y_tensor = torch.tensor(classes_df['class'].values, dtype=torch.long)

        # 4. Create Data Object
        data = Data(x=features_tensor, edge_index=edge_index_tensor, y=y_tensor)

        # 5. Create Masks
        time_steps = data.x[:, 0]
        known_nodes_mask = (data.y != -1)
        
        train_mask = (time_steps >= 1) & (time_steps <= 30)
        val_mask = (time_steps >= 31) & (time_steps <= 40)
        test_mask = (time_steps >= 41) & (time_steps <= 49)
        
        train_perf_eval_mask = train_mask & known_nodes_mask
        val_perf_eval_mask = val_mask & known_nodes_mask
        test_perf_eval_mask = test_mask & known_nodes_mask

        # Adding masks to data object for easy acces with data.py file
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        data.train_perf_eval_mask = train_perf_eval_mask
        data.val_perf_eval_mask = val_perf_eval_mask
        data.test_perf_eval_mask = test_perf_eval_mask
        
        # Save the processed data object.
        torch.save(self.collate([data]), self.processed_paths[0])


class IBMAMLDataset_HiSmall(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # The main transaction file
        return ['HI-Small_Trans_raw.csv']

    @property
    def processed_file_names(self):
        # The file where the processed Data object will be saved
        return ['data.pt']

    def _preprocess_ibm_edges(self, data_df, num_obs, delta_minutes=240):
        """
        Internal helper to replicate the edge creation logic from 
        src/data/DatasetConstruction.py [preprocess_ibm].
        """
        edge_file = os.path.join(self.processed_dir, 'edges.csv')
        
        # Check if edges are already processed to save time
        if os.path.exists(edge_file):
            print("Edge file already exists. Loading...")
            return pd.read_csv(edge_file)

        print("Processing edges... This may take a while.")
        date_format = '%Y/%m/%d %H:%M'
        
        # We only need specific columns for edge processing
        data_df_accounts = data_df[['txId', 'Account', 'Account.1', 'Timestamp']]
        
        source = []
        target = []
        
        # Iterate over the dataframe in pieces to manage memory
        pieces = 100
        for i in tqdm(range(pieces)):
            start = i * num_obs // pieces
            end = (i + 1) * num_obs // pieces
            data_df_right = data_df_accounts.iloc[start:end]
            
            if data_df_right.empty:
                continue
                
            min_timestamp = data_df_right['Timestamp'].min()
            max_timestamp = data_df_right['Timestamp'].max()

            # Define the "left" window based on the time delta
            delta = timedelta(minutes=delta_minutes)
            window_start = min_timestamp - delta
            
            data_df_left = data_df_accounts[
                (data_df_accounts['Timestamp'] >= window_start) & 
                (data_df_accounts['Timestamp'] <= max_timestamp)
            ]

            # Find transactions where recipient of 'left' is sender of 'right'
            # This corresponds to: row['Account.1_1'] == row['Account_2']
            data_df_join = data_df_left.merge(
                data_df_right, 
                left_on='Account.1', 
                right_on='Account', 
                suffixes=('_1', '_2')
            )

            for _, row in data_df_join.iterrows():
                delta_trans = row['Timestamp_2'] - row['Timestamp_1']
                total_minutes = delta_trans.days * 24 * 60 + delta_trans.seconds / 60
                
                # Check if B is within 4 hours *after* A
                if 0 <= total_minutes <= delta_minutes:
                    source.append(row['txId_1'])
                    target.append(row['txId_2'])

        df_edges = pd.DataFrame({'txId1': source, 'txId2': target})
        df_edges.to_csv(edge_file, index=False)
        print(f"Edge processing complete. Found {len(df_edges)} edges.")
        return df_edges

    def process(self):
        # This logic is adapted from src/data/DatasetConstruction.py [load_ibm]
        
        print("Reading raw transaction data...")
        df_features = pd.read_csv(
            self.raw_paths[0],
            dtype={
                'Amount Received': 'float32',
                'Amount Paid': 'float32',
                'class': 'int8' # Saves memory on labels
            }
        )
        
        # 1. Basic filtering and sorting
        date_format = '%Y/%m/%d %H:%M'
        df_features['Timestamp'] = pd.to_datetime(df_features['Timestamp'], format=date_format)
        df_features.sort_values('Timestamp', inplace=True)
        df_features = df_features[df_features['Account'] != df_features['Account.1']]

        # 2. Select last 500k transactions
        num_obs = len(df_features)
        start_index = int(len(df_features) - num_obs)
        df_features = df_features.iloc[start_index:]
        
        # 3. Create new txId (node index) from 0 to N-1
        df_features.reset_index(drop=True, inplace=True)
        df_features.reset_index(inplace=True)
        df_features.rename(columns={'index': 'txId'}, inplace=True) # txId is now 0..N-1
        
        # 4. Select relevant columns
        df_features.columns = ['txId', 'Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1', 'Amount Received', 'Receiving Currency', 'Amount Paid', 'Payment Currency', 'Payment Format', 'class']
        df_features = df_features[['txId', 'Timestamp', 'Amount Received', 'Receiving Currency', 'Amount Paid', 'Payment Currency', 'Payment Format', 'class', 'Account', 'Account.1']]

        # 5. Generate edges
        # We pass the processed df_features to the helper
        df_edges = self._preprocess_ibm_edges(
            data_df=df_features[['txId', 'Account', 'Account.1', 'Timestamp']],
            num_obs=num_obs,
            delta_minutes=240
        )
        
        # 6. Feature Engineering
        print("Performing feature engineering...")
        timestamps = pd.DatetimeIndex(df_features['Timestamp'])
        df_features['Day'] = timestamps.day.astype('float32')
        df_features['Hour'] = timestamps.hour.astype('float32')
        df_features['Minute'] = timestamps.minute.astype('float32')

        # Drop columns not used as features or labels
        df_features = df_features.drop(columns=['Timestamp', 'Account', 'Account.1'])
        
        # One-hot encode categorical features
        df_features = pd.get_dummies(
            df_features, 
            columns=['Receiving Currency', 'Payment Currency', 'Payment Format'], 
            dtype='float32'
        )
        scaler = StandardScaler()
        df_features['Amount Received'] = scaler.fit_transform(df_features[['Amount Received']])
        df_features['Amount Paid'] = scaler.fit_transform(df_features[['Amount Paid']])
        # 7. Prepare Tensors
        # Get labels (y)
        y = torch.tensor(df_features['class'].values, dtype=torch.long)
        
        # Get features (x)
        feature_cols = df_features.columns.drop(['txId', 'class'])
        x = torch.tensor(df_features[feature_cols].values, dtype=torch.float)
        
        # Get edge_index
        # The txId in df_edges already corresponds to the 0..N-1 index
        edge_index = torch.tensor(df_edges[['txId1', 'txId2']].values, dtype=torch.long).t().contiguous()

        # 8. Create Masks (60/20/20 split)
        mask = torch.tensor([False] * num_obs)
        train_size = int(0.6 * num_obs)
        val_size = int(0.2 * num_obs)

        train_mask = mask.clone()
        train_mask[:train_size] = True
        val_mask = mask.clone()
        val_mask[train_size:train_size + val_size] = True
        test_mask = mask.clone()
        test_mask[train_size + val_size:] = True
        
        # 9. Create Data object and save
        data = Data(
            x=x, 
            edge_index=edge_index, 
            y=y, 
            train_mask=train_mask, 
            val_mask=val_mask, 
            test_mask=test_mask
        )
        torch.save(self.collate([data]), self.processed_paths[0])
        print("Processing finished. Data object saved.")
        
class IBMAMLDataset_LiSmall(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # The main transaction file
        return ['LI-Small_Trans_raw.csv']

    @property
    def processed_file_names(self):
        # The file where the processed Data object will be saved
        return ['data.pt']

    def _preprocess_ibm_edges(self, data_df, num_obs, delta_minutes=240):
        """
        Internal helper to replicate the edge creation logic from 
        src/data/DatasetConstruction.py [preprocess_ibm].
        """
        edge_file = os.path.join(self.processed_dir, 'edges.csv')
        
        # Check if edges are already processed to save time
        if os.path.exists(edge_file):
            print("Edge file already exists. Loading...")
            return pd.read_csv(edge_file)

        print("Processing edges... This may take a while.")
        date_format = '%Y/%m/%d %H:%M'
        
        # We only need specific columns for edge processing
        data_df_accounts = data_df[['txId', 'Account', 'Account.1', 'Timestamp']]
        
        source = []
        target = []
        
        # Iterate over the dataframe in pieces to manage memory
        pieces = 100
        for i in tqdm(range(pieces)):
            start = i * num_obs // pieces
            end = (i + 1) * num_obs // pieces
            data_df_right = data_df_accounts.iloc[start:end]
            
            if data_df_right.empty:
                continue
                
            min_timestamp = data_df_right['Timestamp'].min()
            max_timestamp = data_df_right['Timestamp'].max()

            # Define the "left" window based on the time delta
            delta = timedelta(minutes=delta_minutes)
            window_start = min_timestamp - delta
            
            data_df_left = data_df_accounts[
                (data_df_accounts['Timestamp'] >= window_start) & 
                (data_df_accounts['Timestamp'] <= max_timestamp)
            ]

            # Find transactions where recipient of 'left' is sender of 'right'
            # This corresponds to: row['Account.1_1'] == row['Account_2']
            data_df_join = data_df_left.merge(
                data_df_right, 
                left_on='Account.1', 
                right_on='Account', 
                suffixes=('_1', '_2')
            )

            for _, row in data_df_join.iterrows():
                delta_trans = row['Timestamp_2'] - row['Timestamp_1']
                total_minutes = delta_trans.days * 24 * 60 + delta_trans.seconds / 60
                
                # Check if B is within 4 hours *after* A
                if 0 <= total_minutes <= delta_minutes:
                    source.append(row['txId_1'])
                    target.append(row['txId_2'])

        df_edges = pd.DataFrame({'txId1': source, 'txId2': target})
        df_edges.to_csv(edge_file, index=False)
        print(f"Edge processing complete. Found {len(df_edges)} edges.")
        return df_edges

    def process(self):
        # This logic is adapted from src/data/DatasetConstruction.py [load_ibm]
        
        print("Reading raw transaction data...")
        df_features = pd.read_csv(self.raw_paths[0])
        
        # 1. Basic filtering and sorting
        date_format = '%Y/%m/%d %H:%M'
        df_features['Timestamp'] = pd.to_datetime(df_features['Timestamp'], format=date_format)
        df_features.sort_values('Timestamp', inplace=True)
        df_features = df_features[df_features['Account'] != df_features['Account.1']]

        # 2. Select last 500k transactions
        num_obs = len(df_features)
        start_index = int(len(df_features) - num_obs)
        df_features = df_features.iloc[start_index:]
        
        # 3. Create new txId (node index) from 0 to N-1
        df_features.reset_index(drop=True, inplace=True)
        df_features.reset_index(inplace=True)
        df_features.rename(columns={'index': 'txId'}, inplace=True) # txId is now 0..N-1
        
        # 4. Select relevant columns
        df_features.columns = ['txId', 'Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1', 'Amount Received', 'Receiving Currency', 'Amount Paid', 'Payment Currency', 'Payment Format', 'class']
        df_features = df_features[['txId', 'Timestamp', 'Amount Received', 'Receiving Currency', 'Amount Paid', 'Payment Currency', 'Payment Format', 'class', 'Account', 'Account.1']]

        # 5. Generate edges
        # We pass the processed df_features to the helper
        df_edges = self._preprocess_ibm_edges(
            data_df=df_features[['txId', 'Account', 'Account.1', 'Timestamp']],
            num_obs=num_obs,
            delta_minutes=240
        )
        
        # 6. Feature Engineering
        print("Performing feature engineering...")
        timestamps = pd.DatetimeIndex(df_features['Timestamp'])
        df_features['Day'] = timestamps.day.astype('float32')
        df_features['Hour'] = timestamps.hour.astype('float32')
        df_features['Minute'] = timestamps.minute.astype('float32')

        # Drop columns not used as features or labels
        df_features = df_features.drop(columns=['Timestamp', 'Account', 'Account.1'])
        
        # One-hot encode categorical features
        df_features = pd.get_dummies(
            df_features, 
            columns=['Receiving Currency', 'Payment Currency', 'Payment Format'], 
            dtype='float32'
        )
        scaler = StandardScaler()
        df_features['Amount Received'] = scaler.fit_transform(df_features[['Amount Received']])
        df_features['Amount Paid'] = scaler.fit_transform(df_features[['Amount Paid']])
        # 7. Prepare Tensors
        # Get labels (y)
        y = torch.tensor(df_features['class'].values, dtype=torch.long)
        
        # Get features (x)
        feature_cols = df_features.columns.drop(['txId', 'class'])
        x = torch.tensor(df_features[feature_cols].values, dtype=torch.float)
        
        # Get edge_index
        # The txId in df_edges already corresponds to the 0..N-1 index
        edge_index = torch.tensor(df_edges[['txId1', 'txId2']].values, dtype=torch.long).t().contiguous()

        # 8. Create Masks (60/20/20 split)
        mask = torch.tensor([False] * num_obs)
        train_size = int(0.6 * num_obs)
        val_size = int(0.2 * num_obs)

        train_mask = mask.clone()
        train_mask[:train_size] = True
        val_mask = mask.clone()
        val_mask[train_size:train_size + val_size] = True
        test_mask = mask.clone()
        test_mask[train_size + val_size:] = True
        


        # 9. Create Data object and save
        data = Data(
            x=x, 
            edge_index=edge_index, 
            y=y, 
            train_mask=train_mask, 
            val_mask=val_mask, 
            test_mask=test_mask
        )
        torch.save(self.collate([data]), self.processed_paths[0])
        print("Processing finished. Data object saved.")
        
class IBMAMLDataset_LiMedium(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # The main transaction file
        return ['LI-Medium_Trans_raw.csv']

    @property
    def processed_file_names(self):
        # The file where the processed Data object will be saved
        return ['data.pt']

    def _preprocess_ibm_edges(self, data_df, num_obs, delta_minutes=240):
        """
        Internal helper to replicate the edge creation logic from 
        src/data/DatasetConstruction.py [preprocess_ibm].
        """
        edge_file = os.path.join(self.processed_dir, 'edges.csv')
        
        # Check if edges are already processed to save time
        if os.path.exists(edge_file):
            print("Edge file already exists. Loading...")
            return pd.read_csv(edge_file)

        print("Processing edges... This may take a while.")
        date_format = '%Y/%m/%d %H:%M'
        
        # We only need specific columns for edge processing
        data_df_accounts = data_df[['txId', 'Account', 'Account.1', 'Timestamp']]
        
        source = []
        target = []
        
        # Iterate over the dataframe in pieces to manage memory
        pieces = 100
        for i in tqdm(range(pieces)):
            start = i * num_obs // pieces
            end = (i + 1) * num_obs // pieces
            data_df_right = data_df_accounts.iloc[start:end]
            
            if data_df_right.empty:
                continue
                
            min_timestamp = data_df_right['Timestamp'].min()
            max_timestamp = data_df_right['Timestamp'].max()

            # Define the "left" window based on the time delta
            delta = timedelta(minutes=delta_minutes)
            window_start = min_timestamp - delta
            
            data_df_left = data_df_accounts[
                (data_df_accounts['Timestamp'] >= window_start) & 
                (data_df_accounts['Timestamp'] <= max_timestamp)
            ]

            # Find transactions where recipient of 'left' is sender of 'right'
            # This corresponds to: row['Account.1_1'] == row['Account_2']
            data_df_join = data_df_left.merge(
                data_df_right, 
                left_on='Account.1', 
                right_on='Account', 
                suffixes=('_1', '_2')
            )

            for _, row in data_df_join.iterrows():
                delta_trans = row['Timestamp_2'] - row['Timestamp_1']
                total_minutes = delta_trans.days * 24 * 60 + delta_trans.seconds / 60
                
                # Check if B is within 4 hours *after* A
                if 0 <= total_minutes <= delta_minutes:
                    source.append(row['txId_1'])
                    target.append(row['txId_2'])

        df_edges = pd.DataFrame({'txId1': source, 'txId2': target})
        df_edges.to_csv(edge_file, index=False)
        print(f"Edge processing complete. Found {len(df_edges)} edges.")
        return df_edges

    def process(self):
        # This logic is adapted from src/data/DatasetConstruction.py [load_ibm]
        
        print("Reading raw transaction data...")
        df_features = pd.read_csv(self.raw_paths[0])
        
        # 1. Basic filtering and sorting
        date_format = '%Y/%m/%d %H:%M'
        df_features['Timestamp'] = pd.to_datetime(df_features['Timestamp'], format=date_format)
        df_features.sort_values('Timestamp', inplace=True)
        df_features = df_features[df_features['Account'] != df_features['Account.1']]

        # 2. Select last 500k transactions
        num_obs = len(df_features)
        start_index = int(len(df_features) - num_obs)
        df_features = df_features.iloc[start_index:]
        
        # 3. Create new txId (node index) from 0 to N-1
        df_features.reset_index(drop=True, inplace=True)
        df_features.reset_index(inplace=True)
        df_features.rename(columns={'index': 'txId'}, inplace=True) # txId is now 0..N-1
        
        # 4. Select relevant columns
        df_features.columns = ['txId', 'Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1', 'Amount Received', 'Receiving Currency', 'Amount Paid', 'Payment Currency', 'Payment Format', 'class']
        df_features = df_features[['txId', 'Timestamp', 'Amount Received', 'Receiving Currency', 'Amount Paid', 'Payment Currency', 'Payment Format', 'class', 'Account', 'Account.1']]

        # 5. Generate edges
        # We pass the processed df_features to the helper
        df_edges = self._preprocess_ibm_edges(
            data_df=df_features[['txId', 'Account', 'Account.1', 'Timestamp']],
            num_obs=num_obs,
            delta_minutes=240
        )
        
        # 6. Feature Engineering
        print("Performing feature engineering...")
        timestamps = pd.DatetimeIndex(df_features['Timestamp'])
        df_features['Day'] = timestamps.day.astype('float32')
        df_features['Hour'] = timestamps.hour.astype('float32')
        df_features['Minute'] = timestamps.minute.astype('float32')

        # Drop columns not used as features or labels
        df_features = df_features.drop(columns=['Timestamp', 'Account', 'Account.1'])
        
        # One-hot encode categorical features
        df_features = pd.get_dummies(
            df_features, 
            columns=['Receiving Currency', 'Payment Currency', 'Payment Format'], 
            dtype='float32'
        )
        scaler = StandardScaler()
        df_features['Amount Received'] = scaler.fit_transform(df_features[['Amount Received']])
        df_features['Amount Paid'] = scaler.fit_transform(df_features[['Amount Paid']])
        # 7. Prepare Tensors
        # Get labels (y)
        y = torch.tensor(df_features['class'].values, dtype=torch.long)
        
        # Get features (x)
        feature_cols = df_features.columns.drop(['txId', 'class'])
        x = torch.tensor(df_features[feature_cols].values, dtype=torch.float)
        
        # Get edge_index
        # The txId in df_edges already corresponds to the 0..N-1 index
        edge_index = torch.tensor(df_edges[['txId1', 'txId2']].values, dtype=torch.long).t().contiguous()

        # 8. Create Masks (60/20/20 split)
        mask = torch.tensor([False] * num_obs)
        train_size = int(0.6 * num_obs)
        val_size = int(0.2 * num_obs)

        train_mask = mask.clone()
        train_mask[:train_size] = True
        val_mask = mask.clone()
        val_mask[train_size:train_size + val_size] = True
        test_mask = mask.clone()
        test_mask[train_size + val_size:] = True
        


        # 9. Create Data object and save
        data = Data(
            x=x, 
            edge_index=edge_index, 
            y=y, 
            train_mask=train_mask, 
            val_mask=val_mask, 
            test_mask=test_mask
        )
        torch.save(self.collate([data]), self.processed_paths[0])
        print("Processing finished. Data object saved.")
        
class IBMAMLDataset_HiMedium(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # The main transaction file
        return ['HI-Medium_Trans_raw.csv']

    @property
    def processed_file_names(self):
        # The file where the processed Data object will be saved
        return ['data.pt']

    def _preprocess_ibm_edges(self, data_df, num_obs, delta_minutes=240):
        """
        Internal helper to replicate the edge creation logic from 
        src/data/DatasetConstruction.py [preprocess_ibm].
        """
        edge_file = os.path.join(self.processed_dir, 'edges.csv')
        
        # Check if edges are already processed to save time
        if os.path.exists(edge_file):
            print("Edge file already exists. Loading...")
            return pd.read_csv(edge_file)

        print("Processing edges... This may take a while.")
        date_format = '%Y/%m/%d %H:%M'
        
        # We only need specific columns for edge processing
        data_df_accounts = data_df[['txId', 'Account', 'Account.1', 'Timestamp']]
        
        source = []
        target = []
        
        # Iterate over the dataframe in pieces to manage memory
        pieces = 100
        for i in tqdm(range(pieces)):
            start = i * num_obs // pieces
            end = (i + 1) * num_obs // pieces
            data_df_right = data_df_accounts.iloc[start:end]
            
            if data_df_right.empty:
                continue
                
            min_timestamp = data_df_right['Timestamp'].min()
            max_timestamp = data_df_right['Timestamp'].max()

            # Define the "left" window based on the time delta
            delta = timedelta(minutes=delta_minutes)
            window_start = min_timestamp - delta
            
            data_df_left = data_df_accounts[
                (data_df_accounts['Timestamp'] >= window_start) & 
                (data_df_accounts['Timestamp'] <= max_timestamp)
            ]

            # Find transactions where recipient of 'left' is sender of 'right'
            # This corresponds to: row['Account.1_1'] == row['Account_2']
            data_df_join = data_df_left.merge(
                data_df_right, 
                left_on='Account.1', 
                right_on='Account', 
                suffixes=('_1', '_2')
            )

            for _, row in data_df_join.iterrows():
                delta_trans = row['Timestamp_2'] - row['Timestamp_1']
                total_minutes = delta_trans.days * 24 * 60 + delta_trans.seconds / 60
                
                # Check if B is within 4 hours *after* A
                if 0 <= total_minutes <= delta_minutes:
                    source.append(row['txId_1'])
                    target.append(row['txId_2'])

        df_edges = pd.DataFrame({'txId1': source, 'txId2': target})
        df_edges.to_csv(edge_file, index=False)
        print(f"Edge processing complete. Found {len(df_edges)} edges.")
        return df_edges

    def process(self):
        # This logic is adapted from src/data/DatasetConstruction.py [load_ibm]
        
        print("Reading raw transaction data...")
        df_features = pd.read_csv(self.raw_paths[0])
        
        # 1. Basic filtering and sorting
        date_format = '%Y/%m/%d %H:%M'
        df_features['Timestamp'] = pd.to_datetime(df_features['Timestamp'], format=date_format)
        df_features.sort_values('Timestamp', inplace=True)
        df_features = df_features[df_features['Account'] != df_features['Account.1']]

        # 2. Select last 500k transactions
        num_obs = len(df_features)
        start_index = int(len(df_features) - num_obs)
        df_features = df_features.iloc[start_index:]
        
        # 3. Create new txId (node index) from 0 to N-1
        df_features.reset_index(drop=True, inplace=True)
        df_features.reset_index(inplace=True)
        df_features.rename(columns={'index': 'txId'}, inplace=True) # txId is now 0..N-1
        
        # 4. Select relevant columns
        df_features.columns = ['txId', 'Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1', 'Amount Received', 'Receiving Currency', 'Amount Paid', 'Payment Currency', 'Payment Format', 'class']
        df_features = df_features[['txId', 'Timestamp', 'Amount Received', 'Receiving Currency', 'Amount Paid', 'Payment Currency', 'Payment Format', 'class', 'Account', 'Account.1']]

        # 5. Generate edges
        # We pass the processed df_features to the helper
        df_edges = self._preprocess_ibm_edges(
            data_df=df_features[['txId', 'Account', 'Account.1', 'Timestamp']],
            num_obs=num_obs,
            delta_minutes=240
        )
        
        # 6. Feature Engineering
        print("Performing feature engineering...")
        
        timestamps = pd.DatetimeIndex(df_features['Timestamp'])
        df_features['Day'] = timestamps.day.astype('float32')
        df_features['Hour'] = timestamps.hour.astype('float32')
        df_features['Minute'] = timestamps.minute.astype('float32')

        # Drop columns not used as features or labels
        df_features = df_features.drop(columns=['Timestamp', 'Account', 'Account.1'])
        
        # One-hot encode categorical features
        df_features = pd.get_dummies(
            df_features, 
            columns=['Receiving Currency', 'Payment Currency', 'Payment Format'], 
            dtype='float32'
        )
        scaler = StandardScaler()
        df_features['Amount Received'] = scaler.fit_transform(df_features[['Amount Received']])
        df_features['Amount Paid'] = scaler.fit_transform(df_features[['Amount Paid']])
        # 7. Prepare Tensors
        # Get labels (y)
        y = torch.tensor(df_features['class'].values, dtype=torch.long)
        
        # Get features (x)
        feature_cols = df_features.columns.drop(['txId', 'class'])
        x = torch.tensor(df_features[feature_cols].values, dtype=torch.float)
        
        # Get edge_index
        # The txId in df_edges already corresponds to the 0..N-1 index
        edge_index = torch.tensor(df_edges[['txId1', 'txId2']].values, dtype=torch.long).t().contiguous()

        # 8. Create Masks (60/20/20 split)
        mask = torch.tensor([False] * num_obs)
        train_size = int(0.6 * num_obs)
        val_size = int(0.2 * num_obs)

        train_mask = mask.clone()
        train_mask[:train_size] = True
        val_mask = mask.clone()
        val_mask[train_size:train_size + val_size] = True
        test_mask = mask.clone()
        test_mask[train_size + val_size:] = True
        

        
        # 9. Create Data object and save
        data = Data(
            x=x, 
            edge_index=edge_index, 
            y=y, 
            train_mask=train_mask, 
            val_mask=val_mask, 
            test_mask=test_mask
        )
        torch.save(self.collate([data]), self.processed_paths[0])
        print("Processing finished. Data object saved.")
        
#AMLSim dataset
class AMLSimDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(AMLSimDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # elliptic_txs_features_raw.csv
        # elliptic_txs_classes_raw.csv
        # elliptic_txs_edgelist_raw.csv
        return ['accounts.csv',
                'transactions.csv',
                'alerts.csv']

    @property
    def processed_file_names(self):
        # The name of the file where the processed data will be saved.
        return ['data.pt']


    def process(self):
        import numpy as np

        accounts_df = pd.read_csv(self.raw_paths[0])
        transactions_df = pd.read_csv(self.raw_paths[1])
        alerts_df = pd.read_csv(self.raw_paths[2])

        # 1. Merge transactions with alerts to get fraud labels per transaction
        txn = pd.merge(transactions_df, alerts_df, on='TX_ID', how='left')
        txn = txn.rename(columns={
            'SENDER_ACCOUNT_ID_x': 'SENDER_ACCOUNT',
            'RECEIVER_ACCOUNT_ID_x': 'RECEIVER_ACCOUNT',
            'TX_AMOUNT_x': 'TX_AMOUNT',
            'TIMESTAMP_x': 'TIMESTAMP',
            'IS_FRAUD_x': 'IS_FRAUD'
        })
        txn['IS_FRAUD'] = txn['IS_FRAUD'].fillna(0).astype(int)

        # 2. Build account-to-index mapping (nodes = accounts)
        all_account_ids = sorted(accounts_df['ACCOUNT_ID'].unique())
        account_to_idx = {acc_id: idx for idx, acc_id in enumerate(all_account_ids)}
        num_accounts = len(all_account_ids)
        print(f"Number of account nodes: {num_accounts}")

        # 3. Account-level labels: illicit if the account sent or received
        #    any fraudulent transaction
        fraud_senders = set(txn.loc[txn['IS_FRAUD'] == 1, 'SENDER_ACCOUNT'].unique())
        fraud_receivers = set(txn.loc[txn['IS_FRAUD'] == 1, 'RECEIVER_ACCOUNT'].unique())
        fraud_accounts = fraud_senders | fraud_receivers
        y = torch.zeros(num_accounts, dtype=torch.long)
        for acc_id in fraud_accounts:
            if acc_id in account_to_idx:
                y[account_to_idx[acc_id]] = 1
        print(f"Illicit accounts: {y.sum().item()} / {num_accounts} "
              f"({100 * y.sum().item() / num_accounts:.2f}%)")

        # 4. Edge index: each transaction becomes a directed edge
        #    sender_account -> receiver_account (mapped to node indices)
        src = txn['SENDER_ACCOUNT'].map(account_to_idx).values
        dst = txn['RECEIVER_ACCOUNT'].map(account_to_idx).values
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        print(f"Number of edges (transactions): {edge_index.shape[1]}")

        # 5. Account-level features
        #    - INT_BALANCE from accounts.csv (direct feature, no computation)
        #    - Aggregated transaction amounts per account (mean/total sent & received)
        accounts_sorted = accounts_df.set_index('ACCOUNT_ID').loc[all_account_ids]

        sent_agg = txn.groupby('SENDER_ACCOUNT')['TX_AMOUNT'].agg(
            mean_amount_sent='mean', total_amount_sent='sum'
        )
        recv_agg = txn.groupby('RECEIVER_ACCOUNT')['TX_AMOUNT'].agg(
            mean_amount_received='mean', total_amount_received='sum'
        )

        feat_df = pd.DataFrame({'ACCOUNT_ID': all_account_ids})
        feat_df = feat_df.merge(
            accounts_sorted[['INT_BALANCE']].reset_index(),
            on='ACCOUNT_ID', how='left'
        )
        feat_df = feat_df.merge(sent_agg, left_on='ACCOUNT_ID', right_index=True, how='left')
        feat_df = feat_df.merge(recv_agg, left_on='ACCOUNT_ID', right_index=True, how='left')
        feat_df = feat_df.fillna(0)

        # 6. Train/val/test split (60/20/20) on accounts
        train_size = int(0.6 * num_accounts)
        val_size = int(0.2 * num_accounts)

        train_df = feat_df.iloc[:train_size].copy()
        val_df = feat_df.iloc[train_size:train_size + val_size].copy()
        test_df = feat_df.iloc[train_size + val_size:].copy()

        # 7. Normalise numerical features (fit on train only)
        num_cols = ['INT_BALANCE', 'mean_amount_sent', 'total_amount_sent',
                    'mean_amount_received', 'total_amount_received']
        scaler = StandardScaler()
        train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
        val_df[num_cols] = scaler.transform(val_df[num_cols])
        test_df[num_cols] = scaler.transform(test_df[num_cols])

        feat_df = pd.concat([train_df, val_df, test_df])
        feature_cols = [c for c in feat_df.columns if c != 'ACCOUNT_ID']
        x = torch.tensor(feat_df[feature_cols].values, dtype=torch.float)
        print(f"Feature matrix shape: {x.shape}  ({len(feature_cols)} features: {feature_cols})")

        # 8. Create masks
        mask = torch.zeros(num_accounts, dtype=torch.bool)
        train_mask = mask.clone()
        train_mask[:train_size] = True
        val_mask = mask.clone()
        val_mask[train_size:train_size + val_size] = True
        test_mask = mask.clone()
        test_mask[train_size + val_size:] = True

        # 9. Create Data object and save
        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        torch.save(self.collate([data]), self.processed_paths[0])
        print("Processing finished. Data object saved.")