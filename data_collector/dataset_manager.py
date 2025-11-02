import h5py
import numpy as np
from collections import defaultdict


class DatasetManager:

    def __init__(self):
        self.data = []  # List of landmark arrays
        self.labels = []  # List of gesture IDs
        self.metadata = {
            'num_landmarks': 21,
            'landmark_dims': 2,
            'feature_size': 42  # 21 landmarks * 2 coordinates
        }
    
    def add_sample(self, landmarks, gesture_id):
        """Add a new sample to the dataset"""
        if landmarks.shape[0] != self.metadata['feature_size']:
            raise ValueError(f"Expected {self.metadata['feature_size']} features, "
                           f"got {landmarks.shape[0]}")
        
        self.data.append(landmarks.astype(np.float32))
        self.labels.append(int(gesture_id))
    
    def get_statistics(self):
        """Get statistics about the collected dataset"""
        stats = {
            'total_samples': len(self.data),
            'num_classes': len(set(self.labels)) if self.labels else 0,
            'samples_per_class': defaultdict(int)
        }
        
        for label in self.labels:
            stats['samples_per_class'][label] += 1
        
        return stats
    
    def save_to_h5(self, filename):
        """Save the dataset to an HDF5 file"""
        if not self.data:
            raise ValueError("No data to save")
        
        data_array = np.array(self.data, dtype=np.float32)
        labels_array = np.array(self.labels, dtype=np.int32)
        
        # Create HDF5 file
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('landmarks', data=data_array, compression='gzip')
            hf.create_dataset('labels', data=labels_array, compression='gzip')
            
            # Add metadata
            hf.attrs['total_samples'] = len(self.data)
            hf.attrs['num_classes'] = len(set(self.labels))
            hf.attrs['num_landmarks'] = self.metadata['num_landmarks']
            hf.attrs['landmark_dims'] = self.metadata['landmark_dims']
            hf.attrs['feature_size'] = self.metadata['feature_size']
            
            # Add class distribution
            stats = self.get_statistics()
            for class_id, count in stats['samples_per_class'].items():
                hf.attrs[f'class_{class_id}_count'] = count
        
        print(f"Dataset saved to {filename}")
        print(f"Total samples: {len(self.data)}")
        print(f"Number of classes: {len(set(self.labels))}")
    
    def load_from_h5(self, filename):
        """Load dataset from an HDF5 file"""
        with h5py.File(filename, 'r') as hf:
            self.data = list(hf['landmarks'][:])
            self.labels = list(hf['labels'][:])
            
            # Load metadata
            self.metadata['num_landmarks'] = hf.attrs['num_landmarks']
            self.metadata['landmark_dims'] = hf.attrs['landmark_dims']
            self.metadata['feature_size'] = hf.attrs['feature_size']
        
        print(f"Dataset loaded from {filename}")
        print(f"Total samples: {len(self.data)}")
        print(f"Number of classes: {len(set(self.labels))}")
    
    def clear(self):
        """Clear all collected data"""
        self.data = []
        self.labels = []
    
    def get_data_for_training(self):
        """Get data in format ready for training"""
        X = np.array(self.data, dtype=np.float32)
        y = np.array(self.labels, dtype=np.int32)
        return X, y