import torch
import numpy as np
import re
import os
import glob
from itertools import islice

class ChunkedActivationDataset(torch.utils.data.IterableDataset):
    """
    A PyTorch IterableDataset for data with activations stored in chunked .npz files.
    This dataset streams data from disk, loading one chunk at a time, making it
    suitable for datasets that are too large to fit in memory.

    Args:
        base_dataset (torch.utils.data.Dataset): The original map-style dataset 
            (e.g., from torchvision.datasets) that provides the raw images and labels.
        activations_dir (str): The directory containing the 'chunk_*.npz' files.
        recon_key (str): The key for reconstructed activations in the .npz files 
            (e.g., 'recon').
        label_key (str): The key for labels in the .npz files (e.g., 'labels').
        shuffle_chunks (bool): If True, shuffles the order of chunks each epoch.
        shuffle_in_chunk (bool): If True, shuffles the samples within each loaded chunk.
    """
    def __init__(self, base_dataset, activations_dir, recon_key='recon', label_key='labels', 
                 shuffle_chunks=True, shuffle_in_chunk=True):
        super().__init__()
        self.base_dataset = base_dataset
        self.activations_dir = activations_dir
        self.recon_key = recon_key
        self.label_key = label_key
        self.shuffle_chunks = shuffle_chunks
        self.shuffle_in_chunk = shuffle_in_chunk

        # Discover chunk files and sort numerically by chunk number
        chunk_files = glob.glob(os.path.join(self.activations_dir, '*.npz'))
        if not chunk_files:
            raise FileNotFoundError(f"No chunk files found in {self.activations_dir}")
        
        # Sort by extracting the integer after 'chunk_' and before '.npz'
        self.chunk_files = sorted(
            chunk_files,
            key=lambda x: int(re.search(r'chunk_(\d+)\.npz', os.path.basename(x)).group(1))
        )
        
        print(f"Found {len(self.chunk_files)} activation chunks.")
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # Determine which chunks this worker should process
        if worker_info is None:  # Single-process data loading
            worker_id = 0
            num_workers = 1
            chunk_files_for_worker = self.chunk_files
        else:  # Multi-process data loading
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            # Each worker gets a slice of the chunk files
            chunk_files_for_worker = list(islice(self.chunk_files, worker_id, None, num_workers))

        # Optionally shuffle the order of chunks for this worker
        if self.shuffle_chunks:
            np.random.shuffle(chunk_files_for_worker)

        global_sample_idx = 0
        for chunk_path in self.chunk_files: # Iterate through all chunks to calculate global index
            data_chunk = np.load(chunk_path)
            num_samples_in_chunk = len(data_chunk[self.label_key])

            # Only process the chunk if it belongs to this worker
            if chunk_path in chunk_files_for_worker:
                print(f"[Worker {worker_id}] Loading chunk: {os.path.basename(chunk_path)}")
                
                # Load the required data from the chunk
                recon_activations = torch.from_numpy(data_chunk[self.recon_key])
                labels_in_chunk = data_chunk[self.label_key]

                indices = list(range(num_samples_in_chunk))
                # Optionally shuffle samples within the current chunk
                if self.shuffle_in_chunk:
                    np.random.shuffle(indices)

                for local_idx in indices:
                    # The global index corresponds to the original dataset's order
                    current_global_idx = global_sample_idx + local_idx
                    
                    # Fetch the original image and label from the base dataset
                    sample = self.base_dataset[current_global_idx]
                    image = sample['image']
                    original_label = sample['label']
                    
                    # Sanity check: ensure the label from the dataset matches the one saved in the chunk
                    assert original_label == labels_in_chunk[local_idx], \
                        f"Label mismatch at index {current_global_idx}!"

                    # Yield the data tuple your training loop expects
                    yield image, torch.tensor(original_label, dtype=torch.long), recon_activations[local_idx]

            # Update the global index counter for the next chunk
            global_sample_idx += num_samples_in_chunk