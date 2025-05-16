import os
import numpy as np
import torch
from colorama import Fore, Style
import matplotlib.pyplot as plt
from datetime import datetime

class DebugCapture:
    """Utility class for capturing and saving data for debugging purposes"""
    
    def __init__(self, max_samples=100, directory=None):
        """
        Initialize the debug capture
        
        Args:
            max_samples: Maximum number of samples to capture
            directory: Directory to save debug data
        """
        self.max_samples = max_samples
        self.sample_count = 0
        self.captured_data = {}
        
        if directory is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.directory = os.path.join(os.getcwd(), f"debug_data_{timestamp}")
        else:
            self.directory = directory
        
        os.makedirs(self.directory, exist_ok=True)
        
    def capture(self, name, data):
        """
        Capture a data sample
        
        Args:
            name: Name of the data to capture
            data: Data to capture (typically tensor or numpy array)
        """
        if self.sample_count >= self.max_samples:
            return
            
        # Convert to numpy if tensor
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()
            
        # Initialize list for this data name if not exists
        if name not in self.captured_data:
            self.captured_data[name] = []
            
        # Store the data
        self.captured_data[name].append(data)
        self.sample_count += 1
        
    def save_to_file(self):
        """Save all captured data to text files"""
        print(Fore.GREEN + f"[DEBUG] Saving {self.sample_count} debug samples to {self.directory}" + Style.RESET_ALL)
        
        for name, data_list in self.captured_data.items():
            file_path = os.path.join(self.directory, f"{name}.txt")
            
            with open(file_path, 'w') as f:
                f.write(f"# {name} - {len(data_list)} samples\n")
                f.write("# Format: [sample_idx][dim1][dim2]... = value\n\n")
                
                for i, sample in enumerate(data_list):
                    f.write(f"Sample {i}:\n")
                    # Handle different dimensions
                    if sample.ndim == 1:
                        f.write(f"  {sample.tolist()}\n")
                    elif sample.ndim == 2:
                        for j, row in enumerate(sample):
                            f.write(f"  Row {j}: {row.tolist()}\n")
                    else:
                        f.write(f"  Shape: {sample.shape}\n")
                        f.write(f"  Data: {sample.flatten()[:10]}...(truncated)\n")
                    f.write("\n")
                    
            print(Fore.BLUE + f"[DEBUG] Saved {name} data to {file_path}" + Style.RESET_ALL)
    
    def plot_data(self, name, max_dims=5):
        """
        Generate plots for the captured data
        
        Args:
            name: Name of the data to plot
            max_dims: Maximum number of dimensions to plot
        """
        if name not in self.captured_data or not self.captured_data[name]:
            print(Fore.RED + f"[DEBUG] No data for {name} to plot" + Style.RESET_ALL)
            return
            
        data_list = self.captured_data[name]
        sample = data_list[0]
        
        # For 1D data, plot time series for first few dimensions
        if sample.ndim == 1 or sample.ndim == 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if sample.ndim == 1:
                # For 1D samples, each sample is a point
                samples_array = np.array(data_list)
                dims = min(max_dims, samples_array.shape[1])
                for i in range(dims):
                    ax.plot(samples_array[:, i], label=f'Dim {i}')
            else:
                # For 2D samples, we'll plot the first row of each sample
                samples_array = np.array([s[0, :min(max_dims, s.shape[1])] for s in data_list])
                dims = samples_array.shape[1]
                for i in range(dims):
                    ax.plot(samples_array[:, i], label=f'Dim {i}')
                    
            ax.set_title(f'{name} - First {dims} dimensions')
            ax.set_xlabel('Sample index')
            ax.set_ylabel('Value')
            ax.legend()
            
            plot_path = os.path.join(self.directory, f"{name}_plot.png")
            plt.savefig(plot_path)
            plt.close()
            print(Fore.BLUE + f"[DEBUG] Saved plot to {plot_path}" + Style.RESET_ALL)

# Example usage:
# debugger = DebugCapture(max_samples=100)
# for i in range(100):
#     obs = get_observation()  # Your function to get observation
#     debugger.capture("observation", obs)
#     action = get_action(obs)  # Your function to get action
#     debugger.capture("action", action)
# debugger.save_to_file()
# debugger.plot_data("observation")
# debugger.plot_data("action")
