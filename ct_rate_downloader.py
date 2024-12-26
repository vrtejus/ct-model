import os
import subprocess
import logging
from pathlib import Path
from tqdm.notebook import tqdm
import glob
import re

class CTRateDownloader:
    def __init__(self, base_dir="./CT-RATE-new", repo_url="https://huggingface.co/datasets/ibrahimhamamci/CT-RATE", size_limit_gb=20):
        self.base_dir = Path(base_dir)
        self.repo_url = repo_url
        self.size_limit_bytes = size_limit_gb * 1024 * 1024 * 1024  # Convert GB to bytes
        self.current_size = 0
        
        # Setup minimal logging
        logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(__name__)

    def run_command(self, command):
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def clone_repository(self):
        if not self.base_dir.exists():
            os.environ["GIT_LFS_SKIP_SMUDGE"] = "1"
            return self.run_command(["git", "clone", self.repo_url, str(self.base_dir)])
        return True

    def find_nii_files(self):
        patterns = [
            str(self.base_dir / "dataset" / "train" / "**" / "*.nii.gz"),
            str(self.base_dir / "dataset" / "valid" / "**" / "*.nii.gz")
        ]
        all_files = []
        for pattern in patterns:
            all_files.extend(glob.glob(pattern, recursive=True))
        return sorted(all_files)

    def filter_nth_files(self, file_list, n=500):
        def extract_number(filepath):
            match = re.search(r'(?:train|valid)_(\d+)', filepath)
            return int(match.group(1)) if match else 0

        grouped_files = {}
        for file in file_list:
            num = extract_number(file)
            if num % n == 0:
                grouped_files.setdefault(num, []).append(file)

        return [file for group in grouped_files.values() for file in group]

    def get_file_size(self, file_path):
        """Get the size of an LFS file before downloading"""
        try:
            relative_path = os.path.relpath(file_path, self.base_dir)
            result = subprocess.run(
                ["git", "lfs", "ls-files", "-l", relative_path],
                capture_output=True,
                text=True,
                cwd=str(self.base_dir)
            )
            
            # Git LFS verbose output format includes the size in bytes
            # Example output: "* path/to/file.nii.gz (140.1 MB)"
            if result.stdout:
                size_match = re.search(r'\(([\d.]+)\s*([KMGT]?B)\)', result.stdout)
                if size_match:
                    size_num = float(size_match.group(1))
                    unit = size_match.group(2)
                    
                    # Convert to bytes based on unit
                    multiplier = {
                        'B': 1,
                        'KB': 1024,
                        'MB': 1024 * 1024,
                        'GB': 1024 * 1024 * 1024,
                        'TB': 1024 * 1024 * 1024 * 1024
                    }
                    return int(size_num * multiplier.get(unit, 1))
            return 0
        except (subprocess.CalledProcessError, IndexError, ValueError, AttributeError):
            # If any error occurs during size calculation, return 0
            return 0

    def download_lfs_file(self, file_path):
        """Download a single LFS file with explicit error checking"""
        try:
            relative_path = os.path.relpath(file_path, self.base_dir)
            
            # First attempt with checkout
            result = subprocess.run(
                ["git", "lfs", "fetch", "--include", relative_path],
                capture_output=True,
                text=True,
                cwd=str(self.base_dir)
            )
            
            if result.returncode == 0:
                # Now checkout the file
                checkout_result = subprocess.run(
                    ["git", "lfs", "checkout", relative_path],
                    capture_output=True,
                    text=True,
                    cwd=str(self.base_dir)
                )
                return checkout_result.returncode == 0
                
            # Print error for debugging
            if not hasattr(self, '_error_printed'):
                print(f"\nGit LFS command failed for {relative_path}")
                print(f"Error: {result.stderr}")
                self._error_printed = True
                
            return False
            
        except Exception as e:
            if not hasattr(self, '_error_printed'):
                print(f"\nException during download: {str(e)}")
                self._error_printed = True
            return False

    def download_lfs_files(self, file_list):
        # Calculate total size first
        size_pbar = tqdm(file_list, desc="Calculating total size", unit="file")
        total_size = 0
        files_to_download = []
        
        for file_path in size_pbar:
            file_size = self.get_file_size(file_path)
            if total_size + file_size > self.size_limit_bytes:
                print(f"\nSize limit ({self.size_limit_bytes/1024/1024/1024:.1f}GB) would be exceeded")
                break
            total_size += file_size
            files_to_download.append(file_path)
        
        size_pbar.close()
        
        if not files_to_download:
            print("\nNo files to download!")
            return
        
        print(f"\nPreparing to download {len(files_to_download)} files, total size: {total_size/1024/1024/1024:.1f}GB")
        
        # Download files with size tracking
        success_count = 0
        download_pbar = tqdm(files_to_download, 
                           desc=f"Downloading files", 
                           unit="file")
        
        for file_path in download_pbar:
            if self.download_lfs_file(file_path):
                success_count += 1
        
        print(f"\nDownload complete. Successfully downloaded {success_count} out of {len(files_to_download)} files.")

    def run(self, nth_file=500):
        if not self.clone_repository():
            return
        
        all_files = self.find_nii_files()
        if not all_files:
            return
            
        filtered_files = self.filter_nth_files(all_files, nth_file)
        self.download_lfs_files(filtered_files)