import os
import subprocess
import logging
from pathlib import Path
from tqdm.notebook import tqdm
import glob
import re
import SimpleITK as sitk
import os
from pathlib import Path
import datetime
import nibabel as nib

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

# CONVERT NII.GZ TO DICOM FILES
def is_valid_nifti(file_path):
    """
    Validate if a file is a valid NIfTI file.
    """
    try:
        # Check with nibabel
        nib.load(file_path)
        # Check with SimpleITK
        img = sitk.ReadImage(file_path)
        return True
    except Exception as e:
        print(f"Invalid NIfTI file: {file_path}. Error: {e}")
        return False
    
    
def convert_nifti_to_dicom_direct(nifti_path, output_dir):
    """
    Convert NIfTI to DICOM without requiring a reference DICOM.
    
    Args:
        nifti_path (str): Path to input NIfTI file
        output_dir (str): Path to output directory for DICOM files
    """
    if not is_valid_nifti(nifti_path):
        print(f"Skipping invalid file: {nifti_path}")
        return

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Read the NIfTI file
        print(f"Reading NIfTI file: {nifti_path}")
        img = sitk.ReadImage(nifti_path)
        
        # Convert to integer type (necessary for DICOM)
        # First normalize to 0-1, then scale to 16-bit range
        img = sitk.Cast(img, sitk.sitkFloat32)
        img = sitk.RescaleIntensity(img, 0, 32767)
        img = sitk.Cast(img, sitk.sitkUInt16)
        
        # Get basic image properties
        size = img.GetSize()
        spacing = img.GetSpacing()
        
        # Setup basic DICOM metadata
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        
        # Modify metadata
        modification_time = datetime.datetime.now()
        modification_time_str = modification_time.strftime("%H%M%S")
        modification_date = modification_time.strftime("%Y%m%d")
        
        direction = img.GetDirection()
        
        series_tag_values = [
            ("0008|0031", modification_time_str),  # Series Time
            ("0008|0021", modification_date),      # Series Date
            ("0008|0008", "DERIVED\\SECONDARY"),   # Image Type
            ("0020|000d", "1.2.826.0.1.3680043.2.1125." + modification_date + ".1" + modification_time_str), # Study Instance UID
            ("0020|000e", "1.2.826.0.1.3680043.2.1125." + modification_date + ".1" + modification_time_str), # Series Instance UID
            ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],
                                             direction[1], direction[4], direction[7])))),  # Image Orientation (Patient)
            ("0028|0030", '\\'.join(map(str, spacing[:2]))),  # Pixel Spacing
            ("0018|0050", str(spacing[2]) if len(spacing) > 2 else "1"),  # Slice Thickness
            ("0008|0060", "CT"),  # Modality
            ("0008|1030", "Converted from NIfTI"),  # Study Description
            ("0010|0010", nifti_path),  # Patient Name
            ("0010|0020", "12345"),  # Patient ID
            ("0028|0103", "16"),    # Bits Allocated
            ("0028|0100", "16"),    # Bits Stored
            ("0028|0102", "15"),    # High Bit
            ("0028|0101", "0"),     # Bits Stored
            ("0028|0004", "MONOCHROME2"),  # Photometric Interpretation
        ]
        
        # Write slices to DICOM files
        print(f"Converting to DICOM and saving to: {output_dir}")
        for i in range(size[2] if len(size) > 2 else 1):
            if len(size) > 2:
                slice_idx = i
                image_slice = img[:,:,i]
            else:
                slice_idx = 0
                image_slice = img
                
            # Set slice-specific tags
            slice_tag_values = [
                ("0020|0032", '\\'.join(map(str, img.TransformIndexToPhysicalPoint((0, 0, slice_idx))))),  # Image Position (Patient)
                ("0020|0013", str(i + 1))  # Instance Number
            ]
            
            # Set metadata for the slice
            for tag, value in series_tag_values + slice_tag_values:
                image_slice.SetMetaData(tag, value)
            
            # Write the slice
            dicom_file = os.path.join(output_dir, f'slice_{i+1:03d}.dcm')
            writer.SetFileName(dicom_file)
            writer.Execute(image_slice)
        
        print("Conversion completed successfully!")
        print(f"Total slices converted: {size[2] if len(size) > 2 else 1}")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        raise