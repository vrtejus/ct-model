# CONVERT NII.GZ TO DICOM FILES
import SimpleITK as sitk
import os
from pathlib import Path
import datetime
import nibabel as nib

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