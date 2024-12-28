import os
import pydicom
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk
from scipy.ndimage import zoom
import nibabel as nib
from medsam2.modeling.image_encoder import ImageEncoderViT
from medsam2.modeling.mask_decoder import MaskDecoder
from medsam2.modeling.prompt_encoder import PromptEncoder
from medsam2.utils.transforms import ResizeLongestSide

def load_dicom_series(directory):
    """Load DICOM series from directory and convert to numpy array"""
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return sitk.GetArrayFromImage(image)

def preprocess_volume(volume):
    """Preprocess 3D volume for MedSAM-2"""
    # Normalize to 0-1
    volume = (volume - volume.min()) / (volume.max() - volume.min())
    
    # Resize to expected input size (1024x1024 per slice)
    resize_factor = [1, 1024/volume.shape[1], 1024/volume.shape[2]]
    volume_resized = zoom(volume, resize_factor, order=3)
    
    return volume_resized

class MedSAM2:
    def __init__(self, checkpoint_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.image_encoder = ImageEncoderViT()
        self.prompt_encoder = PromptEncoder()
        self.mask_decoder = MaskDecoder()
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.image_encoder.load_state_dict(checkpoint['image_encoder'])
        self.prompt_encoder.load_state_dict(checkpoint['prompt_encoder'])
        self.mask_decoder.load_state_dict(checkpoint['mask_decoder'])
        
        self.transform = ResizeLongestSide(1024)
    
    def generate_bbox_prompt(self, slice_shape):
        """Generate bounding box prompt for entire slice"""
        return np.array([[0, 0, slice_shape[1], slice_shape[0]]])
    
    def segment_volume(self, volume):
        """Segment entire 3D volume"""
        segmentation = np.zeros_like(volume)
        
        with torch.no_grad():
            for i in tqdm(range(volume.shape[0])):
                slice_img = volume[i]
                
                # Convert to torch tensor and add batch/channel dimensions
                slice_tensor = torch.from_numpy(slice_img).float().to(self.device)
                slice_tensor = slice_tensor.unsqueeze(0).unsqueeze(0)
                
                # Generate prompt
                bbox = self.generate_bbox_prompt(slice_img.shape)
                
                # Get embeddings
                image_embedding = self.image_encoder(slice_tensor)
                prompt_embedding = self.prompt_encoder(bbox)
                
                # Generate mask
                mask_logits = self.mask_decoder(image_embedding, prompt_embedding)
                mask_prob = torch.sigmoid(mask_logits)
                
                # Convert to binary mask
                mask = (mask_prob > 0.5).float().cpu().numpy()[0, 0]
                segmentation[i] = mask
        
        return segmentation

def evaluate_segmentation(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    
    dice = (2 * intersection) / (pred.sum() + gt.sum())
    iou = intersection / union
    
    return {
        'dice': dice,
        'iou': iou
    }

def main():
    #TODO: Set Paths
    ct_rate_path = "path/to/ct-rate/dataset"
    checkpoint_path = "path/to/medsam2_checkpoint.pth"
    output_path = "path/to/output"
    
    model = MedSAM2(checkpoint_path)
    
    # Process each case
    results = []
    for case_dir in Path(ct_rate_path).glob("*/"):
        print(f"Processing {case_dir.name}")
        
        # Load DICOM series
        volume = load_dicom_series(str(case_dir))
        
        # Load ground truth if available
        gt_path = case_dir / "segmentation.nii.gz"
        if gt_path.exists():
            gt = nib.load(gt_path).get_fdata()
        else:
            gt = None
        
        # Preprocess
        volume_processed = preprocess_volume(volume)
        
        # Segment
        segmentation = model.segment_volume(volume_processed)
        
        # Save segmentation
        output_file = Path(output_path) / f"{case_dir.name}_seg.nii.gz"
        nib.save(nib.Nifti1Image(segmentation, np.eye(4)), str(output_file))
        
        # Evaluate if ground truth available
        if gt is not None:
            metrics = evaluate_segmentation(segmentation, gt)
            results.append({
                'case': case_dir.name,
                **metrics
            })
    
    if results:
        dice_scores = [r['dice'] for r in results]
        iou_scores = [r['iou'] for r in results]
        
        print(f"Average Dice Score: {np.mean(dice_scores):.3f} ± {np.std(dice_scores):.3f}")
        print(f"Average IoU Score: {np.mean(iou_scores):.3f} ± {np.std(iou_scores):.3f}")

if __name__ == "__main__":
    main()