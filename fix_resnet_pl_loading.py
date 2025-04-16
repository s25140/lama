"""
Patch to fix the ResNetPL model loading issue.
Apply this patch before loading the model.
"""

import logging
import sys
import os
from typing import Dict, Any

# Add the parent directory to sys.path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import pytorch_lightning as pl
from saicinpainting.training.trainers import DefaultInpaintingTrainingModule
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector

logger = logging.getLogger(__name__)

def apply_resnet_pl_loading_fix():
    """
    Patches the DefaultInpaintingTrainingModule to ignore missing keys 
    in the ResNetPL component when loading state_dict.
    """
    # Store the original method
    original_load_state_dict = DefaultInpaintingTrainingModule.load_state_dict
    
    # Create a patched version that forces strict=False
    def patched_load_state_dict(self, state_dict, strict=True):
        logger.info("Applying ResNetPL loading fix - forcing strict=False for DefaultInpaintingTrainingModule")
        return original_load_state_dict(self, state_dict, strict=False)
    
    # Replace the method
    DefaultInpaintingTrainingModule.load_state_dict = patched_load_state_dict
    logger.info("ResNetPL loading patch successfully applied to DefaultInpaintingTrainingModule")

    # Also patch the base LightningModule for complete coverage
    original_pl_load = pl.LightningModule.load_state_dict
    
    def patched_pl_load(self, state_dict, strict=True):
        if any('loss_resnet_pl' in k for k in list(state_dict.keys())):
            logger.info("Detected ResNetPL keys in state_dict - using non-strict loading")
            return original_pl_load(self, state_dict, strict=False)
        return original_pl_load(self, state_dict, strict=strict)
    
    pl.LightningModule.load_state_dict = patched_pl_load
    logger.info("Also applied patch to base LightningModule class")
    
    # Patch the checkpoint connector to handle weights-only checkpoints
    original_restore_training_state = CheckpointConnector.restore_training_state
    
    def patched_restore_training_state(self, checkpoint):
        try:
            return original_restore_training_state(self, checkpoint)
        except KeyError as e:
            if "Trying to restore training state but checkpoint contains only the model" in str(e):
                logger.warning("Checkpoint contains only model weights. Loading weights only and skipping optimizer state.")
                return None
            else:
                raise
    
    CheckpointConnector.restore_training_state = patched_restore_training_state
    logger.info("Applied patch to CheckpointConnector to handle weights-only checkpoints")

if __name__ == "__main__":
    apply_resnet_pl_loading_fix()
    print("ResNetPL loading patch has been applied. Now you can load your model without strict key matching.")
