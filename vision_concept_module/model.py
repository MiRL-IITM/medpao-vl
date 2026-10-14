import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
import math
from PIL import Image
import habana_frameworks.torch.core as htcore
import json

class ParallelizedSlotAttention(nn.Module):
    """Parallelized Slot Attention for all tasks simultaneously"""
    def __init__(self, dim, num_slots, iters=3, eps=1e-8, hidden_dim=128, disease_embeddings=None, embedding_dim=None):
        super().__init__()
        self.num_slots = num_slots  # 236 tasks
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        
        # Initialize slots with disease embeddings if provided
        if disease_embeddings is not None and embedding_dim is not None:
            
            projection = nn.Sequential(
                nn.Linear(embedding_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU()
            )
            for param in projection.parameters():
                param.requires_grad = False
            
          
            projection.to(disease_embeddings.device)
            
            # 3. Project the embeddings
            with torch.no_grad():
                projected_embeddings = projection(disease_embeddings)
         

            slots_init_value = projected_embeddings.squeeze(1).unsqueeze(0)
        
            self.initial_slots = nn.Parameter(slots_init_value)
            
       

        else:
            # Random initialization as fallback
            self.initial_slots = nn.Parameter(torch.randn(1, num_slots, dim))
        
        # Shared attention components for all slots
        # self.to_q = nn.Linear(dim, dim)
        # self.to_k = nn.Linear(dim, dim)
        self.to_k = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        # self.to_v = nn.Linear(dim, dim)
        
        # GRU for slot updates - processes all slots in parallel
        self.gru = nn.GRUCell(dim, dim)
        
        # MLP for slot refinement
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
        
        # Layer norms
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)
        
    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size, num_patches, dim]
        Returns:
            slots: [batch_size, num_slots, dim] - all slot features
            attn: [batch_size, num_slots, num_patches] - attention weights for all slots
        """
        b, n, d = inputs.shape
        
        # Normalize inputs
        inputs = self.norm_input(inputs)
        
        # Initialize all slots: [b, num_slots, d]
        slots = self.initial_slots.expand(b, -1, -1)
        
        # Compute keys and values once for all slots
        k = self.to_k(inputs)  # [b, n, d]
        # v = self.to_v(inputs)  # [b, n, d]
        v = inputs  # Use inputs directly as values
        
        # Iterative attention
        for _ in range(self.iters):
            slots_prev = slots  # [b, num_slots, d]
            slots_norm = self.norm_slots(slots)  # [b, num_slots, d]
            
            # Compute queries for all slots
            # q = self.to_q(slots_norm)  # [b, num_slots, d]
            q = slots_norm
            
            # Parallel attention for all slots: [b, num_slots, num_patches]
            attn_logits = torch.einsum('bsd,bnd->bsn', q, k) * self.scale
            temperature = 10.0
            attn = F.softmax(attn_logits/temperature, dim=-1)
            # attn = torch.sigmoid(attn_logits)


          

            
            # Weighted mean for all slots: [b, num_slots, d]
            updates = torch.einsum('bsn,bnd->bsd', attn, v)
            
            # Update all slots with GRU in parallel
            # Reshape for GRUCell: [b*num_slots, d]
            updates_flat = updates.reshape(b * self.num_slots, d)
            slots_prev_flat = slots_prev.reshape(b * self.num_slots, d)
            
            slots_flat = self.gru(updates_flat, slots_prev_flat)
            slots = slots_flat.reshape(b, self.num_slots, d)
            
            # Residual MLP for all slots
            slots_mlp = self.norm_mlp(slots)  # [b, num_slots, d]
            slots = slots + self.mlp(slots_mlp)  # [b, num_slots, d]

      
        return updates, attn



class MedSIGLIPSlotAttention(nn.Module):
    """MedSIGLIP with Parallelized Slot Attention for multi-label classification"""
    def __init__(
        self, 
        model_name="google/medsiglip-448",
        num_tasks=236,
        num_classes_per_task=3,
        hidden_dim=128,
        slot_iters=3,
        freeze_backbone=True,
        lambda_attn_loss=0.01,
        reduced_dim=256,
        disease_embeddings_path=None 
    ):
        super().__init__()
        
        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task
        self.lambda_attn_loss = lambda_attn_loss
        
        # Load pretrained MedSIGLIP
        self.medsiglip = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.medsiglip.parameters():
                param.requires_grad = False
        
        # Get vision model output dimension
        self.vision_dim = 1152
        
        self.dim_reduction = nn.Sequential(
            nn.Linear(self.vision_dim, reduced_dim),
            nn.LayerNorm(reduced_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Load disease embeddings if path provided
        disease_embeddings = None
        embedding_dim = None
        if disease_embeddings_path is not None:
            disease_embeddings, embedding_dim = self.load_disease_embeddings(
                disease_embeddings_path, num_tasks
            )

        # Single parallelized slot attention for all tasks
        self.slot_attention = ParallelizedSlotAttention(
            dim=reduced_dim,
            num_slots=num_tasks,
            iters=slot_iters,
            hidden_dim=hidden_dim,
            disease_embeddings=disease_embeddings,
            embedding_dim=embedding_dim
        )
        
        # Single classification head that processes all slots in parallel
        # Instead of 236 separate classifiers, use batch matrix multiplication
        self.classifier = nn.Sequential(
            nn.Linear(reduced_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes_per_task)
        )

    def load_disease_embeddings(self, json_path, expected_num_tasks):
        """
        Load disease embeddings from JSON file
        
        JSON format expected:
        {
            "disease_0": [0.1, 0.2, ..., 0.5],  # embedding vector
            "disease_1": [0.3, 0.1, ..., 0.2],
            ...
        }
        
        Args:
            json_path: path to JSON file containing embeddings
            expected_num_tasks: number of expected diseases/tasks
        
        Returns:
            embeddings: torch.Tensor of shape [num_tasks, embedding_dim]
            embedding_dim: dimension of the embeddings
        """
        print(f"Loading disease embeddings from: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON formats
        if isinstance(data, dict):
            # Sort keys to ensure consistent ordering
            sorted_keys = sorted(data.keys())
            embeddings_list = [data[key] for key in sorted_keys]
        elif isinstance(data, list):
            embeddings_list = data
        else:
            raise ValueError("JSON must contain either a dict or list of embeddings")
        
        # Convert to tensor
        embeddings = torch.tensor(embeddings_list, dtype=torch.float32)
        
        # Validate shape
        # print("*"*100)
        # print(f"Loaded embeddings shape from JSON: {embeddings.shape}")
        # print("*"*100)
        num_loaded, _, embedding_dim = embeddings.shape
        if num_loaded != expected_num_tasks:
            raise ValueError(
                f"Number of embeddings in JSON ({num_loaded}) does not match "
                f"expected number of tasks ({expected_num_tasks})"
            )
        
        print(f"Loaded {num_loaded} disease embeddings with dimension {embedding_dim}")
        print("*"*100)
        
        return embeddings, embedding_dim 

    def forward(self, pixel_values, labels=None):
        """
        Args:
            pixel_values: preprocessed images [batch_size, 3, 448, 448]
            labels: [batch_size, num_tasks] - class labels for each task
        Returns:
            outputs: [batch_size, num_tasks, num_classes_per_task]
            loss: total loss if labels provided
        """
        batch_size = pixel_values.shape[0]
        
        # Extract vision features
        vision_outputs = self.medsiglip.vision_model(pixel_values)
        vision_features = vision_outputs.last_hidden_state  # [b, num_patches, dim]
        
        vision_features_reduced = self.dim_reduction(vision_features)
        
        # Apply parallelized slot attention for all tasks at once
        slots, attn = self.slot_attention(vision_features_reduced)
        # slots: [b, num_tasks, dim]
        # attn: [b, num_tasks, num_patches]
        
        # Classify all slots in parallel
        # Reshape to [b*num_tasks, dim], pass through classifier, then reshape back
        slots_flat = slots.reshape(batch_size * self.num_tasks, -1)
        logits_flat = self.classifier(slots_flat)  # [b*num_tasks, num_classes]
        outputs = logits_flat.reshape(batch_size, self.num_tasks, self.num_classes_per_task)
        
        
        return {'logits': outputs, 'attentions': attn}
    
    def predict(self, pixel_values, save_attention=True, save_dir='attention_maps', image_size=(448, 448)):
        """
        Get predictions for inference and optionally save attention maps
        
        Args:
            save_attention: whether to save attention maps as images
            save_dir: directory to save attention maps
            image_size: original image size for resizing attention maps
        
        Returns:
            predictions: [batch_size, num_tasks] - predicted class for each task
            probabilities: [batch_size, num_tasks, num_classes] - class probabilities
            attention_maps: [batch_size, num_tasks, num_patches] - raw attention weights
        """
        import os
        import numpy as np
        from PIL import Image
        import torch.nn.functional as F
        
        with torch.no_grad():
            outputs = self.forward(pixel_values)
            predictions = torch.argmax(outputs['logits'], dim=-1)  # [b, num_tasks]
            probabilities = F.softmax(outputs['logits'], dim=-1)  # [b, num_tasks, num_classes]
            attention_maps = outputs['attentions']  # [b, num_tasks, num_patches]
        
        if save_attention:
            os.makedirs(save_dir, exist_ok=True)
            batch_size = attention_maps.shape[0]
            num_patches = attention_maps.shape[-1]
            patch_grid_size = int(math.sqrt(num_patches))  # e.g., 32 for 1024 patches
            
            for batch_idx in range(batch_size):
                batch_dir = os.path.join(save_dir, f'batch_{batch_idx}')
                os.makedirs(batch_dir, exist_ok=True)
                
                for task_idx in range(self.num_tasks):
                    # Get attention map for this task: [num_patches]
                    attn = attention_maps[batch_idx, task_idx].cpu().numpy()
                    
                    # Reshape to 2D grid: [grid_h, grid_w]
                    attn_2d = attn.reshape(patch_grid_size, patch_grid_size)
                    
                    # Normalize to 0-255 range
                    attn_normalized = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)
                    attn_uint8 = (attn_normalized * 255).astype(np.uint8)
                    
                    # Create PIL image and resize to original image size
                    attn_img = Image.fromarray(attn_uint8, mode='L')
                    attn_img_resized = attn_img.resize(image_size, Image.BILINEAR)
                    
                    # Get prediction info
                    pred_class = predictions[batch_idx, task_idx].item()
                    pred_prob = probabilities[batch_idx, task_idx, pred_class].item()
                    
                    # Save attention map
                    filename = f'task_{task_idx:03d}_pred_{pred_class}_prob_{pred_prob:.3f}.png'
                    attn_img_resized.save(os.path.join(batch_dir, filename))
                
                print(f"Saved attention maps for batch {batch_idx} to {batch_dir}")
        
        return predictions, probabilities, attention_maps

# Usage Example
def train_step_example():
    """Example training step"""
    device='hpu'
    
    # Initialize model
    model = MedSIGLIPSlotAttention(
        num_tasks=236,
        num_classes_per_task=3,
        freeze_backbone=True,
        lambda_attn_loss=0.01
    ).to(device)
    
    # Prepare data
    processor = AutoProcessor.from_pretrained("google/medsiglip-448")
    
    # Dummy image and labels
    from PIL import Image
    img = Image.new('RGB', (448, 448))
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    
    # Dummy labels: [batch_size, num_tasks] with class indices
    labels = torch.randint(0, 3, (1, 236)).to(device)
    
    # Forward pass
    outputs = model(pixel_values, labels=labels)
    
    print(f"Logits shape: {outputs['logits'].shape}")  # [1, 236, 3]
    print(f"Total loss: {outputs['loss'].item():.4f}")
    print(f"CE loss: {outputs['ce_loss'].item():.4f}")
    print(f"Attention loss: {outputs['attn_loss'].item():.4f}")
    
    # Backward
    outputs['loss'].backward()
    
    return model


# Inference example
def inference_example():
    """Example inference with attention map saving"""
    device="hpu"
    
    model = MedSIGLIPSlotAttention(num_tasks=236).to(device)
    model.eval()
    
    processor = AutoProcessor.from_pretrained("google/medsiglip-448")
    img = Image.new('RGB', (448, 448))
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    
    # With saving attention maps
    predictions, probabilities, attention_maps = model.predict(
        pixel_values, 
        save_attention=True,
        save_dir='./attention_visualizations',
        image_size=(448, 448)
    )
    print(f"Predictions shape: {predictions.shape}")  # [1, 236]
    print(f"Probabilities shape: {probabilities.shape}")  # [1, 236, 3]
    print(f"Attention maps shape: {attention_maps.shape}")  # [1, 236, num_patches]
    print(f"\nAttention maps saved to './attention_visualizations/batch_0/'")
    print(f"Total files saved: {len(predictions[0])} (one per task)")


if __name__ == "__main__":
    print("Training step example:")
    train_step_example()
    print("\nInference example:")
    inference_example()