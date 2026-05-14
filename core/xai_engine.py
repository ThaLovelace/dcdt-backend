"""
core/xai_engine.py
------------------
Explainable AI (XAI) engine for the dCDT ViT pipeline.

Generates a Chefer Transformer Interpretability heatmap over the last 
attention block of a 1-Logit ViT-B/16 model, overlays it on the 
preprocessed drawing image, and returns a base-64 encoded PNG.

Responsibilities
~~~~~~~~~~~~~~~~
1. ``disable_fused_attention`` - Disables SDPA to allow attention map extraction.
2. ``_run_chefer_single``      - Computes relevancy map using gradients and attention.
3. ``overlay_heatmap``         - Visually blends the heatmap onto the drawing.
"""

from __future__ import annotations
import base64
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm

# ---------------------------------------------------------------------------
# CRITICAL FIX: Disable fused SDPA in timm
# ---------------------------------------------------------------------------
def disable_fused_attention(model: nn.Module) -> nn.Module:
    """
    Manually overrides the forward pass of attention blocks to prevent
    PyTorch 2.x from using Flash Attention, which hides the attention weights.
    """
    def manual_attention_forward(self, x, attn_mask=None, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        scale = (C // self.num_heads) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            attn = attn + attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        self._attn_weights = attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    import types
    for module in model.modules():
        if type(module).__name__ == 'Attention' and hasattr(module, 'qkv'):
            module.forward = types.MethodType(manual_attention_forward, module)
    return model

# ---------------------------------------------------------------------------
# XAI Logic: Chefer (1-Logit Output)
# ---------------------------------------------------------------------------
def _run_chefer_single(model: nn.Module, input_tensor: torch.Tensor, target_class: int, device: str) -> np.ndarray:
    """
    Extracts gradient-weighted attention maps (Chefer method) for a 1-class output.
    """
    input_tensor = input_tensor.to(device)
    attn_list = []
    
    def hook(module, inp, out):
        if hasattr(module, '_attn_weights'):
            a = module._attn_weights
            a.retain_grad()
            attn_list.append(a)

    hooks = [b.attn.register_forward_hook(hook) for b in model.blocks]
    logits = model(input_tensor)
    
    for h in hooks: 
        h.remove()
    
    model.zero_grad()

    # Gradient computation for 1-Logit model
    if logits.shape[-1] == 1:
        if target_class == 1:
            logits[0, 0].backward(retain_graph=False)
        else:
            (-logits[0, 0]).backward(retain_graph=False)

    num_tokens = attn_list[0].shape[-1]
    R = torch.eye(num_tokens)
    for attn in attn_list:
        g = attn.grad
        if g is None: continue
        a_cpu = attn.detach().cpu()
        g_cpu = g.detach().cpu()
        cam = (g_cpu * a_cpu).mean(dim=1)[0]
        cam = cam + torch.eye(num_tokens)
        cam = cam / (cam.sum(-1, keepdim=True) + 1e-8)
        R = cam @ R

    mask = R[0, 1:].clamp(min=0).numpy()
    n = int(mask.shape[0] ** 0.5)
    mask = mask.reshape(n, n)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return cv2.resize(mask, (224, 224), interpolation=cv2.INTER_CUBIC)

def overlay_heatmap(img_pil, hmap, colormap=cv2.COLORMAP_JET, alpha=0.6, threshold=0.3, contour=True) -> np.ndarray:
    """
    Blends the raw Chefer mask over the original image.
    """
    img  = np.array(img_pil.resize((224, 224)))
    bgr  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h    = hmap.copy()
    h[h < threshold] = 0
    hu8  = (h * 255).astype(np.uint8)
    col  = cv2.applyColorMap(hu8, colormap)
    out  = cv2.addWeighted(bgr, 1 - alpha, col, alpha, 0)
    
    if contour:
        _, bin_ = cv2.threshold(hu8, int(threshold * 255), 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (255, 255, 255), 1)
    
    return out

# ---------------------------------------------------------------------------
# Model Loader & Output Generator
# ---------------------------------------------------------------------------
def map_torchvision_to_timm(tv_state_dict):
    if 'cls_token' in tv_state_dict: return tv_state_dict
    timm_dict = {}
    timm_dict['cls_token']              = tv_state_dict['class_token']
    timm_dict['pos_embed']              = tv_state_dict['encoder.pos_embedding']
    timm_dict['patch_embed.proj.weight']= tv_state_dict['conv_proj.weight']
    timm_dict['patch_embed.proj.bias']  = tv_state_dict['conv_proj.bias']
    timm_dict['norm.weight']            = tv_state_dict['encoder.ln.weight']
    timm_dict['norm.bias']              = tv_state_dict['encoder.ln.bias']
    timm_dict['head.weight']            = tv_state_dict['heads.head.weight']
    timm_dict['head.bias']              = tv_state_dict['heads.head.bias']
    use_long = 'encoder.layers.encoder_layer_0.ln_1.weight' in tv_state_dict
    for i in range(12):
        tv = f'encoder.layers.{"encoder_layer_" if use_long else ""}{i}.'
        tm = f'blocks.{i}.'
        timm_dict[f'{tm}norm1.weight']     = tv_state_dict[f'{tv}ln_1.weight']
        timm_dict[f'{tm}norm1.bias']       = tv_state_dict[f'{tv}ln_1.bias']
        timm_dict[f'{tm}norm2.weight']     = tv_state_dict[f'{tv}ln_2.weight']
        timm_dict[f'{tm}norm2.bias']       = tv_state_dict[f'{tv}ln_2.bias']
        timm_dict[f'{tm}attn.qkv.weight']  = tv_state_dict[f'{tv}self_attention.in_proj_weight']
        timm_dict[f'{tm}attn.qkv.bias']    = tv_state_dict[f'{tv}self_attention.in_proj_bias']
        timm_dict[f'{tm}attn.proj.weight'] = tv_state_dict[f'{tv}self_attention.out_proj.weight']
        timm_dict[f'{tm}attn.proj.bias']   = tv_state_dict[f'{tv}self_attention.out_proj.bias']
        timm_dict[f'{tm}mlp.fc1.weight']   = tv_state_dict[f'{tv}mlp.0.weight']
        timm_dict[f'{tm}mlp.fc1.bias']     = tv_state_dict[f'{tv}mlp.0.bias']
        timm_dict[f'{tm}mlp.fc2.weight']   = tv_state_dict[f'{tv}mlp.3.weight']
        timm_dict[f'{tm}mlp.fc2.bias']     = tv_state_dict[f'{tv}mlp.3.bias']
    return timm_dict

def load_vit_model(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """Instantiates the timm ViT model and loads pre-trained weights."""
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=1)
    ckpt = torch.load(checkpoint_path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt.get("model", ckpt))) if isinstance(ckpt, dict) else ckpt
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    mapped_sd = map_torchvision_to_timm(sd)
    model.load_state_dict(mapped_sd, strict=True)
    
    model.eval().to(device)
    model = disable_fused_attention(model)
    return model

def generate_xai_b64(model: nn.Module, input_tensor: torch.Tensor, img_pil, pred_cls: int, device: str) -> str:
    """Runs Chefer XAI and returns the overlaid image as a Base64 PNG string."""
    model.zero_grad()
    hmap_chefer = _run_chefer_single(model, input_tensor, target_class=pred_cls, device=device)
    ov_chefer = overlay_heatmap(img_pil, hmap_chefer)

    # Encode to Base64
    _, buffer = cv2.imencode('.png', ov_chefer)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"