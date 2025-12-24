import os
import cv2
import torch
import csv
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


# --- 1. CONFIG ---
DATA_DIR = 'data'
PLAYER_MODEL = os.path.join(DATA_DIR, 'football-player-detection.pt')
BALL_MODEL = os.path.join(DATA_DIR, 'football-ball-detection.pt')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# --- 2. LOAD MOLMO AI ---
print("🧠 Loading Molmo-7B into A100...")
# CRITICAL: Disable flash attention which can cause cache issues
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
except:
    pass  # Ignore if not available

model_id = 'allenai/Molmo-7B-D-0924'
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
vlm_model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, device_map="auto", torch_dtype="auto" 
)
# CRITICAL: Disable cache at config level to prevent 340 vs 339 error
vlm_model.config.use_cache = False
vlm_model.eval()  # Ensure eval mode


def verify_pass_vlm(frame_crop):
    """
    Deep Fix for A100: Prevents tensor size mismatch (340 vs 339) and KeyError.
    Uses MANUAL generation loop to bypass KV cache and position_ids issues in transformers v4.51+
    """
    prompt = "Is this soccer player clearly kicking the ball? Answer only 'Pass' or 'Noise'."
    
    # Process inputs using Molmo's custom .process() method (NOT processor())
    inputs = processor.process(images=[Image.fromarray(frame_crop)], text=prompt)
    
    # Manually map all tensors to the model's device and add batch dimension
    processed_inputs = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            processed_inputs[key] = value.to(vlm_model.device).unsqueeze(0)
        else:
            processed_inputs[key] = value
    
    # MANUAL GENERATION LOOP - bypasses generate() which has position_ids issues
    with torch.inference_mode():
        torch.cuda.empty_cache()
        
        input_ids = processed_inputs['input_ids']
        batch_size, seq_len = input_ids.shape
        eos_token_id = processor.tokenizer.eos_token_id
        max_new_tokens = 5
        
        generated_ids = input_ids.clone()
        
        for step in range(max_new_tokens):
            current_seq_len = generated_ids.shape[1]
            
            # Build model inputs for this step
            model_inputs = {
                'input_ids': generated_ids,
                'position_ids': torch.arange(current_seq_len, device=vlm_model.device, dtype=torch.long).unsqueeze(0),
                'attention_mask': torch.ones(batch_size, current_seq_len, device=vlm_model.device, dtype=torch.long),
                'use_cache': False,  # CRITICAL: No cache
            }
            
            # Add vision inputs only on first step (they're already embedded in first pass)
            if step == 0:
                for key in processed_inputs:
                    if key not in ['input_ids', 'position_ids', 'attention_mask']:
                        model_inputs[key] = processed_inputs[key]
            
            # Forward pass
            outputs = vlm_model(**model_inputs)
            
            # Get next token (greedy decoding)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            
            # Stop if EOS
            if next_token_id.item() == eos_token_id:
                break
        
        torch.cuda.empty_cache()
    
    # Extract generated tokens (skip the input tokens)
    input_length = input_ids.shape[1]
    generated_tokens = generated_ids[0, input_length:]
    decoded_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).lower()
    
    return "pass" in decoded_text


# --- 3. MAIN ENGINE ---
def run_analysis(video_path):
    v_info = sv.VideoInfo.from_video_path(video_path)
    p_m = YOLO(PLAYER_MODEL).to(DEVICE)
    b_m = YOLO(BALL_MODEL).to(DEVICE)
    tracker = sv.ByteTrack()
    
    pass_events = []
    current_owner = None
    last_event_frame = -100

    print(f"🎬 Processing: {video_path}...")
    for f_idx, frame in enumerate(tqdm(sv.get_video_frames_generator(video_path), total=v_info.total_frames)):
        
        # Detection
        p_det = tracker.update_with_detections(sv.Detections.from_ultralytics(p_m(frame, imgsz=1280, verbose=False)[0]))
        b_det = sv.Detections.from_ultralytics(b_m(frame, imgsz=640, verbose=False)[0])
        
        # Logic
        ball_coords = b_det.get_anchors_coordinates(sv.Position.CENTER)
        ball_xy = ball_coords[0] if len(ball_coords) > 0 else None
        
        if ball_xy is not None and p_det.tracker_id is not None:
            for tid, p_xy in zip(p_det.tracker_id, p_det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)):
                if np.linalg.norm(ball_xy - p_xy) < 45:
                    if current_owner is not None and tid != current_owner and (f_idx - last_event_frame > 20):
                        # VLM Verification
                        crop = frame[max(0, int(ball_xy[1])-150):int(ball_xy[1])+150, max(0, int(ball_xy[0])-150):int(ball_xy[0])+150]
                        if crop.size > 0 and verify_pass_vlm(crop):
                            pass_events.append({"time": round(f_idx/v_info.fps, 2), "from": int(current_owner), "to": int(tid)})
                            last_event_frame = f_idx
                    current_owner = tid
                    break
        
        # VRAM SAFETY: Flush pending memory every 200 frames to prevent overflow
        if f_idx % 200 == 0 and f_idx > 0:
            torch.cuda.empty_cache()

    # Save CSV
    with open('verified_scout_report.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["time", "from", "to"])
        writer.writeheader()
        writer.writerows(pass_events)
    print(f"✅ Finished! Total Passes: {len(pass_events)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='input_video/video_segment.mp4', help='Path to video file')
    args = parser.parse_args()
    run_analysis(args.source)