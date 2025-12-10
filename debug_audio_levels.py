import sys
import os
import asyncio
import numpy as np
import torch

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from mcp_kokoro import get_pipeline

def test_audio_levels():
    print("Testing audio levels...")
    pipe = get_pipeline()
    if not pipe:
        print("Failed to get pipeline")
        return

    text = "Hello! This is a test to check for audio clipping and distortion."
    voice = "af_heart"
    speed = 1.0

    print(f"Generating audio for: '{text}'")
    
    max_val = 0.0
    min_val = 0.0
    
    # Run the generator
    iterator = pipe(text, voice=voice, speed=speed)
    
    for item in iterator:
        print(f"Yielded item type: {type(item)}")
        
        # Check for Kokoro 0.9.0+ Result object (KPipeline.Result)
        if hasattr(item, 'graphemes') and hasattr(item, 'output'):
             print("Detected KPipeline.Result object")
             # Extract audio from output
             if hasattr(item.output, 'audio'):
                 audio = item.output.audio
                 print(f"Audio found, type: {type(audio)}")
                 
                 if isinstance(audio, torch.Tensor):
                     audio = audio.detach().cpu().numpy()
                 
                 if len(audio) > 0:
                     chunk_max = np.max(audio)
                     chunk_min = np.min(audio)
                     
                     print(f"Chunk stats: Range [{chunk_min:.4f}, {chunk_max:.4f}]")
                     
                     max_val = max(max_val, chunk_max)
                     min_val = min(min_val, chunk_min)
                 else:
                     print("Audio is empty")
             else:
                 print("No audio in output")

        # Fallback to tuple check (older versions)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            G, ps = item

            print(f"Got G, ps (G type: {type(G)})")
            
            # G is likely the generator for audio chunks
            for i, chunk_data in enumerate(G):
                # chunk_data should be (gs, ps, audio)
                if len(chunk_data) == 3:
                    gs, ps_chunk, audio = chunk_data
                    
                    if isinstance(audio, torch.Tensor):
                        audio = audio.detach().cpu().numpy()
                    
                    chunk_max = np.max(audio)
                    chunk_min = np.min(audio)
                    
                    print(f"Chunk {i}: Range [{chunk_min:.4f}, {chunk_max:.4f}]")
                    
                    max_val = max(max_val, chunk_max)
                    min_val = min(min_val, chunk_min)
                else:
                    print(f"Unexpected chunk data length: {len(chunk_data)}")
        else:
            print(f"Unexpected item structure: {item}")

    print(f"\nOverall Stats:")
    print(f"Max Value: {max_val}")
    print(f"Min Value: {min_val}")
    
    if max_val > 1.0 or min_val < -1.0:
        print("CLIPPING DETECTED!")
    else:
        print("Levels look safe.")

if __name__ == "__main__":
    import time
    for _ in range(20):
        if get_pipeline():
            break
        print("Waiting for pipeline...")
        time.sleep(1)
        
    test_audio_levels()
