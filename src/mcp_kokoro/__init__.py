import asyncio
import os
import sys
import threading
import torch
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("Kokoro TTS")

# Global pipeline state
pipeline = None
pipeline_lock = threading.Lock()
pipeline_loading_thread = None

import contextlib

def _create_pipeline():
    from kokoro import KPipeline
    # Initialize pipeline for US English
    # Redirect stdout to stderr to catch any library prints (like tqdm or warnings)
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}", file=sys.stderr)
    
    # Use contextlib to redirect stdout to stderr to capture all output including library prints
    with contextlib.redirect_stdout(sys.stderr):
        # We explicitly pass repo_id to suppress the warning "Defaulting repo_id to..."
        return KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', device=device)

def _load_pipeline_background():
    global pipeline
    print("Starting background initialization of Kokoro pipeline...", file=sys.stderr)
    try:
        p = _create_pipeline()
        
        with pipeline_lock:
            pipeline = p
        print("Kokoro pipeline initialized successfully in background.", file=sys.stderr)
    except Exception as e:
        print(f"Error initializing Kokoro pipeline in background: {e}", file=sys.stderr)

def start_background_loading():
    global pipeline_loading_thread
    if pipeline_loading_thread is None:
        pipeline_loading_thread = threading.Thread(target=_load_pipeline_background, daemon=True)
        pipeline_loading_thread.start()

# Start loading immediately on import
start_background_loading()

def get_pipeline():
    global pipeline
    # Wait for the background thread to finish if it hasn't already
    if pipeline is None:
        if pipeline_loading_thread and pipeline_loading_thread.is_alive():
            print("Waiting for background initialization to complete...", file=sys.stderr)
            pipeline_loading_thread.join()
        
        # If still None (thread failed or didn't run), try loading synchronously as fallback
        with pipeline_lock:
            if pipeline is None:
                print("Pipeline not ready, initializing synchronously...", file=sys.stderr)
                pipeline = _create_pipeline()
            
    return pipeline

import queue
import numpy as np
import hashlib
from pathlib import Path

# Cache directory configuration
CACHE_DIR = Path.home() / ".cache" / "mcp_kokoro"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _get_cache_path(text: str, voice: str, speed: float) -> Path:
    """Generate a unique file path for the given inputs."""
    content_id = f"{text}|{voice}|{speed}"
    file_hash = hashlib.sha256(content_id.encode('utf-8')).hexdigest()
    return CACHE_DIR / f"{file_hash}.npy"

def _speak_sync(text: str, voice: str, speed: float, pipeline):
    """Synchronous function to handle audio generation and playback with persistent file caching."""
    try:
        # Lazy imports
        import sounddevice as sd
        
        cache_path = _get_cache_path(text, voice, speed)
        
        # 1. Check Disk Cache
        if cache_path.exists():
            print(f"Disk cache hit for: '{text[:20]}...'", file=sys.stderr)
            try:
                audio = np.load(cache_path)
                sd.play(audio, 24000)
                sd.wait()
                return None
            except Exception as e:
                print(f"Failed to load cache file: {e}", file=sys.stderr)
                # Fallthrough to regeneration if cache load fails

        print(f"Generating audio for: '{text[:20]}...'", file=sys.stderr)
        
        # 2. Setup Queue for Gapless Playback
        q = queue.Queue()
        playback_error = None
        
        def playback_worker():
            nonlocal playback_error
            try:
                while True:
                    audio_chunk = q.get()
                    if audio_chunk is None:
                        q.task_done()
                        break
                    
                    # Play blockingly in this thread
                    sd.play(audio_chunk, 24000)
                    sd.wait()
                    q.task_done()
            except Exception as e:
                playback_error = e

        # Start consumer thread
        t = threading.Thread(target=playback_worker, daemon=True)
        t.start()

        full_audio_pieces = []
        
        # 3. Generate Audio
        with contextlib.redirect_stdout(sys.stderr):
            # Generator: Process and yield audio
            # Using a more granular split pattern (split on newlines OR sentence endings)
            # to allow for faster time-to-first-audio on long texts
            generator = pipeline(
                text, 
                voice=voice, 
                speed=speed, 
                split_pattern=r'\n+|(?<=[.!?])\s+'
            )
            
            for i, (gs, ps, audio) in enumerate(generator):
                if audio is not None:
                    # Ensure audio is numpy array (MPS/CUDA returns wrappers or tensors)
                    if isinstance(audio, torch.Tensor):
                        audio = audio.detach().cpu().numpy()
                    
                    # Producer: Push to queue
                    q.put(audio.copy()) 
                    full_audio_pieces.append(audio)
            
            # Signal end of stream
            q.put(None)
        
        # Wait for playback to finish
        t.join()

        if playback_error:
            raise playback_error

        # 4. Save to Disk Cache
        if full_audio_pieces:
            try:
                full_audio = np.concatenate(full_audio_pieces)
                np.save(cache_path, full_audio)
                print(f"Saved audio to disk cache: {cache_path}", file=sys.stderr)
            except Exception as e:
                print(f"Failed to save to disk cache: {e}", file=sys.stderr)

        return None
    except Exception as e:
        return e

@mcp.tool()
async def speak(text: str, voice: str = "af_heart", speed: float = 1.0) -> str:
    """
    Speak the provided text using Kokoro TTS.
    
    Args:
        text (str): The text to speak.
        voice (str): The voice to use (default: 'af_heart'). Options often include 'af_bella', 'af_sarah', 'am_adam', 'af_heart', etc.
        speed (float): Speaking speed (default: 1.0).
    """
    if not text or not text.strip():
        return "Speech completed successfully."

    pipe = get_pipeline()
    if not pipe:
        return "Error: Pipeline functionality failed to initialize."

    # Run the blocking generation/playback in a separate thread
    error = await asyncio.to_thread(_speak_sync, text, voice, speed, pipe)
    
    if error:
        return f"Error speaking text: {str(error)}"
        
    return "Speech completed successfully."


def main():
    mcp.run()

if __name__ == "__main__":
    main()
