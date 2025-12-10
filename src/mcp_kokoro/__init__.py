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

def initialize_pipeline():
    global pipeline
    # Initialize pipeline synchronously before server start
    # This prevents race conditions with stdout redirection and ensures
    # the server is fully ready (or fails early) before accepting connections.
    print("Initializing Kokoro pipeline...", file=sys.stderr)
    try:
        # Redirect stdout to stderr to capture all output including library prints
        # This is safe here because the MCP server hasn't started listening/writing yet.
        with contextlib.redirect_stdout(sys.stderr):
            from kokoro import KPipeline
            
            device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
            print(f"Using device: {device}", file=sys.stderr)
            
            # We explicitly pass repo_id to suppress the warning "Defaulting repo_id to..."
            pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', device=device)
            
        print("Kokoro pipeline initialized successfully.", file=sys.stderr)
    except Exception as e:
        print(f"Error initializing Kokoro pipeline: {e}", file=sys.stderr)
        # We don't exit here, allowing the server to start, but 'speak' will fail later if pipeline is None


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
            
            for i, result in enumerate(generator):
                # Compatibility layer: Handle both KPipeline.Result objects and legacy tuples
                audio = None
                
                # Check for KPipeline.Result (v0.9.4+)
                if hasattr(result, 'output') and hasattr(result.output, 'audio'):
                    audio = result.output.audio
                
                # Check for legacy tuple (v0.3.0) -> (graphemes, phonemes, audio)
                elif isinstance(result, (list, tuple)) and len(result) == 3:
                    _, _, audio = result
                
                if audio is not None:
                    # Ensure audio is numpy array (MPS/CUDA returns wrappers or tensors)
                    if isinstance(audio, torch.Tensor):
                        audio = audio.detach().cpu().numpy()
                    
                    # Normalize audio if clipping is detected (prevents "crounch")
                    if len(audio) > 0:
                        max_val = np.max(np.abs(audio))
                        if max_val > 1.0:
                            audio = audio / max_val * 0.99
                    
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

    if pipeline is None:
        return "Error: Pipeline failed to initialize during server startup. Check logs for details."

    # Run the blocking generation/playback in a separate thread
    # Clean text: replace newlines with spaces to avoid TTS issues
    text = text.replace('\n', ' ')
    
    error = await asyncio.to_thread(_speak_sync, text, voice, speed, pipeline)
    
    if error:
        return f"Error speaking text: {str(error)}"
        
    return "Speech completed successfully."


def main():
    initialize_pipeline()
    mcp.run()

if __name__ == "__main__":
    main()
