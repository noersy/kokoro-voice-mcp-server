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

class StdoutRedirector:
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = sys.stderr
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout

def _load_pipeline_background():
    global pipeline
    print("Starting background initialization of Kokoro pipeline...", file=sys.stderr)
    try:
        from kokoro import KPipeline
        # Initialize pipeline for US English
        # Redirect stdout to stderr to catch any library prints (like tqdm or warnings)
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {device}", file=sys.stderr)
        
        with StdoutRedirector():
            p = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', device=device)
        
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
                from kokoro import KPipeline
                device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
                print(f"Using device: {device}", file=sys.stderr)
                with StdoutRedirector():
                    pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', device=device)
            
    return pipeline

import queue
import numpy as np

# Cache for generated audio: key=(text, voice, speed) -> value=audio_array
audio_cache = {}
MAX_CACHE_SIZE = 100

def _speak_sync(text: str, voice: str, speed: float, pipeline):
    """Synchronous function to handle audio generation and playback with caching and gapless queuing."""
    try:
        # Lazy imports
        import sounddevice as sd
        
        cache_key = (text, voice, speed)
        
        # 1. Check Cache
        if cache_key in audio_cache:
            print(f"Cache hit for: '{text[:20]}...'", file=sys.stderr)
            sd.play(audio_cache[cache_key], 24000)
            sd.wait()
            return None

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
        with StdoutRedirector():
            generator = pipeline(
                text, 
                voice=voice, 
                speed=speed, 
                split_pattern=r'\n+'
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

        # 4. Update Cache (LRU-ish: Clear if full)
        if len(audio_cache) >= MAX_CACHE_SIZE:
            audio_cache.clear() # Simple wipe strategy to avoid complex dep
            print("Cache cleared.", file=sys.stderr)
        
        if full_audio_pieces:
            audio_cache[cache_key] = np.concatenate(full_audio_pieces)

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
    pipe = get_pipeline()
    if not pipe:
        return "Error: Pipeline functionality failed to initialize."

    # Run the blocking generation/playback in a separate thread
    error = await asyncio.to_thread(_speak_sync, text, voice, speed, pipe)
    
    if error:
        return f"Error speaking text: {str(error)}"
        
    return f"Successfully spoke: {text}"


def main():
    mcp.run()

if __name__ == "__main__":
    main()
