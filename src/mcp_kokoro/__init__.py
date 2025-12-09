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

def _speak_sync(text: str, voice: str, speed: float, pipeline):
    """Synchronous function to handle audio generation and playback."""
    try:
        # Lazy imports are fast enough if main packages are loaded in background
        import sounddevice as sd
        
        print(f"Generating audio for: '{text}'", file=sys.stderr)
        
        # Generator returns (graphemes, phonemes, audio)
        # We only need the audio
        # Redirect stdout during generation too, just in case
        with StdoutRedirector():
            generator = pipeline(
                text, 
                voice=voice, 
                speed=speed, 
                split_pattern=r'\n+'
            )
            
            for i, (gs, ps, audio) in enumerate(generator):
                if audio is not None:
                    print(f"Playing segment {i+1}...", file=sys.stderr)
                    sd.play(audio, 24000) # Kokoro usually defaults to 24khz
                    sd.wait()
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

@mcp.tool()
async def ask_approval(request_text: str) -> str:
    """
    Ask for user approval audibly. This is a semantic wrapper around speak.
    
    Args:
        request_text (str): The request to ask approval for.
    """
    prompt = f"Attention required. {request_text}. Do you approve?"
    return await speak(prompt, speed=1.1) 

@mcp.tool()
async def announce_task(task_name: str, status: str = "completed") -> str:
    """
    Announce a task update.
    
    Args:
        task_name (str): The name of the task.
        status (str): The status (e.g., 'completed', 'failed', 'started').
    """
    text = f"Task {task_name} has {status}."
    return await speak(text)

def main():
    mcp.run()

if __name__ == "__main__":
    main()
