import sys
import os
import asyncio
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from mcp_kokoro import get_pipeline, _speak_sync

def verify_fix():
    print("Verifying fix...")
    pipe = get_pipeline()
    if not pipe:
        print("Failed to get pipeline (timeout?)")
        # Try waiting a bit
        import time
        for _ in range(10):
            time.sleep(1)
            pipe = get_pipeline()
            if pipe: break
        
    if not pipe:
        print("Still failed to get pipeline")
        return

    text = "Hello! This verification confirms that the code now handles the updated Kokoro library correctly."
    voice = "af_heart"
    speed = 1.0

    print(f"Generating audio via _speak_sync for: '{text}'")
    
    try:
        # We can't easily mock sounddevice in this script without complex mocking, 
        # so this might try to play audio. 
        # However, we mostly care that it doesn't crash.
        # Ideally we would mock sounddevice so it doesn't play sound on the CI/agent environment,
        # but the user environment is macOS likely with audio.
        # Let's mock sounddevice to be safe and silent.
        import unittest.mock
        with unittest.mock.patch('sounddevice.play') as mock_play, \
             unittest.mock.patch('sounddevice.wait') as mock_wait:
            
            error = _speak_sync(text, voice, speed, pipe)
            
            if error:
                print(f"Error returned: {error}")
            else:
                print("Success! Audio generation completed without error.")
                print("Mock play called:", mock_play.called)
                
    except Exception as e:
        print(f"CRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_fix()
