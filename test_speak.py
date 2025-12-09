import asyncio
import sys
from mcp_kokoro import speak

async def test():
    print("Testing speak tool...")
    result = await speak("Hello! This is a test of the Kokoro Text to Speech system.")
    print(result)

if __name__ == "__main__":
    try:
        asyncio.run(test())
    except Exception as e:
        print(f"Test failed: {e}")
