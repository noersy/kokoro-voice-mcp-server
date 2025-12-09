# mcp-kokoro

An MCP server for [Kokoro TTS](https://github.com/hexgrad/kokoro), enabling high-quality text-to-speech capabilities for MCP clients.

## Overview

This server provides tools to generate spoken audio from text using the Kokoro model. It is designed to be used with the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), allowing AI agents to "speak" to the user.

## Tools

### `speak`

Generates audio from text and plays it immediately.

- **Arguments**:
  - `text` (str): The text to speak.
  - `voice` (str, optional): The voice to use (default: `af_heart`).
  - `speed` (float, optional): Speaking speed (default: `1.0`).

### `ask_approval`

A semantic wrapper around `speak` specifically for requesting user confirmation.

- **Arguments**:
  - `request_text` (str): The action or request requiring approval.

### `announce_task`

A semantic wrapper around `speak` for providing status updates on tasks.

- **Arguments**:
  - `task_name` (str): Name of the task.
  - `status` (str, optional): Status of the task (default: `completed`).

## Installation

### Using `uv` (Recommended)

```bash
uv tool install mcp-kokoro
```

### Using `pip`

```bash
pip install mcp-kokoro
```

## Configuration

To use with Claude Desktop or other MCP clients, add the following to your configuration file (e.g., `~/Library/Application Support/Claude/claude_desktop_config.json`):

### uv

```json
{
  "mcpServers": {
    "kokoro": {
      "command": "uv",
      "args": [
        "tool",
        "run",
        "mcp-kokoro"
      ]
    }
  }
}
```

### pip

```json
{
  "mcpServers": {
    "kokoro": {
      "command": "mcp-kokoro",
      "args": []
    }
  }
}
```

## Requirements

- Python 3.10 or higher
- Audio output device (for playback)
- `sounddevice` system dependencies (e.g., PortAudio) may be required on some systems.
  - macOS: `brew install portaudio`
  - Linux: `sudo apt-get install libportaudio2`

## License

[MIT](LICENSE)
