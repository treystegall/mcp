# How to use Anthropic MCP Server with open LLMs, OpenAI or Google Gemini

This repository contains a basic example of how to build an AI agent using the Model Context Protocol (MCP) with an open LLM (Meta Llama 3), OpenAI or Google Gemini, and a SQLite database. It's designed to be a simple, educational demonstration, not a production-ready framework.

OpenAI example: https://github.com/jalr4ever/Tiny-OAI-MCP-Agent


## Setup

This code sets up a simple CLI agent that can interact with a SQLite database through an MCP server. It uses the official SQLite MCP server and demonstrates:

*   Connecting to an MCP server
*   Loading and using tools and resources from the MCP server
*   Converting tools into LLM-compatible function calls
*   Interacting with an LLM using the `openai` SDK or `google-genai` SDK.

## How to use it

*   Docker installed and running.
*   Hugging Face account and an access token (for using the Llama 3 model).
*   Google API key (for using the Gemini model).

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/philschmid/mcp-openai-gemini-llama-example
    cd mcp-openai-gemini-llama-example
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Log in to Hugging Face
    ```bash
    huggingface-cli login --token YOUR_TOKEN
    ```

## Examples

### Llama 3
   
Run the following command

```bash
python sqlite_llama_mcp_agent.py
```

The agent will start in interactive mode. You can type in prompts to interact with the database. Type "quit", "exit" or "q" to stop the agent.

Example conversation:
```
Enter your prompt (or 'quit' to exit): what tables are available?

Response:  The available tables are: albums, artists, customers, employees, genres, invoice_items, invoices, media_types, playlists, playlist_track, tracks

Enter your prompt (or 'quit' to exit): how many artists are there

Response:  There are 275 artists in the database.
```

### Gemini

Run the following command

```bash
GOOGLE_API_KEY=YOUR_API_KEY python sqlite_gemini_mcp_agent.py
```

## Future plans

I'm working on a toolkit to make implementing AI agents using MCP easier. Stay tuned for updates!
