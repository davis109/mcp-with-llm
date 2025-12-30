# Workflow Generator

AI-powered workflow blueprint generator using Gemini AI and Model Context Protocol (MCP).

## Overview

This system generates structured workflow blueprints from natural language descriptions. It provides a modern web interface for creating, visualizing, and exporting workflows in multiple formats (JSON, YAML, BPMN).

## Features

- 🧠 **AI-Powered Generation**: Converts text descriptions into structured workflows
- 🎨 **Modern Web Interface**: Clean, intuitive UI for workflow creation
- 📤 **Multiple Export Formats**: JSON, YAML, and BPMN support
- 🔄 **Real-time Processing**: Instant workflow generation and preview
- 🛠️ **MCP Integration**: Built on Model Context Protocol for extensibility

## Prerequisites

**For Docker:**
- Docker and Docker Compose installed
- Gemini API key (get one at [Google AI Studio](https://aistudio.google.com/))

**For Local Python:**
- Python 3.11 or higher
- Gemini API key (get one at [Google AI Studio](https://aistudio.google.com/))

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Key**
   
   Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

3. **Verify Configuration**
   
   Ensure `config/data.json` exists with required settings.

## Running

### Option 1: Docker (Recommended)

**Build the image first:**
```bash
docker build -t workflow-generator .
```

**Using Docker Compose (modern Docker/cloud environments):**
```bash
docker compose up -d
```

**Or legacy docker-compose (if above doesn't work):**
```bash
docker-compose up -d
```

**Alternative: Using docker run (for cloud environments without compose):**
```bash
docker run -d \
  --name workflow-generator \
  -p 5000:5000 \
  -e GEMINI_API_KEY=your_api_key_here \
  -e GEMINI_MODEL=gemini-2.5-flash \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/client:/app/client \
  -v $(pwd)/server:/app/server \
  --restart unless-stopped \
  workflow-generator
```

Then open your browser to: **http://localhost:5000**

**Stop the application:**
```bash
# If using compose:
docker compose down

# If using docker run:
docker stop workflow-generator
docker rm workflow-generator
```

**View logs:**
```bash
# If using compose:
docker compose logs -f

# If using docker run:
docker logs -f workflow-generator
```

**Rebuild after changes:**
```bash
# If using compose:
docker compose up --build -d

# If using docker run:
docker stop workflow-generator
docker rm workflow-generator
docker build -t workflow-generator .
# Then run the docker run command again
```

### Option 2: Local Python

**Start the application:**
```bash
start.bat
```

Or manually:
```bash
python server\api_server.py
```

Then open your browser to: **http://localhost:5000**

## File Structure

```
neu/
├── server/                # Server components
│   ├── server.py         # MCP server backend
│   ├── api_server.py     # REST API server
│   └── server_web.py     # Static file server
├── client/                # Frontend files
│   ├── index.html        # Web interface
│   ├── script.js         # Frontend logic
│   └── styles.css        # Styles
├── config/
│   └── data.json         # Configuration
├── data/
│   └── workflows/        # Generated workflows
├── start.bat             # Startup script
└── requirements.txt      # Python dependencies
```

## Usage

1. **Launch the application** using `start.bat`
2. **Open the web interface** at http://localhost:5000
3. **Enter a workflow description** in natural language
4. **Select the domain** (HR, IT, Customer Service, etc.)
5. **Generate** and view your workflow
6. **Export** in your preferred format (JSON, YAML, BPMN)

### Example Prompts

```
"Create an employee onboarding workflow with HR approval and IT setup steps"

"Design a customer support ticket escalation process with SLA checks"

"Build a purchase approval workflow with budget validation and multi-level approval"
```

## Architecture

- **MCP Server** (`server.py`): Handles workflow generation logic using Gemini AI
- **REST API** (`api_server.py`): Bridges web interface and MCP server
- **Web Frontend**: Modern HTML/JS interface for workflow creation

## API Key

The system requires a Google Gemini API key:
1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Create an API key
3. Add it to your `.env` file

## Output

Generated workflows are saved to `data/workflows/` in your selected format:
- `workflow_name.json` - JSON format
- `workflow_name.yaml` - YAML format  
- `workflow_name.bpmn` - BPMN XML format

## Troubleshooting

**"GEMINI_API_KEY not found"**
- Ensure `.env` file exists in project root
- Verify API key is correctly set

**"Port 5000 already in use"**
- Stop other applications using port 5000
- Or edit `api_server.py` to use a different port

**"Module not found"**
- Run `pip install -r requirements.txt`

## License

MIT License
