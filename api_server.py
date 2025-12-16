from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import asyncio
import json
from pathlib import Path
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import os

load_dotenv()
print(f"🔑 API Key loaded: {os.getenv('GEMINI_API_KEY', 'NOT FOUND')[:20]}...", file=sys.stderr)

app = Flask(__name__)
CORS(app)

WORKFLOWS_DIR = Path(__file__).parent / "data" / "workflows"

mcp_session = None
exit_stack = None
mcp_loop = None
mcp_thread = None
executor = ThreadPoolExecutor(max_workers=5)


def run_async_in_thread(coro):
    future = asyncio.run_coroutine_threadsafe(coro, mcp_loop)
    return future.result(timeout=90)


async def init_mcp_session():
    global mcp_session, exit_stack
    
    try:
        exit_stack = AsyncExitStack()
        
        import os
        env_path = Path(__file__).parent / '.env'
        if not env_path.exists():
            api_key = os.getenv('GEMINI_API_KEY')
            model = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
            if api_key:
                with open(env_path, 'w') as f:
                    f.write(f"GEMINI_API_KEY={api_key}\n")
                    f.write(f"GEMINI_MODEL={model}\n")
                print(f"📝 Created .env file for subprocess", file=sys.stderr)
        
        log_file = Path(__file__).parent / "server_logs.txt"
        
        server_params = StdioServerParameters(
            command="python",
            args=["-u", "server.py"],
        )
        
        print(f"📝 Server logs will be written to: {log_file}", file=sys.stderr)
        
        print("🔌 Connecting to MCP server...", file=sys.stderr)
        stdio_transport = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        
        read_stream, write_stream = stdio_transport
        mcp_session = await exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        
        await mcp_session.initialize()
        print("✅ MCP server connected!", file=sys.stderr)
        return True
    except Exception as e:
        print(f"❌ Failed to connect to MCP server: {e}", file=sys.stderr)
        return False


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'mcp_connected': mcp_session is not None
    })


@app.route('/api/generate', methods=['POST'])
def generate_workflow():
    global mcp_session
    
    if not mcp_session:
        return jsonify({
            'success': False,
            'error': 'MCP server not connected'
        }), 503
    
    data = request.get_json()
    description = data.get('description', '')
    domain = data.get('domain', 'general')
    
    if not description:
        return jsonify({
            'success': False,
            'error': 'Description is required'
        }), 400
    
    try:
        print(f"\n🎯 [API] Generate workflow request:", file=sys.stderr)
        print(f"   Description: {description[:80]}...", file=sys.stderr)
        print(f"   Domain: {domain}", file=sys.stderr)
        
        async def generate():
            print(f"   📞 Calling MCP tool: generate_workflow_spec", file=sys.stderr)
            result = await mcp_session.call_tool(
                "generate_workflow_spec",
                {
                    "description": description,
                    "domain": domain
                }
            )
            return json.loads(result.content[0].text)
        
        response_data = run_async_in_thread(generate())
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error generating workflow: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/validate', methods=['POST'])
def validate_workflow():
    global mcp_session
    
    if not mcp_session:
        return jsonify({
            'success': False,
            'error': 'MCP server not connected'
        }), 503
    
    workflow = request.get_json()
    
    try:
        async def validate():
            result = await mcp_session.call_tool(
                "validate_workflow",
                {"workflow_json": workflow}
            )
            return json.loads(result.content[0].text)
        
        response_data = run_async_in_thread(validate())
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error validating workflow: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/export', methods=['POST'])
def export_workflow():
    global mcp_session
    
    if not mcp_session:
        return jsonify({
            'success': False,
            'error': 'MCP server not connected'
        }), 503
    
    data = request.get_json()
    workflow = data.get('workflow')
    format_type = data.get('format', 'json')
    
    try:
        async def export():
            result = await mcp_session.call_tool(
                "export_to_format",
                {
                    "workflow_json": workflow,
                    "format": format_type
                }
            )
            return result.content[0].text
        
        content = run_async_in_thread(export())
        return content, 200, {'Content-Type': 'text/plain'}
        
    except Exception as e:
        print(f"Error exporting workflow: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/')
def index():
    return send_from_directory(WORKFLOWS_DIR, 'premium.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(WORKFLOWS_DIR, path)


def run_mcp_loop():
    global mcp_loop, mcp_session
    
    mcp_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(mcp_loop)
    
    success = mcp_loop.run_until_complete(init_mcp_session())
    
    if not success:
        print("\n❌ Failed to start: MCP server connection failed", file=sys.stderr)
        sys.exit(1)
    
    mcp_loop.run_forever()


def main():
    global mcp_thread
    
    print("=" * 70)
    print("🚀 WORKFLOW GENERATOR API SERVER")
    print("=" * 70)
    
    mcp_thread = threading.Thread(target=run_mcp_loop, daemon=True)
    mcp_thread.start()
    
    import time
    for i in range(50):
        time.sleep(0.1)
        if mcp_session:
            break
    
    if not mcp_session:
        print("\n❌ Failed to start: MCP server connection failed")
        print("   Make sure server.py is working correctly")
        sys.exit(1)
    
    print(f"\n✅ API server starting on http://localhost:5000")
    print(f"📂 Serving frontend from: {WORKFLOWS_DIR.absolute()}")
    print(f"\n🌐 Open in browser: http://localhost:5000")
    print("\n⏹️  Press Ctrl+C to stop")
    print("=" * 70)
    
    try:
        print("🌐 Flask server starting...", file=sys.stderr)
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
    except Exception as e:
        print(f"❌ Flask error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        print("Shutting down...", file=sys.stderr)
        if mcp_loop:
            mcp_loop.call_soon_threadsafe(mcp_loop.stop)


if __name__ == "__main__":
    main()