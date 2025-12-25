import http.server
import socketserver
import os
from pathlib import Path

PORT = 8080
DIRECTORY = Path(__file__).parent.parent / "client"

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super().end_headers()

if __name__ == "__main__":
    os.chdir(DIRECTORY)
    
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        print("=" * 70)
        print("🌐 WORKFLOW VIEWER SERVER")
        print("=" * 70)
        print(f"\n✅ Server running at: http://localhost:{PORT}")
        print(f"📂 Serving files from: {DIRECTORY.absolute()}")
        print(f"\n🚀 Open in browser: http://localhost:{PORT}/premium.html")
        print("\n⏹️  Press Ctrl+C to stop")
        print("=" * 70)
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n🛑 Server stopped")
