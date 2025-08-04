#!/usr/bin/env python3
import http.server
import socketserver
import json
import urllib.parse
import os
from pathlib import Path

class PushUpHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('index.html', 'r') as f:
                self.wfile.write(f.read().encode())
        elif self.path == '/start':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'success': True, 'message': 'Push-up counter would start (OpenCV not available in this environment)'}
            self.wfile.write(json.dumps(response).encode())
        elif self.path == '/stop':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'success': True, 'message': 'Push-up counter stopped'}
            self.wfile.write(json.dumps(response).encode())
        elif self.path == '/stats':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            # Mock stats for demonstration
            response = {
                'count': 0,
                'percentage': 0,
                'session_time': 0,
                'is_running': False
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            super().do_GET()

if __name__ == "__main__":
    PORT = 8000
    with socketserver.TCPServer(("", PORT), PushUpHandler) as httpd:
        print(f"Server running at http://localhost:{PORT}")
        print("Note: This is a demo version. OpenCV and MediaPipe are not available in this environment.")
        print("To run the full version with camera support, install the required packages locally:")
        print("pip install flask opencv-python mediapipe numpy")
        httpd.serve_forever()