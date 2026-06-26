#!/usr/bin/env python3
"""
Dashboard backend server to manage drone detector processes.
Provides HTTP endpoints to start/stop acoustic, RF, and camera detectors.
"""

import os
import sys
import json
import subprocess
import signal
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Process tracking
processes = {
    'acoustic': None,
    'rf': None,
    'camera': None,
}

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests for status and control."""
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        
        # CORS headers
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {'success': False, 'message': ''}
        
        # Status endpoint
        if path == '/status':
            response['success'] = True
            response['processes'] = {
                'acoustic': processes['acoustic'] is not None,
                'rf': processes['rf'] is not None,
                'camera': processes['camera'] is not None,
            }
        
        # Start acoustic detector
        elif path == '/start/acoustic':
            if processes['acoustic'] is None:
                try:
                    processes['acoustic'] = subprocess.Popen(
                        ['sudo', 'python3', 'acoustic_drone_detector.py', '--audio-device', '0'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        preexec_fn=os.setsid
                    )
                    response['success'] = True
                    response['message'] = 'Acoustic detector started'
                except Exception as e:
                    response['message'] = f'Failed to start: {str(e)}'
            else:
                response['success'] = True
                response['message'] = 'Acoustic detector already running'
        
        # Stop acoustic detector
        elif path == '/stop/acoustic':
            if processes['acoustic'] is not None:
                try:
                    os.killpg(os.getpgid(processes['acoustic'].pid), signal.SIGTERM)
                    processes['acoustic'].wait(timeout=5)
                    processes['acoustic'] = None
                    response['success'] = True
                    response['message'] = 'Acoustic detector stopped'
                except Exception as e:
                    response['message'] = f'Failed to stop: {str(e)}'
            else:
                response['success'] = True
                response['message'] = 'Acoustic detector not running'
        
        # Start RF scanner
        elif path == '/start/rf':
            if processes['rf'] is None:
                try:
                    processes['rf'] = subprocess.Popen(
                        ['python3', 'rf_drone_scanner.py', '--iface', 'wlan1'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        preexec_fn=os.setsid
                    )
                    response['success'] = True
                    response['message'] = 'RF scanner started'
                except Exception as e:
                    response['message'] = f'Failed to start: {str(e)}'
            else:
                response['success'] = True
                response['message'] = 'RF scanner already running'
        
        # Stop RF scanner
        elif path == '/stop/rf':
            if processes['rf'] is not None:
                try:
                    os.killpg(os.getpgid(processes['rf'].pid), signal.SIGTERM)
                    processes['rf'].wait(timeout=5)
                    processes['rf'] = None
                    response['success'] = True
                    response['message'] = 'RF scanner stopped'
                except Exception as e:
                    response['message'] = f'Failed to stop: {str(e)}'
            else:
                response['success'] = True
                response['message'] = 'RF scanner not running'
        
        # Start camera detector
        elif path == '/start/camera':
            if processes['camera'] is None:
                try:
                    # In /start/camera block (around line 117-122)
                    processes['camera'] = subprocess.Popen(
                        ['python3', 'detect_drones_jetson_working.py', 
                        '--source', '0', 
                        '--tensorrt', 
                        '--fp16',
                        '--ws-port', '8765'],   # ← ADD THIS
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        preexec_fn=os.setsid
                    )
                    response['success'] = True
                    response['message'] = 'Camera detector started'
                except Exception as e:
                    response['message'] = f'Failed to start: {str(e)}'
            else:
                response['success'] = True
                response['message'] = 'Camera detector already running'
        
        # Stop camera detector
        elif path == '/stop/camera':
            if processes['camera'] is not None:
                try:
                    os.killpg(os.getpgid(processes['camera'].pid), signal.SIGTERM)
                    processes['camera'].wait(timeout=5)
                    processes['camera'] = None
                    response['success'] = True
                    response['message'] = 'Camera detector stopped'
                except Exception as e:
                    response['message'] = f'Failed to stop: {str(e)}'
            else:
                response['success'] = True
                response['message'] = 'Camera detector not running'
        
        else:
            response['message'] = 'Unknown endpoint'
        
        self.wfile.write(json.dumps(response).encode())
    
    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

if __name__ == '__main__':
    port = 9000
    server = HTTPServer(('localhost', port), DashboardHandler)
    print(f'Dashboard backend server running on http://localhost:{port}')
    print('Available endpoints:')
    print('  GET /status')
    print('  GET /start/acoustic | /stop/acoustic')
    print('  GET /start/rf | /stop/rf')
    print('  GET /start/camera | /stop/camera')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nShutting down...')
        for name, proc in processes.items():
            if proc is not None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except:
                    pass
        sys.exit(0)
