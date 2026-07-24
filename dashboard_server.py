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
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import time
import threading
from datetime import datetime

# Import the get_audio_devices helper from the acoustic detector module
try:
    from acoustic_drone_detector import get_audio_devices
except ImportError:
    def get_audio_devices():
        return []

# SSE clients: list of dicts { 'wfile': wfile, 'lock': threading.Lock() }
sse_clients: list[dict] = []
sse_lock = threading.Lock()


def broadcast_event(obj: dict):
    """Send a JSON event to all connected EventSource clients."""
    s = json.dumps(obj)
    dead = []
    with sse_lock:
        for entry in list(sse_clients):
            w = entry.get('wfile')
            l = entry.get('lock')
            try:
                with l:
                    w.write(f"data: {s}\n\n".encode())
                    w.flush()
            except Exception:
                dead.append(entry)
        for d in dead:
            try:
                sse_clients.remove(d)
            except ValueError:
                pass

# Process tracking
processes = {
    'acoustic': None,
    'rf': None,
    'camera': None,
}

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests for status, control, and camera MJPEG stream."""
        parsed = urlparse(self.path)
        path = parsed.path
        
        # ── MJPEG camera stream (long-lived connection, different headers) ──
        if path == '/camera/stream':
            return self._handle_camera_stream()
        
        # ── Server-Sent Events for live detections ──
        if path == '/events':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()

            # Register client
            client = {'wfile': self.wfile, 'lock': threading.Lock()}
            with sse_lock:
                sse_clients.append(client)

            try:
                # Keep the connection open; block until client disconnects
                while True:
                    time.sleep(1)
                    try:
                        # send a heartbeat comment to keep some proxies happy
                        with client['lock']:
                            client['wfile'].write(b": heartbeat\n\n")
                            client['wfile'].flush()
                    except Exception:
                        break
            finally:
                with sse_lock:
                    try:
                        sse_clients.remove(client)
                    except ValueError:
                        pass
            return
        
        query = parse_qs(parsed.query)
        
        # CORS headers for JSON endpoints
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
        
        # List available audio devices
        elif path == '/audio-devices':
            try:
                devices = get_audio_devices()
                response['success'] = True
                response['devices'] = devices
            except Exception as e:
                response['success'] = False
                response['message'] = f'Failed to list audio devices: {str(e)}'
        
        # Start acoustic detector
        elif path == '/start/acoustic':
            if processes['acoustic'] is None:
                try:
                    # Allow user to specify audio device via query param: ?device=1
                    # and microphone direction via ?mic_direction=0
                    device_index = query.get('device', [None])[0]
                    mic_direction = query.get('mic_direction', [None])[0]
                    cmd = ['python3', 'acoustic_drone_detector.py']
                    if device_index is not None:
                        cmd.extend(['--audio-device', device_index])
                    if mic_direction is not None:
                        cmd.extend(['--mic-direction', mic_direction])
                    processes['acoustic'] = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,  # Suppress ALSA library debug noise
                        preexec_fn=os.setsid
                    )
                    response['success'] = True
                    response['message'] = 'Acoustic detector started'
                    # Start a thread to forward JSON lines from the acoustic process stdout
                    def _reader(proc, name):
                        try:
                            for raw in iter(proc.stdout.readline, b''):
                                try:
                                    line = raw.decode(errors='replace').strip()
                                except Exception:
                                    continue
                                if not line:
                                    continue
                                # Try to parse JSON; otherwise forward as a LOG event
                                try:
                                    obj = json.loads(line)
                                    broadcast_event(obj)
                                except Exception:
                                    try:
                                        evt = {'type': 'LOG', 'source': name.upper(), 'message': line, 'timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}
                                        broadcast_event(evt)
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                    threading.Thread(target=_reader, args=(processes['acoustic'], 'acoustic'), daemon=True).start()
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
                        stderr=subprocess.STDOUT,
                        preexec_fn=os.setsid
                    )
                    response['success'] = True
                    response['message'] = 'RF scanner started'
                    # Thread to forward JSON lines from RF scanner stdout
                    def _reader_rf(proc, name):
                        try:
                            for raw in iter(proc.stdout.readline, b''):
                                try:
                                    line = raw.decode(errors='replace').strip()
                                except Exception:
                                    continue
                                if not line:
                                    continue
                                try:
                                    obj = json.loads(line)
                                    broadcast_event(obj)
                                except Exception:
                                    try:
                                        evt = {'type': 'LOG', 'source': name.upper(), 'message': line, 'timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}
                                        broadcast_event(evt)
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                    threading.Thread(target=_reader_rf, args=(processes['rf'], 'rf'), daemon=True).start()
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
                    _camera_log = open('/tmp/drone_camera_detector.log', 'w')
                    _camera_log.write('=== Camera detector started ===\n')
                    _camera_log.flush()
                    processes['camera'] = subprocess.Popen(
                        ['python3', 'detect_drones_jetson_working.py', 
                        '--source', '0', 
                        '--no-display'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        preexec_fn=os.setsid
                    )
                    # Start a thread to log subprocess output in real-time
                    def _log_camera_output(proc, logfile):
                        for line in iter(proc.stdout.readline, b''):
                            logfile.write(line.decode(errors='replace'))
                            logfile.flush()
                        logfile.close()
                    t = threading.Thread(
                        target=_log_camera_output,
                        args=(processes['camera'], _camera_log),
                        daemon=True
                    )
                    t.start()
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
    
    def _handle_camera_stream(self):
        """Serve the latest detection frame as an MJPEG stream.

        Reads the JPEG file written by detect_drones_jetson_working.py and pushes
        it to the browser repeatedly, keeping the connection open with the
        'multipart/x-mixed-replace' content type.
        """
        import time
        import logging
        frame_path = Path('/tmp/drone_detect_frame.jpg')

        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        self.send_header('Connection', 'keep-alive')
        self.end_headers()

        last_mtime = 0
        sent_first = False
        stall_count = 0
        logging.info('MJPEG stream client connected')
        try:
            
            while True:
                if processes['camera'] is not None and processes['camera'].poll() is not None:
                    logging.error('Camera detector process has exited (code=%s). Stream ending.',processes['camera'].poll())
                    break
                if frame_path.exists():
                    mtime = frame_path.stat().st_mtime
                    if mtime > last_mtime:
                        last_mtime = mtime
                        data = frame_path.read_bytes()
                        if len(data) > 100:  # Sanity check: valid JPEG
                            self.wfile.write(b'--FRAME\r\n')
                            self.wfile.write(b'Content-Type: image/jpeg\r\n')
                            self.wfile.write(f'Content-Length: {len(data)}\r\n'.encode())
                            self.wfile.write(b'\r\n')
                            self.wfile.write(data)
                            self.wfile.write(b'\r\n')
                            self.wfile.flush()
                            if not sent_first:
                                sent_first = True
                                logging.info('MJPEG stream: first frame sent')
                            stall_count = 0
                        else:
                            stall_count += 1
                    else:
                        stall_count += 1
                else:
                    if not sent_first:
                        stall_count += 1
                        if stall_count % 30 == 0:  # Log every ~1 second
                            logging.info(
                                'MJPEG stream waiting for frame at %s ... '
                                '(start the camera detector from the dashboard)',
                                frame_path
                            )
                time.sleep(0.033)  # ~30 fps
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            logging.info('MJPEG stream client disconnected: %s', e)

    def _monitor_processes():
        while True:
            for name in list(processes.keys()):
                proc = processes[name]
                if proc is not None and proc.poll() is not None:
                    logging.warning('%s detector process exited (rc=%s)', name, proc.poll())
                    processes[name] = None
            time.sleep(2)

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
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    port = 9000
    try:
        server = ThreadingHTTPServer(('localhost', port), DashboardHandler)
    except NameError:
        # Fallback for Python < 3.7
        from http.server import HTTPServer
        server = HTTPServer(('localhost', port), DashboardHandler)
    print(f'Dashboard backend server running on http://localhost:{port}')
    print('Available endpoints:')
    print('  GET /status')
    print('  GET /audio-devices')
    print('  GET /start/acoustic | /stop/acoustic')
    print('  GET /start/rf | /stop/rf')
    print('  GET /start/camera | /stop/camera')
    print('  GET /camera/stream')
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
