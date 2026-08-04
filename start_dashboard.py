#!/usr/bin/env python3
"""
Startup script for Drone Detection Dashboard
Starts dashboard_server.py and opens the HTML file in your default browser
"""

import subprocess
import time
import webbrowser
import os
import sys
from pathlib import Path

# Suppress GTK/Canberra warnings on Linux
os.environ['GTK_DEBUG'] = ''
os.environ['G_MESSAGES_DEBUG'] = 'none'

def main():
    print("=" * 70)
    print("🚀 Drone Detection Dashboard Startup")
    print("=" * 70)
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    html_file = script_dir / "drone_dashboard.html"
    
    # Check if HTML file exists
    if not html_file.exists():
        print(f"❌ Error: HTML file not found at {html_file}")
        sys.exit(1)
    
    print(f"\n📁 Working directory: {script_dir}")
    print(f"📄 HTML file: {html_file}")
    
    # Start dashboard_server.py in background
    print("\n▶️  Starting dashboard_server.py...")
    try:
        # Start the server subprocess
        server_proc = subprocess.Popen(
            [sys.executable, 'dashboard_server.py'],
            cwd=script_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"✓ Server process started (PID: {server_proc.pid})")
    except Exception as e:
        print(f"❌ Failed to start dashboard_server.py: {e}")
        sys.exit(1)
    
    # Wait for server to initialize
    print("\n⏳ Waiting for server to initialize...")
    time.sleep(2)
    
    # Check if server is running
    try:
        import requests
        for attempt in range(10):
            try:
                response = requests.get('http://localhost:9000/status', timeout=1)
                if response.status_code == 200:
                    print("✓ Server is ready!")
                    break
            except:
                if attempt < 9:
                    print(f"  Attempt {attempt + 1}/10...")
                    time.sleep(0.5)
                else:
                    print("⚠️  Server may not have started correctly")
    except ImportError:
        print("⚠️  requests library not available, skipping server check")
    
    # Open HTML file in default browser
    html_url = f"file://{html_file.absolute()}"
    print(f"\n🌐 Opening dashboard in browser...")
    print(f"   URL: {html_url}")
    
    try:
        # Suppress GTK warnings on Linux
        devnull = open(os.devnull, 'w')
        old_stderr = sys.stderr
        sys.stderr = devnull
        webbrowser.open(html_url)
        sys.stderr = old_stderr
        devnull.close()
        print("✓ Browser opened")
    except Exception as e:
        print(f"⚠️  Could not open browser: {e}")
        print(f"   Manually open: {html_url}")
    
    print("\n" + "=" * 70)
    print("✓ Dashboard started successfully!")
    print("=" * 70)
    print("\n📊 Dashboard running at: file://" + str(html_file.absolute()))
    print("🔌 Backend server running at: http://localhost:9000")
    print("\n💡 To stop: Press Ctrl+C in this terminal")
    print("=" * 70 + "\n")
    
    # Keep the server running
    try:
        server_proc.wait()
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print("✓ Server stopped")
        sys.exit(0)

if __name__ == '__main__':
    main()
