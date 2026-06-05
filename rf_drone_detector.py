#!/usr/bin/env python3
"""
RF Drone Detection using USB WiFi Adapter (Monitor Mode + Scapy)
For NVIDIA Jetson Nano / Jetson series.
Detects common drone WiFi signatures (DJI, Parrot, etc.).
"""

import argparse
import time
import threading
import queue
import subprocess
from datetime import datetime
from collections import deque
import os

try:
    from scapy.all import sniff, Dot11, Dot11Beacon, Dot11ProbeReq, Dot11ProbeResp
except ImportError:
    print("❌ Scapy not installed. Run: pip install scapy")
    print("Also: sudo apt install aircrack-ng")
    exit(1)

class RFDroneDetector:
    def __init__(self, monitor_iface="wlan0mon", channel_hop=True):
        self.monitor_iface = monitor_iface
        self.channel_hop = channel_hop
        self.running = False
        self.thread = None
        self.detections = queue.Queue(maxsize=50)
        
        # Common drone signatures
        self.drone_ssids = [
            "DJI_", "Phantom", "Mavic", "Air", "Mini", "Parrot", "Bebop",
            "Anafi", "FPV_", "drone", "RC_", "Sky", "Tello"
        ]
        # Known drone MAC OUIs (first 3 bytes / 6 hex chars)
        self.drone_ouis = [
            "60:60", "00:0c", "90:3a", "e4:83", "00:26",  # DJI / Parrot common
        ]
        
        self.recent_drones = deque(maxlen=20)  # For deduplication
        self.stats = {"beacons": 0, "probes": 0, "drones": 0}

    def setup_monitor_mode(self):
        """Prepare the USB WiFi adapter for monitor mode."""
        print(f"🔧 Setting up monitor mode on {self.monitor_iface}...")
        try:
            # Kill interfering processes
            subprocess.run(["sudo", "airmon-ng", "check", "kill"], check=True)
            
            # Start monitor mode
            result = subprocess.run(["sudo", "airmon-ng", "start", self.monitor_iface.replace("mon", "")],
                                  capture_output=True, text=True)
            print(result.stdout)
            
            # Verify
            iw = subprocess.run(["iwconfig"], capture_output=True, text=True)
            if "Mode:Monitor" in iw.stdout or self.monitor_iface in iw.stdout:
                print(f"✅ Monitor mode active on {self.monitor_iface}")
                return True
            else:
                print("⚠️  Could not confirm monitor mode. Check adapter support.")
                return False
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            print("Make sure your USB WiFi adapter supports monitor mode (e.g. Alfa AWUS036 series).")
            return False

    def _packet_handler(self, pkt):
        """Process captured 802.11 packets for drone signatures."""
        if not pkt.haslayer(Dot11):
            return

        try:
            mac = None
            ssid = None
            is_drone = False

            # Beacon frames (AP mode - common for drones)
            if pkt.haslayer(Dot11Beacon):
                self.stats["beacons"] += 1
                ssid = pkt[Dot11Beacon].info.decode(errors='ignore').strip()
                mac = pkt.addr2

            # Probe requests/responses (client mode)
            elif pkt.haslayer(Dot11ProbeReq) or pkt.haslayer(Dot11ProbeResp):
                self.stats["probes"] += 1
                if pkt.haslayer(Dot11ProbeReq):
                    ssid = pkt[Dot11ProbeReq].info.decode(errors='ignore').strip()
                mac = pkt.addr2 or pkt.addr1

            if mac:
                mac_upper = mac.upper()
                # Check OUI
                oui = mac_upper[:8]  # e.g., "60:60:xx"
                if any(oui.startswith(o) for o in self.drone_ouis):
                    is_drone = True

                # Check SSID
                if ssid and any(d in ssid.upper() for d in self.drone_ssids):
                    is_drone = True

                if is_drone:
                    detection = {
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "mac": mac,
                        "ssid": ssid or "Hidden/Unknown",
                        "type": "Beacon" if pkt.haslayer(Dot11Beacon) else "Probe",
                        "signal": pkt.dBm_AntSignal if hasattr(pkt, 'dBm_AntSignal') else "N/A"
                    }
                    
                    # Deduplicate
                    key = f"{mac}_{ssid}"
                    if key not in self.recent_drones:
                        self.recent_drones.append(key)
                        self.detections.put(detection)
                        self.stats["drones"] += 1
                        print(f"🚁 DRONE DETECTED! MAC: {mac} | SSID: {ssid} | Signal: {detection['signal']}")
        except:
            pass  # Ignore malformed packets

    def _channel_hopper(self):
        """Hop channels for broader coverage (2.4 GHz focus)."""
        channels = list(range(1, 12))  # 2.4 GHz
        idx = 0
        while self.running:
            try:
                subprocess.run(["sudo", "iwconfig", self.monitor_iface, "channel", str(channels[idx])],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                idx = (idx + 1) % len(channels)
                time.sleep(1.5)  # Hop every ~1.5s
            except:
                time.sleep(2)

    def start(self):
        if not self.setup_monitor_mode():
            return False

        self.running = True
        
        # Start channel hopper if enabled
        if self.channel_hop:
            hopper = threading.Thread(target=self._channel_hopper, daemon=True)
            hopper.start()

        # Start sniffing in background thread
        self.thread = threading.Thread(target=self._sniff_loop, daemon=True)
        self.thread.start()
        
        print("🚀 RF Drone Detector Started! Press Ctrl+C to stop.")
        return True

    def _sniff_loop(self):
        try:
            sniff(iface=self.monitor_iface, prn=self._packet_handler, store=False, monitor=True)
        except Exception as e:
            if self.running:
                print(f"Sniff error: {e}")

    def get_detections(self):
        dets = []
        while not self.detections.empty():
            dets.append(self.detections.get())
        return dets

    def print_stats(self):
        print(f"\n📊 Stats - Beacons: {self.stats['beacons']} | Probes: {self.stats['probes']} | Drones: {self.stats['drones']}")


def main():
    parser = argparse.ArgumentParser(description="RF Drone Detector (WiFi Adapter)")
    parser.add_argument("--iface", default="wlan0mon", help="Monitor interface (default: wlan0mon)")
    parser.add_argument("--no-hop", action="store_true", help="Disable channel hopping")
    parser.add_argument("--duration", type=int, default=0, help="Run for N seconds (0 = infinite)")
    args = parser.parse_args()

    detector = RFDroneDetector(monitor_iface=args.iface, channel_hop=not args.no_hop)
    
    if not detector.start():
        exit(1)

    try:
        start_time = time.time()
        while True:
            time.sleep(5)
            detector.print_stats()
            
            if args.duration > 0 and (time.time() - start_time) > args.duration:
                break
    except KeyboardInterrupt:
        print("\n⏹️ Stopping detector...")
    finally:
        detector.running = False
        print("✅ Detector stopped. Cleanup: sudo airmon-ng stop <interface>")


if __name__ == "__main__":
    main()