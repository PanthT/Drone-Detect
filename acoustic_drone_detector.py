#!/usr/bin/env python3
"""
RF Drone Detection using USB WiFi Adapter (Monitor Mode + Scapy)
Optimized for NVIDIA Jetson Nano.
"""

import argparse
import time
import threading
import queue
import subprocess
from datetime import datetime
from collections import deque
import sys

try:
    from scapy.all import sniff, Dot11, Dot11Beacon, Dot11ProbeReq, Dot11ProbeResp
except ImportError:
    print("❌ Scapy not installed.")
    print("Run: pip3 install scapy")
    print("Also: sudo apt update && sudo apt install aircrack-ng")
    sys.exit(1)


class RFDroneDetector:
    def __init__(self, monitor_iface="wlan0mon", channel_hop=True):
        self.monitor_iface = monitor_iface
        self.channel_hop = channel_hop
        self.running = False
        self.thread = None
        self.detections = queue.Queue(maxsize=50)
        
        self.drone_ssids = ["DJI_", "Phantom", "Mavic", "Air", "Mini", "Parrot", 
                           "Bebop", "Anafi", "FPV_", "drone", "RC_", "Sky", "Tello"]
        self.drone_ouis = ["60:60", "00:0C", "90:3A", "E4:83", "00:26", "18:64"]  # Expanded
        
        self.recent_drones = deque(maxlen=30)
        self.stats = {"beacons": 0, "probes": 0, "drones": 0}

    def setup_monitor_mode(self):
        print(f"🔧 Setting up monitor mode on {self.monitor_iface}...")
        try:
            # Kill NetworkManager and interfering processes
            subprocess.run(["sudo", "airmon-ng", "check", "kill"], check=True, stderr=subprocess.DEVNULL)
            
            base_iface = self.monitor_iface.replace("mon", "")
            subprocess.run(["sudo", "airmon-ng", "start", base_iface], 
                         capture_output=True, text=True, check=True)
            
            # Verify
            iw_out = subprocess.check_output(["iwconfig"], text=True)
            if "Mode:Monitor" in iw_out or self.monitor_iface in iw_out:
                print(f"✅ Monitor mode ready on {self.monitor_iface}")
                return True
        except Exception as e:
            print(f"❌ Monitor mode setup failed: {e}")
            print("💡 Try manually: sudo airmon-ng start wlan0")
            return False
        return False

    def _packet_handler(self, pkt):
        if not pkt.haslayer(Dot11):
            return

        try:
            mac = None
            ssid = None
            is_drone = False

            if pkt.haslayer(Dot11Beacon):
                self.stats["beacons"] += 1
                ssid = pkt[Dot11Beacon].info.decode(errors='ignore').strip()
                mac = pkt.addr2
            elif pkt.haslayer(Dot11ProbeReq) or pkt.haslayer(Dot11ProbeResp):
                self.stats["probes"] += 1
                if pkt.haslayer(Dot11ProbeReq):
                    ssid = pkt[Dot11ProbeReq].info.decode(errors='ignore').strip()
                mac = pkt.addr2 or pkt.addr1

            if mac:
                mac_upper = mac.upper()
                oui = mac_upper[:8]
                if any(oui.startswith(o) for o in self.drone_ouis):
                    is_drone = True
                if ssid and any(d in ssid.upper() for d in self.drone_ssids):
                    is_drone = True

                if is_drone:
                    detection = {
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "mac": mac,
                        "ssid": ssid or "Unknown",
                        "type": "Beacon" if pkt.haslayer(Dot11Beacon) else "Probe",
                        "signal": getattr(pkt, 'dBm_AntSignal', 'N/A')
                    }
                    key = f"{mac}_{ssid}"
                    if key not in self.recent_drones:
                        self.recent_drones.append(key)
                        self.detections.put(detection)
                        self.stats["drones"] += 1
                        print(f"🚁 DRONE DETECTED! {mac} | {ssid} | Signal: {detection['signal']}")
        except:
            pass

    def _channel_hopper(self):
        channels = list(range(1, 14))
        idx = 0
        while self.running:
            try:
                subprocess.run(["sudo", "iwconfig", self.monitor_iface, "channel", str(channels[idx])],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                idx = (idx + 1) % len(channels)
                time.sleep(1.2)
            except:
                time.sleep(2)

    def start(self):
        if not self.setup_monitor_mode():
            return False

        self.running = True
        if self.channel_hop:
            threading.Thread(target=self._channel_hopper, daemon=True).start()

        self.thread = threading.Thread(target=self._sniff_loop, daemon=True)
        self.thread.start()
        print("🚀 RF Drone Detector running... (Ctrl+C to stop)")
        return True

    def _sniff_loop(self):
        try:
            sniff(iface=self.monitor_iface, prn=self._packet_handler, store=False, monitor=True)
        except Exception as e:
            if self.running:
                print(f"Sniff error: {e}")

    def print_stats(self):
        print(f"📊 Beacons: {self.stats['beacons']} | Probes: {self.stats['probes']} | Drones: {self.stats['drones']}")


def main():
    parser = argparse.ArgumentParser(description="RF Drone Detector")
    parser.add_argument("--iface", default="wlan0mon", help="Monitor interface")
    parser.add_argument("--no-hop", action="store_true", help="Disable channel hopping")
    args = parser.parse_args()

    detector = RFDroneDetector(monitor_iface=args.iface, channel_hop=not args.no_hop)
    
    if not detector.start():
        sys.exit(1)

    try:
        while True:
            time.sleep(4)
            detector.print_stats()
    except KeyboardInterrupt:
        print("\n⏹️  Stopping...")
    finally:
        detector.running = False
        print("✅ Stopped. Cleanup: sudo airmon-ng stop wlan0mon")


if __name__ == "__main__":
    main()