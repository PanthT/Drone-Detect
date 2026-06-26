#!/usr/bin/env python3
"""Simple WebSocket camera streamer.

Sends MJPEG-like JPEG frames as binary messages over a websocket so the
browser dashboard can render live video without HTTP streaming setup.

Usage:
    python3 web_cam_ws_server.py --source 0 --fps 20

Dependencies: opencv-python, websockets
    pip3 install opencv-python websockets
"""
import asyncio
import argparse
import cv2
import time
import logging

import websockets

logging.basicConfig(level=logging.INFO)


async def camera_handler(websocket, path, source, fps):
    logging.info("Client connected for camera stream")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logging.error("Could not open camera source %s", source)
        await websocket.close()
        return

    try:
        interval = 1.0 / max(1.0, fps)
        while True:
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.05)
                continue

            # Resize moderately to reduce bandwidth (browser canvas will scale)
            h, w = frame.shape[:2]
            max_w = 640
            if w > max_w:
                scale = max_w / float(w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            # Encode as JPEG and send raw bytes
            ok, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue
            data = jpg.tobytes()
            try:
                await websocket.send(data)
            except Exception:
                break

            elapsed = time.time() - start
            to_wait = interval - elapsed
            if to_wait > 0:
                await asyncio.sleep(to_wait)
    finally:
        cap.release()
        logging.info("Client disconnected from camera stream")


async def main_async(args):
    server = await websockets.serve(
        lambda ws, path: camera_handler(ws, path, args.source, args.fps),
        host=args.host,
        port=args.port,
        max_size=2**20,
    )
    logging.info("Camera WS server listening on ws://%s:%s", args.host, args.port)
    await server.wait_closed()


def main():
    parser = argparse.ArgumentParser(description="WebSocket camera streamer")
    parser.add_argument('--source', type=int, default=0, help='cv2.VideoCapture source (default 0)')
    parser.add_argument('--fps', type=int, default=20, help='Frames per second to stream')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind websocket')
    parser.add_argument('--port', type=int, default=8765, help='Port to bind websocket')
    args = parser.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
