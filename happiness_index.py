import cv2
import asyncio
import base64
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from deepface import DeepFace

app = FastAPI()

# --- Face detection ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# --- Shared variables ---
connected_clients = set()
latest_frame = None
stop_event = threading.Event()

# --- Web UI (for all users) ---
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Live Emotion Stream</title>
    <style>
        body { font-family: Arial; text-align: center; background: #0e0e0e; color: #eee; margin: 0; }
        h1 { margin-top: 20px; color: #0f0; }
        img { width: 640px; border: 3px solid #0f0; border-radius: 10px; margin-top: 20px; }
        .status { margin-top: 10px; font-size: 1.1em; }
    </style>
</head>
<body>
    <h1>🧠 Real-Time Emotion Detection (Multi-User)</h1>
    <div class="status" id="status">Connecting to WebSocket...</div>
    <img id="video" src="" alt="Waiting for stream..." />
    <script>
        const ws = new WebSocket("ws://" + window.location.host + "/ws");
        const video = document.getElementById("video");
        const status = document.getElementById("status");

        ws.onopen = () => {
            status.innerText = "✅ Connected to live stream!";
            console.log("WebSocket connected.");
        };
        ws.onmessage = (event) => {
            video.src = "data:image/jpeg;base64," + event.data;
        };
        ws.onclose = () => {
            status.innerText = "❌ Disconnected from stream.";
            console.warn("WebSocket disconnected.");
        };
        ws.onerror = (err) => {
            status.innerText = "⚠️ WebSocket error.";
            console.error("WebSocket error:", err);
        };
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return html

# --- Background camera thread ---
def camera_loop():
    global latest_frame
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(" Could not open webcam.")
        return

    print(" Webcam started (shared across users).")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            try:
                result = DeepFace.analyze(roi, actions=["emotion"], enforce_detection=False)
                if isinstance(result, list):
                    result = result[0]
                emotions = result.get("emotion", {})
                if emotions:
                    max_emotion = max(emotions, key=emotions.get)
                    message = f"{max_emotion}: {emotions[max_emotion]:.1f}"
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, message, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            except Exception:
                pass

        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        latest_frame = base64.b64encode(buffer).decode("utf-8")

        # Broadcast to all connected WebSocket clients
        asyncio.run(send_to_all_clients(latest_frame))

        cv2.waitKey(1)

    cap.release()
    print(" Webcam stopped.")

# --- Broadcast helper ---
async def send_to_all_clients(frame_data):
    remove_clients = []
    for ws in list(connected_clients):
        try:
            await ws.send_text(frame_data)
        except WebSocketDisconnect:
            remove_clients.append(ws)
        except Exception as e:
            print(f" Broadcast error: {e}")
            remove_clients.append(ws)

    # Clean disconnected clients
    for ws in remove_clients:
        connected_clients.discard(ws)
        print(f" Client removed. Active clients: {len(connected_clients)}")

# --- WebSocket endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    print(f" New WebSocket client connected! Active: {len(connected_clients)}")

    try:
        while not stop_event.is_set():
            await asyncio.sleep(1)
    except Exception as e:
        print(f" WebSocket error: {e}")
    finally:
        connected_clients.discard(websocket)
        print(f" Client disconnected. Active: {len(connected_clients)}")

# --- Startup & Shutdown events ---
@app.on_event("startup")
def startup_event():
    print(" Starting FastAPI and webcam thread...")
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()

@app.on_event("shutdown")
def shutdown_event():
    print(" Shutting down server and webcam.")
    stop_event.set()
