"""
app.py
-------
Flask server, copied/adapted directly from the Lab7 (OldManFalls) app.py
template. Receives camera frames + classification results posted by
cameraStartRPS.py and streams them to the browser via SSE
(/get_camera_stream), rendered in templates/camera_view.html.

Run on the Raspberry Pi:
    $ python app.py
Then browse to http://<pi-ip>:5000/visualize-camera-view
"""

import json
import time

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

app = Flask(__name__, template_folder="templates")


# ---------------------------- 前端頁面 ----------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.get("/visualize-camera-view")
def visualize_camera_view():
    return render_template("camera_view.html")


# ---------------------------- 後端資料 ----------------------------
class CameraView:
    def __init__(self):
        # image: base64-encoded JPEG string
        # result: classification result, e.g. "石頭" / "剪刀" / "布" / "Error"
        self.a_camera_frame = {"image": "", "result": "Error", "timestamp": ""}

    def update_camera_frame(self, frame):
        self.a_camera_frame = frame

    def get_camera_frame(self):
        return self.a_camera_frame


a_camera_view = CameraView()


@app.post("/post_camera_frame")
def receive_camera_frame():
    try:
        content = request.get_json()
        a_camera_view.update_camera_frame(content)
    except Exception as e:
        print(f"Error receiving data: {str(e)}")
        return jsonify({"success": False, "error": str(e)})
    return "camera frame updated successfully"


@app.route("/get_camera_stream")
def make_stream():
    # SSE (Server-Sent Events) endpoint, polled by camera_view.html
    @stream_with_context
    def generate():
        while True:
            try:
                yield "data:" + json.dumps(a_camera_view.get_camera_frame()) + "\n\n"
                time.sleep(0.1)
            except GeneratorExit:
                print("closed")
                break

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
