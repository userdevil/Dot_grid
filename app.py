from flask import Flask, render_template, request, send_file, jsonify
from PIL import Image, ImageDraw
import math, io, base64, cv2
import numpy as np

app = Flask(__name__)

# =========================
# DOT GENERATOR
# =========================
def generate_dot_grid(text, grid_size=35, sigma=6, max_r=10, min_r=1):
    size = 800
    spacing = size // grid_size

    img = Image.new("RGB", (size, size), "#f5f5f5")
    draw = ImageDraw.Draw(img)

    cx = cy = grid_size // 2

    def gaussian(x, y):
        return math.exp(-((x-cx)**2 + (y-cy)**2)/(2*sigma**2))

    bits = ''.join(format(ord(c), '08b') for c in text)
    bit_idx = 0

    for row in range(grid_size):
        for col in range(grid_size):

            weight = gaussian(col, row)
            if weight < 0.05:
                continue

            base = min_r + weight * (max_r - min_r)

            # encode data subtly
            if bit_idx < len(bits):
                if bits[bit_idx] == '1':
                    radius = base * 1.2
                else:
                    radius = base * 0.7
                bit_idx += 1
            else:
                radius = base

            x = col * spacing + spacing//2
            y = row * spacing + spacing//2

            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill="black")

    return img


# =========================
# ROUTES
# =========================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    text = request.form['text']
    grid = int(request.form.get('grid', 35))
    sigma = float(request.form.get('sigma', 6))

    img = generate_dot_grid(text, grid_size=grid, sigma=sigma)

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    return send_file(buf, mimetype='image/png')


# =========================
# SCANNER (UPLOAD)
# =========================
@app.route('/scan', methods=['POST'])
def scan():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

    # basic threshold
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # detect blobs (dots)
    detector = cv2.SimpleBlobDetector_create()
    keypoints = detector.detect(thresh)

    return jsonify({
        "dots_detected": len(keypoints),
        "message": "Decoding logic can be added here"
    })


if __name__ == '__main__':
    app.run(debug=True)
