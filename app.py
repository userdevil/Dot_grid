from flask import Flask, render_template, request, send_file, jsonify
from PIL import Image, ImageDraw
import math, io, base64, cv2
import numpy as np

app = Flask(__name__)

# =========================
# GENERATOR
# =========================
def generate_dot_grid(text, grid_size, sigma, max_r, min_r, dot_color, bg_color):
    size = 800
    spacing = size // grid_size

    img = Image.new("RGB", (size, size), bg_color)
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

            # subtle encoding
            if bit_idx < len(bits):
                radius = base * (1.2 if bits[bit_idx] == '1' else 0.7)
                bit_idx += 1
            else:
                radius = base

            x = col * spacing + spacing // 2
            y = row * spacing + spacing // 2

            draw.ellipse(
                (x-radius, y-radius, x+radius, y+radius),
                fill=dot_color
            )

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
    max_r = float(request.form.get('max_r', 10))
    min_r = float(request.form.get('min_r', 1))

    dot_color = request.form.get('dot_color', '#000000')
    bg_color = request.form.get('bg_color', '#ffffff')

    img = generate_dot_grid(text, grid, sigma, max_r, min_r, dot_color, bg_color)

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    return send_file(buf, mimetype='image/png')


# =========================
# SCANNER
# =========================
@app.route('/scan', methods=['POST'])
def scan():
    file = request.files['image']

    img = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 3
    )

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 5000

    params.filterByCircularity = True
    params.minCircularity = 0.6

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(thresh)

    debug_img = cv2.drawKeypoints(
        img, keypoints, None, (0, 0, 255),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    _, buffer = cv2.imencode('.png', debug_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "dots_detected": len(keypoints),
        "debug_image": img_base64
    })


if __name__ == '__main__':
    app.run(debug=True)
