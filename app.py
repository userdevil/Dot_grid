from flask import Flask, render_template, request, send_file, jsonify
from PIL import Image, ImageDraw
import math, io, base64, cv2
import numpy as np

app = Flask(__name__)

GRID_SIZE = 25  # smaller = more reliable


# =========================
# ENCODING
# =========================
def text_to_bits(text):
    binary = ''.join(format(ord(c), '08b') for c in text)
    length = format(len(binary), '016b')  # header
    return length + binary


# =========================
# GENERATOR
# =========================
def generate_code(text, dot_color, bg_color):
    size = 800
    spacing = size // GRID_SIZE

    img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)

    bits = text_to_bits(text)
    bit_idx = 0

    cx = cy = GRID_SIZE // 2

    def gaussian(x, y):
        return math.exp(-((x-cx)**2 + (y-cy)**2)/(2*(GRID_SIZE/4)**2))

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):

            x = col * spacing + spacing // 2
            y = row * spacing + spacing // 2

            weight = gaussian(col, row)

            if weight < 0.08:
                continue

            base = 2 + weight * 10

            # encode bit
            if bit_idx < len(bits):
                bit = bits[bit_idx]
                radius = base * (1.4 if bit == '1' else 0.6)
                bit_idx += 1
            else:
                radius = base

            draw.ellipse(
                (x-radius, y-radius, x+radius, y+radius),
                fill=dot_color
            )

    return img


# =========================
# DECODING
# =========================
def decode_image(img):
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

    if len(keypoints) < 20:
        return "Detection failed"

    pts = np.array([kp.pt for kp in keypoints])

    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)

    grid = [[0]*GRID_SIZE for _ in range(GRID_SIZE)]

    for kp in keypoints:
        x, y = kp.pt

        gx = int((x - min_x)/(max_x - min_x)*(GRID_SIZE-1))
        gy = int((y - min_y)/(max_y - min_y)*(GRID_SIZE-1))

        bit = 1 if kp.size > 8 else 0
        grid[gy][gx] = bit

    binary = ''.join(str(bit) for row in grid for bit in row)

    # read header
    length = int(binary[:16], 2)
    data_bits = binary[16:16+length]

    chars = []
    for i in range(0, len(data_bits), 8):
        byte = data_bits[i:i+8]
        if len(byte) == 8:
            chars.append(chr(int(byte, 2)))

    return ''.join(chars)


# =========================
# ROUTES
# =========================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    text = request.form['text']
    dot_color = request.form.get('dot_color', '#000000')
    bg_color = request.form.get('bg_color', '#ffffff')

    img = generate_code(text, dot_color, bg_color)

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    return send_file(buf, mimetype='image/png')


@app.route('/scan', methods=['POST'])
def scan():
    file = request.files['image']

    img = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    decoded = decode_image(img)

    return jsonify({
        "decoded_text": decoded
    })


if __name__ == '__main__':
    app.run(debug=True)
