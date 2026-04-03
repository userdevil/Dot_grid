from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image, ImageDraw
import numpy as np
import cv2, math, io

app = Flask(__name__)

GRID_SIZE = 25
SIZE = 800


# =========================
# ENCODING
# =========================
def encode_text(text):
    binary = ''.join(format(ord(c), '08b') for c in text)
    length = format(len(binary), '016b')
    data = length + binary
    return data + data  # redundancy


# =========================
# RADIAL FUNCTION
# =========================
def radial_weight(x, y):
    cx = cy = GRID_SIZE // 2
    dist = math.sqrt((x-cx)**2 + (y-cy)**2)
    max_dist = math.sqrt(2*(cx**2))
    return 1 - (dist / max_dist)


# =========================
# GENERATOR
# =========================
def generate_code(text, dot_color, bg_color):
    img = Image.new("RGB", (SIZE, SIZE), bg_color)
    draw = ImageDraw.Draw(img)

    spacing = SIZE // GRID_SIZE
    bits = encode_text(text)
    bit_idx = 0

    # ===== MARKERS =====
    def marker(x, y):
        draw.ellipse((x-20, y-20, x+20, y+20), fill="black")

    corners = [(40,40),(SIZE-40,40),(40,SIZE-40),(SIZE-40,SIZE-40)]
    for c in corners:
        marker(*c)

    # ===== GRID + RADIAL =====
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):

            x = col * spacing + spacing//2
            y = row * spacing + spacing//2

            weight = radial_weight(col, row)

            if weight < 0.05:
                continue

            base = 2 + weight * 8

            if bit_idx < len(bits):
                bit = bits[bit_idx]
                radius = base * (1.4 if bit == '1' else 0.6)
                bit_idx += 1
            else:
                radius = base

            draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                         fill=dot_color)

    return img


# =========================
# DETECT MARKERS
# =========================
def find_markers(gray):
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    centers = []

    for c in contours:
        area = cv2.contourArea(c)
        if area > 500:
            (x, y), r = cv2.minEnclosingCircle(c)
            centers.append((int(x), int(y)))

    if len(centers) < 4:
        return None

    return np.array(centers[:4], dtype="float32")


# =========================
# PERSPECTIVE FIX
# =========================
def warp_image(img, pts):
    pts = sorted(pts, key=lambda x: x[1])

    top = sorted(pts[:2], key=lambda x: x[0])
    bottom = sorted(pts[2:], key=lambda x: x[0])

    ordered = np.array([top[0], top[1], bottom[0], bottom[1]], dtype="float32")

    dst = np.array([
        [0,0],
        [SIZE,0],
        [0,SIZE],
        [SIZE,SIZE]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(img, matrix, (SIZE, SIZE))


# =========================
# GRID SAMPLING
# =========================
def extract_bits(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    spacing = SIZE // GRID_SIZE

    bits = ""

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):

            x = col * spacing + spacing//2
            y = row * spacing + spacing//2

            roi = gray[y-3:y+3, x-3:x+3]
            mean = np.mean(roi)

            bits += '1' if mean < 128 else '0'

    return bits


# =========================
# ERROR CORRECTION
# =========================
def decode_bits(bits):
    half = len(bits)//2
    d1 = bits[:half]
    d2 = bits[half:]

    corrected = ''.join('1' if (a=='1' or b=='1') else '0'
                        for a,b in zip(d1,d2))

    length = int(corrected[:16], 2)
    payload = corrected[16:16+length]

    chars = []
    for i in range(0, len(payload), 8):
        byte = payload[i:i+8]
        if len(byte)==8:
            chars.append(chr(int(byte,2)))

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

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8),
                       cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    markers = find_markers(gray)
    if markers is None:
        return jsonify({"error": "Markers not found"})

    warped = warp_image(img, markers)
    bits = extract_bits(warped)
    decoded = decode_bits(bits)

    return jsonify({"decoded_text": decoded})


if __name__ == '__main__':
    app.run(debug=True)
