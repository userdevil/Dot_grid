from flask import Flask, render_template, request, send_file, jsonify
from PIL import Image, ImageDraw
import math, io, cv2
import numpy as np

app = Flask(__name__)

GRID_SIZE = 25
SIZE = 800


# =========================
# BIT POSITION MAP (CRITICAL FIX)
# =========================
def get_bit_positions():
    cx = cy = GRID_SIZE // 2
    RADIUS = GRID_SIZE // 3

    positions = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            dx = col - cx
            dy = row - cy
            dist = math.sqrt(dx*dx + dy*dy)

            if dist < RADIUS:
                positions.append((row, col))

    return positions

def get_capacity():
    positions = get_bit_positions()
    return len(positions) // 2 - 16   # divide by 2 (ECC) + header


# =========================
# ENCODING
# =========================
def text_to_bits(text):
    binary = ''.join(format(ord(c), '08b') for c in text)
    length = format(len(binary), '016b')

    data = length + binary

    # simple error correction
    encoded = ''.join(bit * 3 for bit in data)

    return encoded


# =========================
# GENERATOR
# =========================
def generate_code(text, dot_color, bg_color):
    spacing = SIZE // GRID_SIZE

    img = Image.new("RGB", (SIZE, SIZE), bg_color)
    draw = ImageDraw.Draw(img)

    bits = text_to_bits(text)
    positions = get_bit_positions()

    cx = cy = GRID_SIZE // 2
    MAX_RADIUS = GRID_SIZE // 2

    bit_idx = 0

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):

            x = col * spacing + spacing // 2
            y = row * spacing + spacing // 2

            dx = col - cx
            dy = row - cy
            dist = math.sqrt(dx * dx + dy * dy)

            # smooth radial fade
            fade = 1 - (dist / MAX_RADIUS)
            fade = max(fade, 0)
            fade = fade ** 2.2

            base = 2 + fade * 10

            if fade <= 0:
                continue

            # encode using SAME position map
            if (row, col) in positions and bit_idx < len(bits):
                bit = bits[bit_idx]
                radius = base * (1.4 if bit == '1' else 0.6)
                bit_idx += 1
            else:
                radius = base

            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=dot_color
            )

    return img


# =========================
# DECODER
# =========================
def decode_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (SIZE, SIZE))

    spacing = SIZE // GRID_SIZE
    positions = get_bit_positions()

    binary = ""

    for (row, col) in positions:

        x = col * spacing + spacing // 2
        y = row * spacing + spacing // 2

        roi = img[y-8:y+8, x-8:x+8]

        if roi.size == 0:
            binary += '0'
            continue

        _, thresh = cv2.threshold(roi, 180, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            binary += '0'
            continue

        area = max(cv2.contourArea(c) for c in contours)

        bit = '1' if area > 50 else '0'
        binary += bit

    # =========================
    # ERROR CORRECTION
    # =========================
    corrected = ""
    
    for i in range(0, len(binary), 3):
        triplet = binary[i:i+3]
    
        if len(triplet) < 3:
            continue
    
        # majority voting
        corrected += '1' if triplet.count('1') >= 2 else '0'
    
    binary = corrected

    # =========================
    # HEADER
    # =========================
    try:
        length = int(binary[:16], 2)
    except:
        return "Decode error"

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

    max_bits = get_capacity()
    max_chars = max_bits // 8

    if len(text) > max_chars:
        text = text[:max_chars]  # truncate safely

    dot_color = request.form.get('dot_color', '#000000')
    bg_color = request.form.get('bg_color', '#f5f5f5')

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
