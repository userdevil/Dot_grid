from flask import Flask, render_template, request, send_file, jsonify
from PIL import Image, ImageDraw
import math, io, cv2
import numpy as np

app = Flask(__name__)

GRID_SIZE = 25
SIZE = 800

# center region for encoding (IMPORTANT)
CENTER_START = 6
CENTER_END = GRID_SIZE - 6


# =========================
# ENCODING
# =========================
def text_to_bits(text):
    binary = ''.join(format(ord(c), '08b') for c in text)
    length = format(len(binary), '016b')

    data = length + binary

    # 🔥 repeat each bit twice (error correction)
    encoded = ''.join(bit * 2 for bit in data)

    return encoded

# =========================
# RADIAL STYLE
# =========================
def radial_weight(x, y):
    cx = cy = GRID_SIZE // 2
    RADIUS = GRID_SIZE // 3  # encoding circle size
    dist = math.sqrt((x - cx)**2 + (y - cy)**2)
    max_dist = math.sqrt(2 * (cx**2))
    return 1 - (dist / max_dist)


# =========================
# GENERATOR (STYLE + DATA)
# =========================
def generate_code(text, dot_color, bg_color):
    spacing = SIZE // GRID_SIZE

    img = Image.new("RGB", (SIZE, SIZE), bg_color)
    draw = ImageDraw.Draw(img)

    bits = text_to_bits(text)
    bit_idx = 0

    cx = cy = GRID_SIZE // 2
    MAX_RADIUS = GRID_SIZE // 2   # 🔥 FIXED
    RADIUS = GRID_SIZE // 3

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):

            x = col * spacing + spacing // 2
            y = row * spacing + spacing // 2

            dx = col - cx
            dy = row - cy
            dist = math.sqrt(dx * dx + dy * dy)

            # 🔥 TRUE CIRCULAR FADE (NO SQUARE EFFECT)
            fade = 1 - (dist / MAX_RADIUS)
            fade = max(fade, 0)

            # smooth curve
            fade = fade ** 2.2

            base = 2 + fade * 10

            # remove very outer dots
            if fade <= 0:
                continue

            # encoding center
            if dist < RADIUS:
                if bit_idx < len(bits):
                    bit = bits[bit_idx]
                    radius = base * (1.4 if bit == '1' else 0.6)
                    bit_idx += 1
                else:
                    radius = base
            else:
                radius = base

            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=dot_color
            )

    return img

# =========================
# DECODER (STABLE)
# =========================
def decode_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # normalize image size
    img = cv2.resize(gray, (SIZE, SIZE))
    spacing = SIZE // GRID_SIZE

    binary = ""

    cx = cy = GRID_SIZE // 2
    RADIUS = GRID_SIZE // 3
    
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
    
            dx = col - cx
            dy = row - cy
            dist = math.sqrt(dx * dx + dy * dy)
    
            # only read inside circular region
            if dist >= RADIUS:
                continue
    
            x = col * spacing + spacing // 2
            y = row * spacing + spacing // 2
    
            roi = img[y-6:y+6, x-6:x+6]
    
            if roi.size == 0:
                binary += '0'
                continue
    
            # global adaptive threshold
            mean = np.mean(roi)
            std = np.std(roi)
            
            bit = '1' if (mean < 180 and std > 8) else '0'
    
            # ✅ IMPORTANT FIX
            binary += bit
    
    
    # =========================
    # SIMPLE ERROR CORRECTION
    # =========================
    corrected = ""
    
    for i in range(0, len(binary), 2):
        pair = binary[i:i+2]
    
        if len(pair) < 2:
            continue
    
        corrected += '1' if pair.count('1') >= 1 else '0'
    
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
