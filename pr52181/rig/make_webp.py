import sys
from PIL import Image, ImageDraw, ImageFont
ui, out = sys.argv[1], sys.argv[2]
font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
frames_spec = [
    ("base", "49b2058c1a7", "01_authorize_page", "1. MCP client opens the proxy's authorize page"),
    ("base", "49b2058c1a7", "02_api_key_entered", "2. User types their API key and submits"),
    ("base", "49b2058c1a7", "03_callback_failed", "3. Token request lands on another worker: 400 invalid_grant"),
    ("base", "49b2058c1a7", "04_callback_tally", "4. Six connects, 5 failed with invalid_grant"),
    ("head", "848a89e51c8", "01_authorize_page", "1. Same authorize page"),
    ("head", "848a89e51c8", "02_api_key_entered", "2. Same key form"),
    ("head", "848a89e51c8", "04_callback_tally", "3. Six connects, every token request 200"),
]
frames = []
for side, sha, step, caption in frames_spec:
    im = Image.open(f"{ui}/{step}_{side}_{sha}.png").convert("RGB").crop((0, 0, 1280, 900))
    canvas = Image.new("RGB", (1280, 960), "#ffffff")
    canvas.paste(im, (0, 60))
    d = ImageDraw.Draw(canvas)
    color = "#b3261e" if side == "base" else "#0a7d32"
    d.rectangle((0, 0, 1280, 60), fill=color)
    d.text((20, 15), f"{'Before' if side == 'base' else 'After'} ({side} {sha})   {caption}", font=font, fill="#ffffff")
    frames.append(canvas)
frames[0].save(out, save_all=True, append_images=frames[1:], duration=2800, loop=0, lossless=True)
print("frames", len(frames))
