from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math
import os
import random


W, H = 1280, 720
SCALE = 2
OUT = os.path.join(os.getcwd(), "nohohon_comical_bgm_thumbnail.png")


def sc(v):
    return int(round(v * SCALE))


def pt(p):
    return tuple(sc(x) for x in p)


def pts(items):
    return [pt(p) for p in items]


def font(name, size):
    return ImageFont.truetype(os.path.join("C:\\Windows\\Fonts", name), sc(size))


random.seed(42)
img = Image.new("RGB", (W * SCALE, H * SCALE), "#f8efd3")
draw = ImageDraw.Draw(img)

# Soft sky gradient.
for y in range(H * SCALE):
    t = y / (H * SCALE)
    r = int(244 * (1 - t) + 255 * t)
    g = int(225 * (1 - t) + 241 * t)
    b = int(190 * (1 - t) + 214 * t)
    draw.line([(0, y), (W * SCALE, y)], fill=(r, g, b))

# Paper grain.
grain = Image.new("RGBA", img.size, (0, 0, 0, 0))
gd = ImageDraw.Draw(grain)
for _ in range(18000):
    x = random.randrange(W * SCALE)
    y = random.randrange(H * SCALE)
    a = random.randrange(8, 22)
    col = (122, 94, 62, a) if random.random() < 0.5 else (255, 255, 255, a)
    gd.point((x, y), fill=col)
img = Image.alpha_composite(img.convert("RGBA"), grain)
draw = ImageDraw.Draw(img)


def blob(points, fill, outline, width=4):
    draw.polygon(pts(points), fill=fill)
    for dx, dy in [(0, 0), (1.2, -0.8), (-1.0, 0.9)]:
        jittered = []
        for x, y in points:
            jittered.append((sc(x + dx + random.uniform(-1.2, 1.2)), sc(y + dy + random.uniform(-1.2, 1.2))))
        draw.line(jittered + [jittered[0]], fill=outline, width=sc(width), joint="curve")


def ellipse_box(box, fill, outline=None, width=3):
    b = tuple(sc(v) for v in box)
    draw.ellipse(b, fill=fill, outline=outline, width=sc(width) if outline else 1)


# Distant fantasy hills.
for base, color, outline, off in [
    (470, "#cfe6b3", "#87a86c", 0),
    (520, "#b9ddb0", "#6f9c65", 70),
    (585, "#9fcb88", "#5e8b57", 20),
]:
    points = [(-30, H), (-30, base)]
    for x in range(-30, W + 70, 90):
        y = base + math.sin((x + off) * 0.018) * 34 + random.uniform(-10, 10)
        points.append((x, y))
    points += [(W + 30, H)]
    blob(points, color, outline, 3)

# Big warm sun/moon.
ellipse_box((888, 72, 1118, 302), "#ffe78a", "#d7a85e", 5)
for r, a in [(145, 36), (185, 22), (225, 14)]:
    halo = Image.new("RGBA", img.size, (0, 0, 0, 0))
    hd = ImageDraw.Draw(halo)
    hd.ellipse((sc(1003 - r), sc(187 - r), sc(1003 + r), sc(187 + r)), fill=(255, 230, 125, a))
    img = Image.alpha_composite(img, halo)
draw = ImageDraw.Draw(img)

# Clouds.
def cloud(cx, cy, s):
    shadow = "#d6c8a4"
    fill = "#fff8dc"
    parts = [
        (-84, 12, 88, 58),
        (-58, -20, 20, 46),
        (0, -34, 92, 48),
        (52, -12, 138, 52),
    ]
    for ox1, oy1, ox2, oy2 in parts:
        ellipse_box((cx + ox1 * s, cy + oy1 * s + 5, cx + ox2 * s, cy + oy2 * s + 5), shadow)
    for ox1, oy1, ox2, oy2 in parts:
        ellipse_box((cx + ox1 * s, cy + oy1 * s, cx + ox2 * s, cy + oy2 * s), fill, "#b99f70", 2)


cloud(170, 92, 0.8)
cloud(650, 115, 0.58)
cloud(1115, 405, 0.56)

# Cozy cottage.
blob([(730, 448), (880, 438), (916, 573), (707, 586)], "#f7d59d", "#855f48", 4)
blob([(690, 453), (810, 350), (940, 459)], "#d96b5e", "#7b4e45", 5)
blob([(800, 501), (842, 493), (850, 579), (790, 582)], "#8d6949", "#5c3f2f", 3)
ellipse_box((815, 536, 826, 548), "#f7db80", "#5c3f2f", 2)
for x in [748, 872]:
    draw.rounded_rectangle((sc(x), sc(478), sc(x + 50), sc(520)), radius=sc(8), fill="#bfe8f2", outline="#6f5c4a", width=sc(3))
    draw.line((sc(x + 25), sc(480), sc(x + 25), sc(518)), fill="#6f5c4a", width=sc(2))
    draw.line((sc(x + 2), sc(499), sc(x + 48), sc(499)), fill="#6f5c4a", width=sc(2))

# Foreground path and flowers.
blob([(438, 718), (562, 605), (657, 584), (818, 720)], "#e8c886", "#aa8352", 3)
for _ in range(115):
    x = random.randint(40, 1220)
    y = random.randint(525, 685)
    if 420 < x < 835 and y > 590:
        continue
    stem = "#5d9b5d"
    draw.line((sc(x), sc(y + 8), sc(x + random.uniform(-2, 2)), sc(y - 5)), fill=stem, width=sc(2))
    petal = random.choice(["#ff8fa3", "#fff0a6", "#8fd3ff", "#f4a6ff"])
    ellipse_box((x - 4, y - 9, x + 4, y - 1), petal)

# Musical notes and sparkles.
note_font = font("NotoSansJP-VF.ttf", 50)
small_font = font("NotoSansJP-VF.ttf", 30)
for text, x, y, col, rot in [
    ("♪", 105, 310, "#7aa6d8", -10),
    ("♫", 214, 220, "#d47a91", 8),
    ("♪", 1032, 334, "#75b68a", 11),
    ("♬", 956, 502, "#c58edb", -12),
    ("♪", 540, 310, "#d7a54f", 7),
]:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    ld.text((sc(x), sc(y)), text, font=note_font, fill=col, stroke_width=sc(2), stroke_fill="#fff7d9")
    layer = layer.rotate(rot, resample=Image.Resampling.BICUBIC, center=(sc(x + 22), sc(y + 30)))
    img = Image.alpha_composite(img, layer)
draw = ImageDraw.Draw(img)
for _ in range(42):
    x = random.randint(45, 1200)
    y = random.randint(48, 410)
    r = random.choice([4, 5, 6])
    col = random.choice(["#fff7b0", "#ffffff", "#f4b7cf"])
    draw.line((sc(x - r), sc(y), sc(x + r), sc(y)), fill=col, width=sc(2))
    draw.line((sc(x), sc(y - r), sc(x), sc(y + r)), fill=col, width=sc(2))

# Friendly title typography.
title_font = font("UDDigiKyokashoN-B.ttc", 94)
sub_font = font("BIZ-UDGothicB.ttc", 64)
tag_font = font("BIZ-UDGothicB.ttc", 38)

def centered_text(text, y, fnt, fill, stroke, sw):
    box = draw.textbbox((0, 0), text, font=fnt, stroke_width=sc(sw))
    x = (W * SCALE - (box[2] - box[0])) // 2
    draw.text((x + sc(4), sc(y + 5)), text, font=fnt, fill=(123, 92, 58, 80), stroke_width=sc(sw), stroke_fill=(123, 92, 58, 35))
    draw.text((x, sc(y)), text, font=fnt, fill=fill, stroke_width=sc(sw), stroke_fill=stroke)


draw.rounded_rectangle((sc(64), sc(32), sc(426), sc(88)), radius=sc(20), fill="#fff8dc", outline="#b89464", width=sc(3))
draw.text((sc(89), sc(40)), "1時間耐久", font=tag_font, fill="#7b5a42")
centered_text("のほほん", 205, title_font, "#ff8f70", "#fff8dc", 8)
centered_text("COMICAL BGM", 320, sub_font, "#6aa7d8", "#fff8dc", 7)

# Gentle vignette and downsample for antialiasing.
vignette = Image.new("L", img.size, 0)
vd = ImageDraw.Draw(vignette)
vd.ellipse((sc(-160), sc(-120), sc(W + 160), sc(H + 180)), fill=255)
vignette = vignette.filter(ImageFilter.GaussianBlur(sc(90)))
shade = Image.new("RGBA", img.size, (105, 75, 45, 45))
img = Image.composite(img, Image.alpha_composite(img, shade), vignette)
img = img.convert("RGB").resize((W, H), Image.Resampling.LANCZOS)

img.save(OUT, quality=95)
print(OUT)
