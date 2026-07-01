from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math
import os
import random


W, H = 1280, 720
SCALE = 2
OUT = os.path.join(os.getcwd(), "nohohon_round_platformer_bgm_thumbnail.png")


def sc(v):
    return int(round(v * SCALE))


def font(name, size):
    return ImageFont.truetype(os.path.join("C:\\Windows\\Fonts", name), sc(size))


def pbox(box):
    return tuple(sc(v) for v in box)


random.seed(86)
img = Image.new("RGBA", (W * SCALE, H * SCALE), "#fff2bd")
draw = ImageDraw.Draw(img)

# Bright handmade sky.
for y in range(H * SCALE):
    t = y / (H * SCALE)
    r = int(255 * (1 - t) + 247 * t)
    g = int(234 * (1 - t) + 255 * t)
    b = int(174 * (1 - t) + 218 * t)
    draw.line([(0, y), (W * SCALE, y)], fill=(r, g, b, 255))

texture = Image.new("RGBA", img.size, (0, 0, 0, 0))
td = ImageDraw.Draw(texture)
for _ in range(14000):
    x = random.randrange(W * SCALE)
    y = random.randrange(H * SCALE)
    a = random.randrange(5, 16)
    td.point((x, y), fill=random.choice([(130, 95, 54, a), (255, 255, 255, a)]))
img = Image.alpha_composite(img, texture)
draw = ImageDraw.Draw(img)


def rounded_poly(points, fill, outline, width=5):
    pts = [(sc(x), sc(y)) for x, y in points]
    draw.polygon(pts, fill=fill)
    for _ in range(2):
        jitter = [(sc(x + random.uniform(-1.4, 1.4)), sc(y + random.uniform(-1.4, 1.4))) for x, y in points]
        draw.line(jitter + [jitter[0]], fill=outline, width=sc(width), joint="curve")


def ellipse(box, fill, outline=None, width=4):
    draw.ellipse(pbox(box), fill=fill, outline=outline, width=sc(width) if outline else 1)


def round_rect(box, radius, fill, outline=None, width=4):
    draw.rounded_rectangle(pbox(box), radius=sc(radius), fill=fill, outline=outline, width=sc(width) if outline else 1)


# Large playful sun.
ellipse((928, 54, 1158, 284), "#ffe27a", "#d9a14a", 6)
for i in range(18):
    ang = i * math.tau / 18
    cx, cy = 1043 + math.cos(ang) * 150, 169 + math.sin(ang) * 150
    ellipse((cx - 9, cy - 9, cx + 9, cy + 9), "#ffe99c")

# Puffed clouds.
def cloud(cx, cy, s):
    for dx, dy, w, h in [(-80, 20, 108, 46), (-48, -5, 80, 64), (14, -28, 102, 76), (76, 6, 94, 50)]:
        ellipse((cx + dx * s, cy + dy * s, cx + (dx + w) * s, cy + (dy + h) * s), "#fffbe7", "#b89561", 3)


cloud(172, 95, 0.9)
cloud(662, 104, 0.62)
cloud(1110, 382, 0.55)

# Rounded platformer hills with spot patterns.
hill_sets = [
    (505, "#b6e887", "#548b54", 0, 40),
    (555, "#8fd16e", "#417942", 75, 54),
    (625, "#63b95d", "#386d38", 12, 72),
]
for base, fill, outline, off, amp in hill_sets:
    points = [(-40, H), (-40, base)]
    for x in range(-40, W + 100, 80):
        y = base + math.sin((x + off) * 0.015) * amp + math.sin((x + off) * 0.038) * 12
        points.append((x, y))
    points += [(W + 40, H)]
    rounded_poly(points, fill, outline, 5)

for _ in range(60):
    x = random.randint(30, 1240)
    y = random.randint(470, 660)
    if random.random() < 0.55:
        ellipse((x - 18, y - 8, x + 18, y + 8), "#d9f6a2", None)

# Chunky toy-like platforms.
for x, y, w, c in [(70, 440, 190, "#ffcd68"), (930, 496, 188, "#ffb77c"), (410, 510, 180, "#a9df70")]:
    round_rect((x, y, x + w, y + 42), 18, c, "#8b6943", 5)
    for k in range(0, w, 38):
        ellipse((x + k + 10, y + 13, x + k + 22, y + 25), "#fff0a8")

# Original round creature, deliberately not a known character.
shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
sd = ImageDraw.Draw(shadow)
sd.ellipse(pbox((612, 602, 822, 646)), fill=(74, 96, 44, 70))
img = Image.alpha_composite(img, shadow)
draw = ImageDraw.Draw(img)
ellipse((640, 428, 804, 610), "#74d46f", "#3f7c45", 6)
ellipse((575, 468, 695, 584), "#78d876", "#3f7c45", 6)
ellipse((602, 512, 762, 632), "#fff6cf", "#3f7c45", 5)
ellipse((555, 545, 612, 610), "#f6c56e", "#8b6943", 5)
ellipse((755, 548, 822, 610), "#f6c56e", "#8b6943", 5)
ellipse((585, 438, 638, 492), "#fff8e8", "#3f7c45", 5)
ellipse((633, 430, 690, 489), "#fff8e8", "#3f7c45", 5)
ellipse((607, 459, 623, 476), "#303030")
ellipse((654, 453, 670, 471), "#303030")
draw.arc(pbox((612, 486, 672, 530)), start=12, end=166, fill="#5a3d32", width=sc(4))
for x, y in [(716, 410), (756, 396), (790, 421)]:
    ellipse((x - 18, y - 18, x + 18, y + 18), "#ffef82", "#8b6943", 4)

# Fruit, flowers, and notes for the BGM mood.
for x, y, col in [(172, 390, "#ff6868"), (984, 458, "#ff6868"), (1120, 540, "#ff83b5"), (368, 486, "#ffdd5f")]:
    ellipse((x - 17, y - 17, x + 17, y + 17), col, "#8b6943", 3)
    draw.arc(pbox((x - 3, y - 35, x + 30, y - 8)), 195, 290, fill="#4c8c43", width=sc(4))

note_font = font("NotoSansJP-VF.ttf", 55)
for text, x, y, col in [
    ("♪", 245, 262, "#5ba8e6"),
    ("♫", 384, 345, "#e577a2"),
    ("♪", 1000, 318, "#59b881"),
    ("♬", 1110, 464, "#bd7ce0"),
    ("♪", 520, 246, "#e3a94c"),
]:
    draw.text((sc(x), sc(y)), text, font=note_font, fill=col, stroke_width=sc(3), stroke_fill="#fff9df")

for _ in range(90):
    x = random.randint(36, 1210)
    y = random.randint(56, 420)
    r = random.choice([4, 5, 7])
    col = random.choice(["#fff9a7", "#ffffff", "#ff9ed0", "#8ed9ff"])
    draw.line((sc(x - r), sc(y), sc(x + r), sc(y)), fill=col, width=sc(2))
    draw.line((sc(x), sc(y - r), sc(x), sc(y + r)), fill=col, width=sc(2))

# Readable thumbnail text.
tag_font = font("BIZ-UDGothicB.ttc", 36)
title_font = font("UDDigiKyokashoN-B.ttc", 92)
sub_font = font("BIZ-UDGothicB.ttc", 60)

round_rect((64, 34, 404, 88), 22, "#fff9df", "#9b7447", 4)
draw.text((sc(90), sc(42)), "1時間耐久", font=tag_font, fill="#73513d")


def centered(text, y, fnt, fill, stroke, sw):
    box = draw.textbbox((0, 0), text, font=fnt, stroke_width=sc(sw))
    x = (W * SCALE - (box[2] - box[0])) // 2
    draw.text((x + sc(5), sc(y + 6)), text, font=fnt, fill=(91, 65, 41, 92), stroke_width=sc(sw), stroke_fill=(91, 65, 41, 50))
    draw.text((x, sc(y)), text, font=fnt, fill=fill, stroke_width=sc(sw), stroke_fill=stroke)


centered("のほほん", 186, title_font, "#ff825f", "#fff9df", 9)
centered("COMICAL BGM", 304, sub_font, "#4e9dde", "#fff9df", 8)

vignette = Image.new("L", img.size, 0)
vd = ImageDraw.Draw(vignette)
vd.ellipse(pbox((-120, -120, W + 120, H + 120)), fill=255)
vignette = vignette.filter(ImageFilter.GaussianBlur(sc(90)))
shade = Image.new("RGBA", img.size, (115, 82, 38, 42))
img = Image.composite(img, Image.alpha_composite(img, shade), vignette)

img = img.convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
img.save(OUT, quality=95)
print(OUT)
