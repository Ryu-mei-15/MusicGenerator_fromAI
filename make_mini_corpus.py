# make_mini_corpus.py
import os, random, textwrap

OUT = "ABC_CORPUS"
os.makedirs(OUT, exist_ok=True)

TEMPLATES = [
    ("C",  "4/4", 120, "CDEF GABc | cBAG FEDC |]"),
    ("G",  "4/4", 110, "GABc d2 e2 | dBAG E2 D2 |]"),
    ("F",  "3/4",  90, "c2 d e | f2 g a | g f e |]"),
    ("Am", "4/4", 100, "A B c d | e f g a | a g f e |]"),
    ("D",  "2/4", 130, "d e f# g | a b a g |]"),
]

def abc_one(idx, key, meter, bpm, body):
    mode = "min" if key.endswith("m") else ""
    kline = key[:-1] if key.endswith("m") else key
    return textwrap.dedent(f"""\
    X:{idx}
    T:Seed {idx}
    M:{meter}
    L:1/8
    Q:1/4={bpm}
    K:{kline}{mode}
    {body}
    """)

# だいたい40曲くらい作る
N = 40
for i in range(1, N+1):
    key, meter, bpm, body = random.choice(TEMPLATES)
    # ちょっとだけバリエーション
    body2 = body.replace(" |", " | ").replace("  ", " ")
    if random.random() < 0.5:
        body2 = body2.replace("c", "c'").replace("d", "d'")
    abc = abc_one(i, key, meter, bpm, body2)
    with open(os.path.join(OUT, f"seed_{i:03d}.abc"), "w", encoding="utf-8") as f:
        f.write(abc)

print(f"Done. Wrote {N} files to {OUT}/")
