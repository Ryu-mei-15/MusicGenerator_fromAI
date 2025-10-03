from typing import List, Dict, Tuple, Optional
from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import os, re, random, sqlite3, csv
from io import StringIO
from datetime import datetime

# ===== AI 推論モジュール =====
from model_infer import ABCGenerator

# FastAPI 起動時にジェネレータ用意（models/ があれば AI 生成 ON）
GEN = ABCGenerator()

app = FastAPI(title="Text2Melody ABC")

# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ===== 保存ディレクトリ（カレント/ABC）＆静的配信 =====
SAVE_DIR = os.path.join(os.getcwd(), "ABC")
os.makedirs(SAVE_DIR, exist_ok=True)
app.mount("/abc", StaticFiles(directory=SAVE_DIR), name="abc")

# ===== SQLite 準備 =====
DB_PATH = os.path.join(os.getcwd(), "abc_logs.sqlite3")

def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _init_db():
    conn = _db()
    conn.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        prompt TEXT NOT NULL,
        key TEXT NOT NULL,
        mode TEXT NOT NULL,
        tempo INTEGER NOT NULL,
        bars INTEGER NOT NULL,
        meter TEXT NOT NULL,
        mood TEXT NOT NULL,
        low INTEGER NOT NULL,
        high INTEGER NOT NULL,
        saved_name TEXT NOT NULL,
        saved_url TEXT NOT NULL,
        saved_file TEXT NOT NULL,
        note_count INTEGER NOT NULL
    )
    """)
    conn.commit()
    conn.close()

_init_db()

# ===== 音階定義・従来ロジック（フォールバック用） =====
SEMITONES = {
    "C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,
    "F":5,"F#":6,"Gb":6,"G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11
}
MAJOR_SCALE = [0,2,4,5,7,9,11]
NAT_MINOR_SCALE = [0,2,3,5,7,8,10]

def midi_to_abc_pitch(midi_num:int) -> str:
    octave = (midi_num // 12) - 1
    pc = midi_num % 12
    name = None
    for k,v in SEMITONES.items():
        if v==pc and len(k)==1:
            name = k
            break
    if name is None: name = "C"
    delta = octave - 4  # C4 を基準
    if delta == 0:
        base = name.upper()
    elif delta > 0:
        base = name.lower() + ("'" * delta)
    else:
        base = name.upper() + ("," * abs(delta))
    return base

def build_scale(key:str, mode:str) -> List[int]:
    root = SEMITONES.get(key, 0)
    template = MAJOR_SCALE if mode=="major" else NAT_MINOR_SCALE
    return [(root + deg) % 12 for deg in template]

def parse_prompt(prompt:str) -> Dict:
    txt = prompt.lower()
    m_key = re.search(r"(c|c#|db|d|d#|eb|e|f|f#|gb|g|g#|ab|a|a#|bb|b)\s*(major|minor|maj|min|メジャー|マイナー)?", txt)
    key = (m_key.group(1).upper() if m_key else "C")
    mode = "major"
    if m_key and m_key.group(2):
        if "min" in m_key.group(2) or "マイナー" in m_key.group(2):
            mode = "minor"
    mood = "bright" if ("bright" in txt or "明る" in txt or "元気" in txt) else ("sad" if ("sad" in txt or "哀" in txt or "切な" in txt) else "neutral")
    m_tempo = re.search(r"(bpm|tempo)\s*[:=]?\s*(\d+)", txt) or re.search(r"(\d+)\s*bpm", txt)
    tempo = int(m_tempo.group(2) if m_tempo and len(m_tempo.groups())>=2 else (m_tempo.group(1) if m_tempo else 100))
    m_bars = re.search(r"(bars?|小節)\s*[:=]?\s*(\d+)", txt)
    bars = int(m_bars.group(2) if m_bars else 8)
    meter = "4/4"
    if "3/4" in txt or "waltz" in txt or "ワルツ" in txt:
        meter = "3/4"
    low = 55; high = 77
    if "高め" in txt or "high" in txt: low, high = 64, 79
    if "低め" in txt or "low" in txt:  low, high = 52, 67
    return {"key":key, "mode":mode, "tempo":tempo, "bars":bars, "meter":meter, "mood":mood, "low":low, "high":high}

def generate_melody(params:Dict) -> List[Tuple[int, float]]:
    scale = build_scale(params["key"], params["mode"])
    def snap_to_scale(m:int)->int:
        pc = m % 12
        best = min(scale, key=lambda s: min((pc - s) % 12, (s - pc) % 12))
        delta = (best - pc)
        if delta > 6: delta -= 12
        if delta < -6: delta += 12
        return m + delta
    cur = snap_to_scale(random.randint(params["low"], params["high"]))
    q_per_bar = 3 if params["meter"]=="3/4" else 4
    total_q = params["bars"] * q_per_bar
    if params["mood"]=="bright":
        step_probs = [ -2,-1,0,1,2,3,5 ]; weights = [1,3,2,4,3,2,1]
    elif params["mood"]=="sad":
        step_probs = [ -5,-3,-2,-1,0,1,2 ]; weights = [1,2,3,4,2,2,1]
    else:
        step_probs = [ -3,-2,-1,0,1,2,3 ]; weights = [2,3,4,2,4,3,2]
    dur_candidates = [0.5,0.5,0.5,1.0,1.0,0.25,0.75]
    melody : List[Tuple[int,float]] = []
    q_sum = 0.0
    while q_sum < total_q - 1e-6:
        step = random.choices(step_probs, weights=weights, k=1)[0]
        nxt = min(max(cur + step, params["low"]), params["high"])
        nxt = snap_to_scale(nxt)
        dur = random.choice(dur_candidates)
        if q_sum + dur > total_q: dur = total_q - q_sum
        melody.append((nxt, dur))
        cur = nxt; q_sum += dur
    return melody

def melody_to_abc(melody:List[Tuple[int,float]], params:Dict) -> str:
    header = [
        "X:1", "T:Generated Melody",
        f"M:{params['meter']}", "L:1/8",
        f"Q:1/4={params['tempo']}",
        f"K:{params['key']}{'' if params['mode']=='major' else 'min'}"
    ]
    body = []
    unit = 0.5
    count_in_bar = 6 if params["meter"]=="3/4" else 8
    cell = []; cells_in_bar = 0
    for midi, qlen in melody:
        abc_note = midi_to_abc_pitch(midi)
        units = round(qlen / unit)  # 0.5→1，1.0→2，0.25→/2≈0
        if units == 1:   token = abc_note
        elif units == 2: token = f"{abc_note}2"
        elif units == 3: token = f"{abc_note}3"
        elif units == 4: token = f"{abc_note}4"
        elif units == 0: token = f"{abc_note}/2"
        else:            token = f"{abc_note}{units}"
        cell.append(token)
        cells_in_bar += units if units>0 else 1
        if cells_in_bar >= count_in_bar:
            body.append(" ".join(cell) + " |")
            cell = []; cells_in_bar = 0
    if cell: body.append(" ".join(cell) + " |]")
    return "\n".join(header + [" ".join(body)])

# ===== モデル I/O =====
class ComposeRequest(BaseModel):
    prompt: str

def _safe_name(s:str) -> str:
    s = re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE)
    return re.sub(r"_+", "_", s).strip("_")

def _insert_log(
    prompt:str, params:Dict, saved_name:str, saved_url:str, saved_file:str, note_count:int
) -> int:
    conn = _db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO logs (ts, prompt, key, mode, tempo, bars, meter, mood, low, high,
                          saved_name, saved_url, saved_file, note_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        prompt, params["key"], params["mode"], params["tempo"], params["bars"],
        params["meter"], params["mood"], params["low"], params["high"],
        saved_name, saved_url, saved_file, note_count
    ))
    conn.commit()
    log_id = cur.lastrowid
    conn.close()
    return log_id

def build_control_header(params:Dict, prompt:str)->str:
    """
    学習時にも自然に見える条件付き ABC ヘッダを作る．
    """
    title = prompt.strip().replace("\n"," ")[:80]
    header = [
        "X:1",
        f"T:Prompt: {title}",
        f"M:{params['meter']}",
        "L:1/8",
        f"Q:1/4={params['tempo']}",
        f"K:{params['key']}{'' if params['mode']=='major' else 'min'}",
        "\n"
    ]
    return "\n".join(header)

# ===== 作曲エンドポイント（AI→フォールバック） =====
@app.post("/compose")
def compose(req: ComposeRequest):
    params = parse_prompt(req.prompt)

    # AI 生成（学習済みがあれば）
    abc_text: Optional[str] = None
    if GEN.ok:
        control = build_control_header(params, req.prompt)
        sampled = GEN.sample(control_header=control, max_len=1200, temperature=0.95, top_p=0.9)
        if sampled:
            abc_text = control + sampled

    # フォールバック（従来ロジック）
    if abc_text is None:
        melody = generate_melody(params)
        abc_text = melody_to_abc(melody, params)

    # 保存
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{params['key']}_{params['mode']}_{params['tempo']}bpm_{params['bars']}bars"
    fname = f"{ts}_{_safe_name(base)}.abc"
    fpath = os.path.join(SAVE_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(abc_text)
    saved_url = f"/abc/{fname}"

    # 簡易ノート数推定（ABC文字からの粗カウント）
    note_count = sum(abc_text.count(ch) for ch in list("ABCDEFG"))

    log_id = _insert_log(
        prompt=req.prompt, params=params,
        saved_name=fname, saved_url=saved_url,
        saved_file=fpath, note_count=note_count
    )

    return JSONResponse({
        "abc": abc_text, "params": params,
        "saved_file": fpath, "saved_name": fname, "saved_url": saved_url,
        "note_count": note_count, "log_id": log_id,
        "ai_used": GEN.ok
    })

# ===== ログ取得：JSON（ページング対応） =====
@app.get("/logs")
def get_logs(page:int = Query(1, ge=1), per_page:int = Query(20, ge=1, le=200)):
    offset = (page - 1) * per_page
    conn = _db()
    rows = conn.execute(
        "SELECT * FROM logs ORDER BY id DESC LIMIT ? OFFSET ?", (per_page, offset)
    ).fetchall()
    total = conn.execute("SELECT COUNT(*) AS c FROM logs").fetchone()["c"]
    conn.close()
    items = [dict(r) for r in rows]
    return {"page": page, "per_page": per_page, "total": total, "items": items}

# ===== ログ全件：CSV ダウンロード =====
@app.get("/logs.csv")
def download_logs_csv():
    conn = _db()
    rows = conn.execute("SELECT * FROM logs ORDER BY id DESC").fetchall()
    conn.close()

    buff = StringIO()
    writer = csv.writer(buff)
    headers = rows[0].keys() if rows else [
        "id","ts","prompt","key","mode","tempo","bars","meter","mood","low","high",
        "saved_name","saved_url","saved_file","note_count"
    ]
    writer.writerow(headers)
    for r in rows:
        writer.writerow([r[h] for h in headers])
    buff.seek(0)

    return StreamingResponse(
        buff, media_type="text/csv",
        headers={"Content-Disposition":"attachment; filename=abc_logs.csv"}
    )
