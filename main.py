# main.py
from typing import List, Dict, Tuple, Optional
from fastapi import FastAPI, Query, Request, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import os, re, random, sqlite3, csv, logging, shutil, uuid, json
from io import StringIO
from datetime import datetime
from urllib.parse import urljoin

# optional model_infer
try:
    from model_infer import ABCGenerator
except Exception:
    ABCGenerator = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("text2melody")

app = FastAPI(title="Text2Melody ABC")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# static dirs
SAVE_DIR = os.path.join(os.getcwd(), "ABC")
MM_DIR = os.path.join(os.getcwd(), "MM")  # multimodal saved files
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(MM_DIR, exist_ok=True)
app.mount("/abc", StaticFiles(directory=SAVE_DIR), name="abc")
app.mount("/mm", StaticFiles(directory=MM_DIR), name="mm")

# DB
DB_PATH = os.path.join(os.getcwd(), "abc_logs.sqlite3")
def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _init_db():
    """
    DB 初期化＋既存 DB への安全なマイグレーションを行う．
    - テーブルがなければ作成
    - 既存テーブルにカラムが無ければ ALTER TABLE で追加（冪等）
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 1) 基本テーブルがなければ作成（旧スキーマとの互換を保つ）
    cur.execute("""
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

    # 2) 欠損カラムを追加するユーティリティ
    def add_column_if_not_exists(table: str, column_def: str):
        """
        column_def 例: "valence REAL"
        """
        col_name = column_def.split()[0]
        # SQLite の PRAGMA table_info で列一覧を取得
        cur.execute(f"PRAGMA table_info({table})")
        cols = [row[1] for row in cur.fetchall()]  # row[1] はカラム名
        if col_name not in cols:
            logger.info(f"Adding column {col_name} to {table}")
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")

    # 3) 追加したいカラム群（ここに必要カラムを列挙）
    add_column_if_not_exists("logs", "valence REAL")
    add_column_if_not_exists("logs", "arousal REAL")
    add_column_if_not_exists("logs", "multimodal_audio TEXT")
    add_column_if_not_exists("logs", "multimodal_image TEXT")

    # 4) multimodal_logs テーブル（なければ作る）
    cur.execute("""
    CREATE TABLE IF NOT EXISTS multimodal_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        type TEXT NOT NULL,
        file_path TEXT,
        meta JSON
    )
    """)

    conn.commit()
    conn.close()


_init_db()

# init model if available
GEN = None
if ABCGenerator:
    try:
        GEN = ABCGenerator()
        logger.info(f"ABCGenerator loaded, ok={getattr(GEN, 'ok', False)}")
    except Exception as e:
        GEN = None
        logger.exception("Failed to init ABCGenerator")

# --- music helpers (same as before) ---
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
    delta = octave - 4
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
        units = round(qlen / unit)
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

# utils
def normalize_abc_text(s:str) -> str:
    if not isinstance(s, str):
        s = s.decode("utf-8", errors="ignore")
    s = s.lstrip('\ufeff')
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.strip() + "\n"
    return s

class ComposeRequest(BaseModel):
    prompt: str
    use_ai: Optional[bool] = None
    valence: Optional[float] = None
    arousal: Optional[float] = None

def _safe_name(s:str) -> str:
    s = re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE)
    return re.sub(r"_+", "_", s).strip("_")

def _insert_log(
    prompt:str, params:Dict, saved_name:str, saved_url:str, saved_file:str, note_count:int,
    valence:Optional[float]=None, arousal:Optional[float]=None,
    multimodal_audio:Optional[str]=None, multimodal_image:Optional[str]=None
) -> int:
    conn = _db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO logs (ts, prompt, key, mode, tempo, bars, meter, mood, low, high,
                          valence, arousal, multimodal_audio, multimodal_image,
                          saved_name, saved_url, saved_file, note_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        prompt, params["key"], params["mode"], params["tempo"], params["bars"],
        params["meter"], params["mood"], params["low"], params["high"],
        valence, arousal, multimodal_audio, multimodal_image,
        saved_name, saved_url, saved_file, note_count
    ))
    conn.commit()
    log_id = cur.lastrowid
    conn.close()
    return log_id

def _insert_mm_log(type_:str, file_path:str, meta:dict):
    conn = _db()
    cur = conn.cursor()
    cur.execute("INSERT INTO multimodal_logs (ts, type, file_path, meta) VALUES (?, ?, ?, ?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), type_, file_path, json.dumps(meta, ensure_ascii=False)))
    conn.commit()
    conn.close()

def build_control_header(params:Dict, prompt:str)->str:
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

# model status
@app.get("/model_status")
def model_status():
    available = False
    info = {}
    if GEN:
        available = getattr(GEN, "ok", False)
        info["model_loaded"] = available
    else:
        info["model_loaded"] = False
    return {"available": available, "info": info}

# multimodal upload endpoints
@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...), meta: str = Form(None)):
    # save file to MM_DIR, return public URL and log
    ext = os.path.splitext(file.filename)[1] or ".webm"
    name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}{ext}"
    path = os.path.join(MM_DIR, name)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    meta_obj = {}
    if meta:
        try:
            meta_obj = json.loads(meta)
        except:
            meta_obj = {"raw_meta": meta}
    _insert_mm_log("audio", path, meta_obj)
    url = f"/mm/{name}"
    return {"ok": True, "path": path, "url": url, "meta": meta_obj}

@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...), meta: str = Form(None)):
    ext = os.path.splitext(file.filename)[1] or ".png"
    name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}{ext}"
    path = os.path.join(MM_DIR, name)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    meta_obj = {}
    if meta:
        try:
            meta_obj = json.loads(meta)
        except:
            meta_obj = {"raw_meta": meta}
    _insert_mm_log("image", path, meta_obj)
    url = f"/mm/{name}"
    return {"ok": True, "path": path, "url": url, "meta": meta_obj}

# compose endpoint respects valence/arousal and multimodal flags
@app.post("/compose")
def compose(req: ComposeRequest, request: Request):
    params = parse_prompt(req.prompt)
    valence = req.valence
    arousal = req.arousal

    abc_text: Optional[str] = None
    ai_used = False
    debug_info = {}

    want_ai = req.use_ai
    attempt_ai = False
    if want_ai is True:
        attempt_ai = True
    elif want_ai is False:
        attempt_ai = False
    else:
        attempt_ai = True if (GEN and getattr(GEN, "ok", False)) else False

    if attempt_ai and GEN and getattr(GEN, "ok", False):
        try:
            control = build_control_header(params, req.prompt)
            # we can append valence/arousal as tags into control header for conditioning
            if valence is not None or arousal is not None:
                control += f"%VALENCE:{valence if valence is not None else ''} ARousal:{arousal if arousal is not None else ''}\n"
            sampled = GEN.sample(control_header=control, max_len=1200, temperature=0.95, top_p=0.9)
            if sampled and isinstance(sampled, str) and sampled.strip():
                abc_text = normalize_abc_text(control + sampled)
                ai_used = True
                debug_info["ai_note"] = "generated_by_model"
            else:
                debug_info["ai_note"] = "model returned empty"
        except Exception as e:
            logger.exception("AI generation failed")
            debug_info["ai_error"] = str(e)

    if not abc_text:
        melody = generate_melody(params)
        abc_text = normalize_abc_text(melody_to_abc(melody, params))
        debug_info["fallback"] = "rule_based"

    # save ABC
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{params['key']}_{params['mode']}_{params['tempo']}bpm_{params['bars']}bars"
    fname = f"{ts}_{_safe_name(base)}.abc"
    fpath = os.path.join(SAVE_DIR, fname)
    try:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(abc_text)
    except Exception as e:
        logger.exception("Failed to write ABC file")
        return JSONResponse({"error":"failed_to_save","detail":str(e)}, status_code=500)

    base_url = str(request.base_url)
    saved_url_abs = urljoin(base_url, f"abc/{fname}")
    saved_url_rel = f"/abc/{fname}"

    note_count = sum(abc_text.count(ch) for ch in list("ABCDEFG"))

    # For now we do not attempt to attach multimodal file paths here, front-end should call upload endpoints and then pass URLs in metadata if desired
    multimodal_audio = None
    multimodal_image = None

    log_id = _insert_log(
        prompt=req.prompt, params=params, saved_name=fname, saved_url=saved_url_rel, saved_file=fpath,
        note_count=note_count, valence=valence, arousal=arousal,
        multimodal_audio=multimodal_audio, multimodal_image=multimodal_image
    )

    resp = {
        "abc": abc_text,
        "params": params,
        "saved_file": fpath,
        "saved_name": fname,
        "saved_url": saved_url_rel,
        "saved_url_abs": saved_url_abs,
        "note_count": note_count,
        "log_id": log_id,
        "ai_used": ai_used,
        "attempted_ai": attempt_ai,
        "debug_info": debug_info
    }

    logger.info(f"compose done id={log_id} ai_used={ai_used} attempted_ai={attempt_ai} saved={fname}")
    return JSONResponse(resp)

# logs endpoint
@app.get("/logs")
def get_logs(page:int = Query(1, ge=1), per_page:int = Query(1, ge=1, le=200)):
    offset = (page - 1) * per_page
    conn = _db()
    rows = conn.execute("SELECT * FROM logs ORDER BY id DESC LIMIT ? OFFSET ?", (per_page, offset)).fetchall()
    total = conn.execute("SELECT COUNT(*) AS c FROM logs").fetchone()["c"]
    conn.close()
    items = [dict(r) for r in rows]
    return {"page": page, "per_page": per_page, "total": total, "items": items}

@app.get("/logs.csv")
def download_logs_csv():
    conn = _db()
    rows = conn.execute("SELECT * FROM logs ORDER BY id DESC").fetchall()
    conn.close()
    buff = StringIO()
    writer = csv.writer(buff)
    headers = rows[0].keys() if rows else [
        "id","ts","prompt","key","mode","tempo","bars","meter","mood","low","high",
        "valence","arousal","multimodal_audio","multimodal_image",
        "saved_name","saved_url","saved_file","note_count"
    ]
    writer.writerow(headers)
    for r in rows:
        writer.writerow([r[h] for h in headers])
    buff.seek(0)
    return StreamingResponse(buff, media_type="text/csv", headers={"Content-Disposition":"attachment; filename=abc_logs.csv"})
