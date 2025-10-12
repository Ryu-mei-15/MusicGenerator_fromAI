# 🎵 Rhythm-Linked Interaction System (Backend)

音楽のOnset（発音開始）に連動して動作するインタラクションシステムのバックエンドです．  
FastAPI + SQLite（SQLAlchemy）ベースで構築されています．  
研究・実験用のプロトタイプ環境として，すぐに動かせる構成になっています．

---

## 🚀 導入手順

# Windows
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# サーバ起動
python -m uvicorn main:app --reload
# または
uvicorn main:app --reload
```

# mac/Linux
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# もしくは手動で：
# pip install fastapi uvicorn sqlalchemy passlib[bcrypt] python-multipart pydantic[email] apscheduler qrcode

# サーバ起動
python -m uvicorn main:app --reload
```
