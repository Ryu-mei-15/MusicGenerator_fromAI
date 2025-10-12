# 🎵 Interactive Melody Generation with AI

生成AIを用いた対話的メロディ生成システム

---

## 🚀 概要

本プロジェクトは **学習済みLSTMモデル** を用いて、  
ABC記譜法（ABC notation）形式でメロディを生成する **インタラクティブ作曲環境** です。  
ユーザーの入力（プロンプトやパラメータ）に応じて、  
FastAPIバックエンド + Webフロントエンド上でリアルタイムにAI作曲を行います。

---

## 🧩 システム構成

| コンポーネント | 役割 |
|----------------|------|
| **`main.py`** | FastAPIサーバー。`/compose` エンドポイントでAI生成を呼び出す。 |
| **`model_infer.py`** | 学習済みLSTMモデルをロード・推論するモジュール。AI部分の中核。 |
| **`make_mini_corpus.py`** | コーパス生成スクリプト。ABC形式のテンプレートを使い、人工的な訓練用データを生成。 |
| **`index.html`** | フロントエンド（ブラウザUI）。プロンプトやパラメータを入力し、AI生成をトリガー。 |

---

## 🧠 モデル概要

モデルは単純な **LSTM言語モデル**（PyTorch実装）です。

- 入力：ABC記譜法文字列  
- 出力：次トークン（音符・記号）の確率分布  
- 条件付け：`control_header`（プロンプト＋キー・テンポ・拍子など）  
- サンプリング：温度（`temperature`）と確率上位率（`top_p`）による制御  
- 学習済みパラメータ：`models/abc_lstm.pt`  
- 語彙：`models/vocab.json`

---

## ⚙️ セットアップ手順

### 1. 環境構築

```bash
git clone https://github.com/<yourname>/luvopx-beta.git
cd luvopx-beta
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
※ requirements.txt 例：
```

txt
コードをコピーする
torch
fastapi
uvicorn
jinja2
numpy
matplotlib
### 2. モデル配置
models/abc_lstm.pt と models/vocab.json を配置します。


## ▶️ 実行方法
### 1. バックエンド起動
bash
コードをコピーする
```
uvicorn main:app --reload
```

### 2. ブラウザでアクセス
cpp
コードをコピーする
http://127.0.0.1:8000
### 3. 生成操作
プロンプト欄に「bright melody in C major」などを入力

AI生成を使用 をONにして「生成」

AIが model_infer.ABCGenerator.sample() を呼び出し、ABCメロディを生成

結果が画面下に出力され、コピーまたは保存可能

## 💡 AI生成の流れ
ユーザー入力から build_control_header() によりコントロールヘッダを作成

ABCGenerator.sample() が呼び出される

LSTMモデルにより逐次トークンを生成（確率分布→top-pサンプリング）

結果をABC形式文字列として返却

## 🎼 非AIフォールバック
もしモデルがロードできない場合（例：models/ が存在しないとき）、
main.py は自動的にルールベース生成（generate_melody()）へ切り替えます。
この場合はAIを使わず、確率的な規則ベースでメロディを生成します。

## 🔬 分析・可視化ツール
AI生成の確率構造を可視化・評価するために、
analyze_model.py スクリプトを利用できます。

### 実行方法
bash
コードをコピーする
python analyze_model.py
各条件（valence/arousal、temperature、top_p）で複数サンプルを生成

各トークンの平均確率ヒートマップを出力

結果は analysis_out/ に PNG / CSV で保存されます。

## 💞 感情パラメータ（Valence / Arousal）
model_infer.py の ABCGenerator.sample() は
感情パラメータによる生成バイアスに対応しています。

パラメータ	範囲	意味
valence	[-1.0, 1.0]	明るさ・快さ（正→明るい / 負→暗い）
arousal	[-1.0, 1.0]	活動性・緊張感（正→速い / 負→静かな）

これにより、

python
コードをコピーする
GEN.sample(control_header, valence=0.8, arousal=0.6)
のように呼ぶと、明るく速いメロディにバイアスがかかります。

## 🧭 再学習・強化方法
### 1.データ拡張
既存ABCコーパスを転調（transpose）・リズムずらしで拡張

make_mini_corpus.py により人工データを自動生成

### 2. 再学習（例）
bash
コードをコピーする
python train_lstm.py --corpus data/abc_corpus.txt --epochs 10
### 3. 感情制御（valence/arousal）の強化
model_infer.py 内の valence_arousal_bias() を学習可能MLP化

biasネットを bias_net として fine-tune（教師信号付きコーパスを用意）

評価時に valence / arousal を調整し出力傾向を比較

## 📊 評価指標（例）
指標	内容
トークン多様性	一意トークン数 / 総トークン数
平均音高	ピッチ中心傾向（C4=60基準で換算）
n-gramエントロピー	メロディ構造の複雑さ
感情相関	valence, arousal 値と出力統計の相関係数


## 🧪 今後の展望
Transformer系モデルへの置換（ABC Transformer）

音声合成（MIDI→Audio）統合

対話履歴に基づく生成指向の継続

主観評価実験による感情制御精度の検証

## 👨‍🔬 作者 / 開発背景
本システムは、生成AIを用いたメロディの対話的生成 研究の一環として構築されました。
インタラクティブな作曲支援と、AIモデルの「感情的生成傾向」の解析を目的としています。

## 研究キーワード：

LSTM, Music Generation, ABC Notation, Valence-Arousal, FastAPI, Interactive Composition
│   └── abc_corpus.txt
└── analysis_out/
🧪 今後の展望
Transformer系モデルへの置換（ABC Transformer）

音声合成（MIDI→Audio）統合

対話履歴に基づく生成指向の継続

主観評価実験による感情制御精度の検証

👨‍🔬 作者 / 開発背景
本システムは、生成AIを用いたメロディの対話的生成 研究の一環として構築されました。
インタラクティブな作曲支援と、AIモデルの「感情的生成傾向」の解析を目的としています。

研究キーワード：

LSTM, Music Generation, ABC Notation, Valence-Arousal, FastAPI, Interactive Composition