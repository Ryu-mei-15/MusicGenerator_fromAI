# Text2Melody ABC: 音楽生成システム

## 概要

このプロジェクトは，テキストプロンプトからABC記譜法形式のメロディを生成するWebアプリケーションです．
「明るい感じのハ長調の曲」や「悲しい雰囲気のイ短調のワルツ」といった自然言語の指示に基づいて，ルールベースまたはAI（LSTMモデル）によってメロディを自動生成します．

生成されたメロディはサーバーに保存され，Web API経由で利用できます．また，すべての生成履歴はデータベースに記録されます．

## 主な機能

- **テキストからのメロディ生成**:
    - 曲の調（キー），長調/短調（モード），テンポ，拍子，雰囲気（ムード）などをプロンプトから自動で解釈します．
    - APIリクエストで各パラメータを直接指定することも可能です．
- **2つの生成モード**:
    - **ルールベース生成**: 安定した品質のメロディを確実に生成します．AIモデルがない場合のフォールバックとしても機能します．
    - **AIモデル生成 (オプション)**: PyTorchで実装されたLSTMモデルを使用し，より多様で表現力豊かなメロディを生成します．
        - `valence` (感情価) と `arousal` (覚醒度) のパラメータで，生成される曲の感情的なニュアンスをコントロールできます．
- **ログ機能**:
    - 生成されたすべてのメロディの情報（プロンプト，パラメータ，保存先など）をSQLiteデータベースに記録します．
    - API経由でログの閲覧やCSV形式でのダウンロードが可能です．
- **静的ファイル配信**:
    - 生成された `.abc` ファイルはWebから直接アクセスできます．

## 技術スタック

- **バックエンド**: Python 3, FastAPI
- **AIモデル**: PyTorch
- **データベース**: SQLite
- **サーバー**: Uvicorn

## セットアップ手順

### 1. 前提条件

- Python 3.8 以降
- Git

### 2. リポジトリのクローン

```bash
git clone https://github.com/your-username/MusicGenerator_fromAI.git
cd MusicGenerator_fromAI
```

### 3. 仮想環境の作成と有効化

プロジェクトのルートディレクトリで以下のコマンドを実行し，仮想環境を作成します．

```bash
python -m venv venv
```

次に，作成した仮想環境を有効化します．

- **Windowsの場合**:
  ```bash
  .\venv\Scripts\activate
  ```
- **macOS / Linuxの場合**:
  ```bash
  source venv/bin/activate
  ```

### 4. 依存ライブラリのインストール

必要なPythonライブラリをインストールします．

```bash
pip install "fastapi[all]" torch numpy
```

*(注: プロジェクトに `requirements.txt` ファイルがあれば，代わりに `pip install -r requirements.txt` を実行してください．)*

### 5. (オプション) AIモデルの準備

AIによる生成機能を利用するには，学習済みのモデルファイルが必要です．

- **学習済みモデルがある場合**:
  `models/` ディレクトリを作成し，そこに `abc_lstm.pt` と `vocab.json` を配置してください．

- **モデルを自分で学習する場合**:
  1. **学習データの準備**:
     ABC記譜法のデータセット（`.abc` ファイル群）を任意のディレクトリに用意します．
     テスト用に，以下のスクリプトで小さなダミーコーパスを生成できます．
     ```bash
     python make_mini_corpus.py
     ```
     これにより `ABC_CORPUS/` ディレクトリにデータが生成されます．

  2. **学習の実行**:
     以下のコマンドでモデルの学習を開始します．`--data_dir` には学習データがあるディレクトリを指定してください．
     ```bash
     python train_abc_lstm.py --data_dir ABC_CORPUS --epochs 10
     ```
     学習が完了すると，`models/` ディレクトリに `abc_lstm.pt` と `vocab.json` が保存されます．

### 6. アプリケーションの起動

以下のコマンドでWebサーバーを起動します．

```bash
uvicorn main:app --reload
```

サーバーが起動すると，`http://127.0.0.1:8000` でAPIにアクセスできるようになります．
データベースファイル `abc_logs.sqlite3` は，初回起動時に自動的に作成・初期化されます．

## 使用方法

APIは `curl` や各種プログラミング言語から利用できます．

### メロディを生成する (`/compose`)

`POST /compose` エンドポイントにJSON形式でリクエストを送信します．

**例1: ルールベースで「速くて明るいヘ長調の曲」を生成**

```bash
curl -X POST "http://127.0.0.1:8000/compose" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "速くて明るいヘ長調の曲"}'
```

**例2: AIモデルで「悲しくてゆっくりなイ短調のワルツ」を生成**

`use_ai: true` を指定するとAIモデルが使用されます．`valence` (ネガティブ-ポジティブ) と `arousal` (穏やか-興奮) を指定して感情を制御できます．

```bash
curl -X POST "http://127.0.0.1:8000/compose" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "sad slow a minor waltz",
           "use_ai": true,
           "valence": -0.7,
           "arousal": -0.5
         }'
```

### その他のエンドポイント

- `GET /model_status`: AIモデルが利用可能かどうかを確認します．
- `GET /logs`: 生成履歴の一覧をJSON形式で取得します．
- `GET /logs.csv`: 生成履歴をCSV形式でダウンロードします．

## ディレクトリ構成

```
.
├── ABC/                  # 生成されたABCファイルが保存される
├── models/               # (オプション) 学習済みモデルと語彙ファイル
│   ├── abc_lstm.pt
│   └── vocab.json
├── venv/                 # Python仮想環境
├── main.py               # FastAPIアプリケーション本体
├── model_infer.py        # AIモデルの推論ロジック
├── train_abc_lstm.py     # AIモデルの学習スクリプト
├── make_mini_corpus.py   # 学習用のダミーコーパス生成スクリプト
├── analyze_model.py      # モデル分析用スクリプト
└── abc_logs.sqlite3      # ログデータベース
```