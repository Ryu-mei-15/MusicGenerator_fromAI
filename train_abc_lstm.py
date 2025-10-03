# -*- coding: utf-8 -*-
"""
ABCコーパスから“条件付き”ABCトークン列を学習する最小LSTM．
使い方：
  python train_abc_lstm.py --data_dir ABC_CORPUS --epochs 8
→ 学習済み: models/abc_lstm.pt と vocab.json を出力
"""
import argparse, os, json, random, math
from glob import glob
from typing import List
import torch
import torch.nn as nn

def read_abc_files(data_dir:str)->List[str]:
    """
    data_dir 以下を再帰的に走査．.abc / .ABC / .txt を読み込み，
    読めなかったファイルやヘッダ不足を警告ログに出す（デバッグしやすくする）．
    """
    import glob, sys
    patterns = ["**/*.abc", "**/*.ABC", "**/*.txt"]
    paths = []
    for pat in patterns:
        paths += glob.glob(os.path.join(data_dir, pat), recursive=True)

    print(f"[INFO] data_dir = {os.path.abspath(data_dir)}")
    print(f"[INFO] found files = {len(paths)}")
    for p in paths[:10]:
        print(f"  - {p}")
    if len(paths) == 0:
        print("[WARN] 1件も見つかりません．パス指定や拡張子を確認してください．", file=sys.stderr)

    texts = []
    for p in paths:
        ok = False
        for enc in ("utf-8-sig","utf-8","cp932","latin-1"):
            try:
                with open(p, "r", encoding=enc, errors="strict") as f:
                    s = f.read()
                texts.append(s.replace("\r",""))
                ok = True
                break
            except Exception as e:
                continue
        if not ok:
            print(f"[WARN] 読み込み失敗: {p}")
    return texts

def normalize_abc(s:str)->str:
    # 見出しは適度に残す．複数曲入っていてもOK．
    # 制御トークンで条件付けできるよう，先頭にタグを付ける設計にする（学習コーパス側でも許容）
    return s.replace("\r","")

def build_vocab(texts:List[str]):
    # 文字単位でシンプルに（安定）．必要ならBPE等に拡張可能．
    chars = sorted(list(set("".join(texts))))
    stoi = {c:i+4 for i,c in enumerate(chars)}  # 0~3は特殊トークン
    itos = {i:c for c,i in stoi.items()}
    # 特殊トークン
    stoi["<PAD>"]=0; itos[0]=""
    stoi["<BOS>"]=1; itos[1]=""
    stoi["<EOS>"]=2; itos[2]=""
    stoi["<SEP>"]=3; itos[3]=""
    return stoi, itos

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, stoi, ctx_len=256):
        self.stoi = stoi
        self.ctx_len = ctx_len
        ids = []
        for t in texts:
            t = normalize_abc(t)
            x = [1] + [stoi.get(c,0) for c in t] + [2]  # <BOS> text <EOS>
            ids += x + [3]  # <SEP>
        self.ids = ids

    def __len__(self):
        return max(1, len(self.ids)-self.ctx_len)

    def __getitem__(self, idx):
        x = self.ids[idx:idx+self.ctx_len]
        y = self.ids[idx+1:idx+self.ctx_len+1]
        if len(y)<self.ctx_len:  # 末尾パディング
            pad = [0]*(self.ctx_len-len(y))
            x = x + pad
            y = y + pad
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

class LSTMModel(nn.Module):
    def __init__(self, vocab, emb=256, hid=512, layers=2):
        super().__init__()
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, emb)
        self.lstm = nn.LSTM(emb, hid, num_layers=layers, batch_first=True)
        self.lin = nn.Linear(hid, vocab)

    def forward(self, x, h=None):
        e = self.emb(x)
        o, h = self.lstm(e, h)
        logits = self.lin(o)
        return logits, h

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="ABCコーパスのフォルダ")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ctx_len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)

    texts = read_abc_files(args.data_dir)
    texts = [t for t in texts if "K:" in t and "M:" in t]  # 最低限の体裁
    assert len(texts)>0, "ABCファイルが見つかりません．--data_dir を確認してください．"

    stoi, itos = build_vocab(texts)
    os.makedirs("models", exist_ok=True)
    with open("models/vocab.json","w",encoding="utf-8") as f:
        json.dump({"stoi":stoi,"itos":itos}, f, ensure_ascii=False)

    ds = TextDataset(texts, stoi, ctx_len=args.ctx_len)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMModel(vocab=len(stoi)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(1, args.epochs+1):
        model.train()
        total = 0.0; steps = 0
        for x,y in dl:
            x = x.to(device); y = y.to(device)
            opt.zero_grad()
            logits, _ = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); steps += 1
        print(f"[epoch {epoch}] loss={total/max(1,steps):.4f}")
        torch.save(model.state_dict(), "models/abc_lstm.pt")

if __name__ == "__main__":
    main()
