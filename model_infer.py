# -*- coding: utf-8 -*-
"""
学習済み LSTM を使って ABC をサンプリングする推論モジュール．
条件付けは“プロンプトから整形したコントロールヘッダ”で実現．
"""
import os, json, numpy as np, torch
from typing import Dict
from torch import nn

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

class ABCGenerator:
    def __init__(self, model_path="models/abc_lstm.pt", vocab_path="models/vocab.json"):
        self.ok = False
        if not (os.path.exists(model_path) and os.path.exists(vocab_path)):
            return
        with open(vocab_path,"r",encoding="utf-8") as f:
            v = json.load(f)
        # expecting v["stoi"] (str->int) and v["itos"] (int->str)
        self.stoi = v["stoi"]
        # ensure itos keys are ints
        itos_raw = v["itos"]
        self.itos = {int(k):val for k,val in itos_raw.items()} if isinstance(list(itos_raw.keys())[0], str) else itos_raw
        self.vocab = len(self.stoi)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = LSTMModel(vocab=self.vocab).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.ok = True

    def _encode(self, s: str):
        ids = [self.stoi.get(c, self.stoi.get("<UNK>", 0)) for c in s]
        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

    def _decode(self, ids):
        return "".join(self.itos.get(int(i), "") for i in ids)

    def valence_arousal_bias(self, valence: float = 0.0, arousal: float = 0.0):
        b = np.zeros(self.vocab, dtype=np.float32)
        for i in range(self.vocab):
            tok = self.itos.get(i, "")
            if tok in {"E","F","G"}:
                b[i] += max(0.0, valence) * 1.5
            if tok in {"A","B","C","D"}:
                b[i] += max(0.0, -valence) * 1.2
            if tok == " ":
                b[i] += -0.5 * max(0.0, arousal)
            if tok == "|":
                b[i] += 0.3 * max(0.0, -arousal)
        return b

    @torch.inference_mode()
    def sample(self, control_header: str, max_len=1200, temperature=0.9, top_p=0.9,
               valence: float = 0.0, arousal: float = 0.0, return_probs: bool = False):
        if not self.ok:
            return None
        prefix = "<BOS>" + control_header
        x = self._encode(prefix)
        h = None
        out_ids = []
        probs_list = []

        bias_np = self.valence_arousal_bias(valence, arousal)
        bias_t = torch.from_numpy(bias_np).to(self.device).float()

        for _ in range(max_len):
            logits, h = self.model(x, h)
            last = logits[:, -1, :].squeeze(0)
            last = last + bias_t
            last = last / max(1e-5, temperature)
            probs = torch.softmax(last, dim=-1)
            probs_np = probs.detach().cpu().numpy()
            # top-p
            sorted_idx = np.argsort(-probs_np)
            sorted_probs = probs_np[sorted_idx]
            cumsum = np.cumsum(sorted_probs)
            cutoff = np.searchsorted(cumsum, top_p)
            cutoff = min(cutoff, len(sorted_idx)-1)
            allowed = sorted_idx[:cutoff+1]
            allowed_probs = probs_np[allowed]
            allowed_probs = allowed_probs / (allowed_probs.sum()+1e-12)
            next_idx = int(np.random.choice(allowed, p=allowed_probs))
            out_ids.append(next_idx)
            probs_list.append(probs_np.copy())
            if next_idx == self.stoi.get("<EOS>", 2):
                break
            x = torch.tensor([[next_idx]], dtype=torch.long, device=self.device)

        generated = self._decode(out_ids)
        if return_probs:
            return generated, probs_list
        return generated

"""
class ABCGenerator:
    def __init__(self, model_path="models/abc_lstm.pt", vocab_path="models/vocab.json"):
        self.ok = False
        if not (os.path.exists(model_path) and os.path.exists(vocab_path)):
            return
        with open(vocab_path,"r",encoding="utf-8") as f:
            v = json.load(f)
        self.stoi = v["stoi"]; self.itos = {int(k):v for k,v in v["itos"].items()} if isinstance(list(v["itos"].keys())[0], str) else v["itos"]
        self.vocab = len(self.stoi)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = LSTMModel(vocab=self.vocab).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.ok = True

    def _encode(self, s:str):
        ids = [self.stoi.get(c, 0) for c in s]
        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

    def _decode(self, ids):
        return "".join(self.itos.get(int(i), "") for i in ids)

    @torch.inference_mode()
    def sample(self, control_header:str, max_len=1200, temperature=0.9, top_p=0.9):
        if not self.ok:
            return None
        # <BOS> + control + 本文
        prefix = "<BOS>" + control_header
        x = self._encode(prefix)
        h = None
        out_ids = []
        for _ in range(max_len):
            logits, h = self.model(x, h)
            last = logits[:,-1,:].squeeze(0)
            # 温度
            last = last / max(1e-5, temperature)
            probs = torch.softmax(last, dim=-1)
            # top-p
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum <= top_p
            mask[0] = True
            filt_idx = sorted_idx[mask]
            filt_probs = sorted_probs[mask]
            filt_probs = filt_probs / filt_probs.sum()
            next_id = filt_idx[torch.multinomial(filt_probs, 1)].item()
            out_ids.append(next_id)
            # EOSで停止
            if next_id == 2:  # <EOS>
                break
            x = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
        return self._decode(out_ids)
"""