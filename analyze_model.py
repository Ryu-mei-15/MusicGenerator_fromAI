#!/usr/bin/env python3
# analyze_model.py
import os, json, numpy as np, csv
from model_infer import ABCGenerator
import matplotlib.pyplot as plt

# 設定
OUT_DIR = "analysis_out"
os.makedirs(OUT_DIR, exist_ok=True)

# コントロールヘッダ例（main.py の build_control_header に合わせて作る）
controls = {
    "bright_C_major": "X:1\nT:Prompt: bright, C major\nM:4/4\nQ:1/4=120\nK:C\n%VALENCE:0.8 ARousal:0.6\nbright C major",
    "sad_A_minor":    "X:1\nT:Prompt: sad, A minor\nM:3/4\nQ:1/4=60\nK:Am\n%VALENCE:-0.6 ARousal:-0.4\nsad A minor",
    "neutral_G":      "X:1\nT:Prompt: neutral, G major\nM:4/4\nQ:1/4=100\nK:G\nneutral"
}

# sampling setups
temperatures = [0.6, 1.0, 1.4]
top_ps = [0.8, 0.95]
num_samples = 30  # 各条件ごとのサンプル数（十分な数にする）

# load model
GEN = ABCGenerator()
if not getattr(GEN, "ok", False):
    raise SystemExit("Model not loaded. Check models/abc_lstm.pt and vocab.json")

vocab = GEN.vocab
itos = GEN.itos

def token_label(i):
    return itos.get(i, str(i))

# run experiments
for ctrl_name, ctrl in controls.items():
    for temp in temperatures:
        for tp in top_ps:
            all_probs = []  # list of (steps, vocab) arrays
            generated_texts = []
            for n in range(num_samples):
                txt, probs_list = GEN.sample(control_header=ctrl, max_len=400, temperature=temp, top_p=tp,
                                             return_probs=True)
                generated_texts.append(txt)
                # probs_list: list of arrays shape (vocab,)
                # align: pad to same length (we'll later average per position)
                all_probs.append(probs_list)

            # compute per-step mean probability for top K steps (truncate/pad)
            max_steps = max(len(p) for p in all_probs)
            stacked = np.zeros((len(all_probs), max_steps, vocab), dtype=np.float32)
            for i, p_list in enumerate(all_probs):
                for t_idx, p in enumerate(p_list):
                    stacked[i, t_idx, :] = p
                # pad remaining with small epsilon
                if len(p_list) < max_steps:
                    stacked[i, len(p_list):, :] = 1e-8

            mean_probs = stacked.mean(axis=0)  # shape: (max_steps, vocab)

            # Visualize: heatmap of mean_probs (rows=time, cols=token)
            fig, ax = plt.subplots(figsize=(min(20, vocab*0.4), 6))
            im = ax.imshow(mean_probs.T, aspect='auto', origin='lower')
            ax.set_yticks(range(vocab))
            ax.set_yticklabels([token_label(i) for i in range(vocab)])
            ax.set_xlabel("time step")
            ax.set_title(f"{ctrl_name} T={temp} top_p={tp} (mean probs)")
            plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.02)
            out_png = os.path.join(OUT_DIR, f"{ctrl_name}_T{temp}_tp{tp}_heatmap.png")
            fig.tight_layout()
            fig.savefig(out_png, dpi=150)
            plt.close(fig)
            print("Saved", out_png)

            # token frequency summary across all generated texts
            freq = np.zeros(vocab, dtype=int)
            for txt in generated_texts:
                for ch in txt:
                    # if your tokens are multi-char, you may need to tokenize differently
                    # here we assume single-char tokens that appear in itos values (A,B,C,|,...)
                    for i, tok in itos.items():
                        if tok == ch:
                            freq[int(i)] += 1
                            break
            csv_out = os.path.join(OUT_DIR, f"{ctrl_name}_T{temp}_tp{tp}_freq.csv")
            with open(csv_out, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["token", "count"])
                for i in range(vocab):
                    w.writerow([token_label(i), int(freq[i])])
            print("Saved", csv_out)
