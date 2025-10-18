## 🔧 メモリエラーが出た場合の対処法

Colabで `CUDA out of memory` エラーが出た場合：

### 解決方法1: バッチサイズを下げる（最も効果的）

```bash
# batch_size を 8 → 4 に変更
/Users/yoppii/Desktop/code/watermark/.venv/bin/python src/train.py \
  --clean-dir data/train/clean \
  --watermarked-dir data/train/watermarked \
  --epochs 100 \
  --batch-size 4 \
  --val-split 0.1

# それでもエラーが出る場合: 2 に変更
/Users/yoppii/Desktop/code/watermark/.venv/bin/python src/train.py \
  --clean-dir data/train/clean \
  --watermarked-dir data/train/watermarked \
  --epochs 100 \
  --batch-size 2 \
  --val-split 0.1
```

### 解決方法2: 画像サイズを小さくする

`config/config.yaml` を編集：

```yaml
image:
  input_size: [256, 256]  # デフォルトの [512, 512] から変更
```

### 解決方法3: GPUメモリをクリア（Colabの場合）

```python
import torch
import gc

# GPUメモリをクリア
torch.cuda.empty_cache()
gc.collect()

# GPUメモリ使用状況を確認
print(torch.cuda.memory_summary())
```

### 解決方法4: ランタイムを再起動

Colab上部のメニュー:
1. 「ランタイム」→「ランタイムを再起動」
2. セルを最初から再実行

### バッチサイズの目安

| GPU | VRAM | 推奨 batch_size | 画像サイズ 512x512 |
|-----|------|----------------|-------------------|
| Google Colab T4 | 15GB | 4-8 | ✅ |
| Apple M1/M2 (MPS) | 8-16GB | 2-4 | ✅ |
| CPU | - | 1-2 | ⚠️ 遅い |

### エラーメッセージの例

```
torch.OutOfMemoryError: CUDA out of memory. 
Tried to allocate 1024.00 MiB. 
GPU 0 has a total capacity of 14.74 GiB of which 54.12 MiB is free.
```

**→ この場合**: batch_size を半分に減らしてください

---

**推奨の学習設定 (Colab T4 GPU)**:

```bash
!python src/train.py \
  --clean-dir data/train/clean \
  --watermarked-dir data/train/watermarked \
  --epochs 100 \
  --batch-size 4 \
  --val-split 0.1
```

この設定なら安定して学習できます！
