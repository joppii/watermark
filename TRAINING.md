# 🎓 モデル学習ガイド

このガイドでは、AIウォーターマーク除去モデルの学習方法を説明します。

## 📋 目次

1. [学習データの準備](#学習データの準備)
2. [学習の実行](#学習の実行)
3. [学習済みモデルの使用](#学習済みモデルの使用)
4. [高度な設定](#高度な設定)

## 🎯 学習データの準備

### 方法1: 合成ウォーターマークを使用（推奨・簡単）

クリーンな画像があれば、自動的にウォーターマークを追加して学習データを生成できます。

#### Step 1: クリーンな画像を収集

```bash
# 画像を data/original/ に配置
mkdir -p data/original
# あなたの画像をこのフォルダにコピー
```

**推奨画像数**: 最低 500枚、理想は 2000枚以上

#### Step 2: テキストウォーターマークを作成

```bash
/Users/yoppii/Desktop/code/watermark/.venv/bin/python src/prepare_data.py text \
  -t "SAMPLE" \
  -o data/watermarks \
  -n 20
```

これで20種類のテキストウォーターマークが `data/watermarks/` に生成されます。

#### Step 3: 合成データセットを作成

```bash
/Users/yoppii/Desktop/code/watermark/.venv/bin/python src/prepare_data.py synthetic \
  -i data/original \
  --output-clean data/train/clean \
  --output-watermarked data/train/watermarked \
  -w data/watermarks/*.png \
  -n 3
```

パラメータ説明:
- `-i`: クリーン画像のディレクトリ
- `--output-clean`: クリーン画像の出力先
- `--output-watermarked`: ウォーターマーク付き画像の出力先
- `-w`: 使用するウォーターマーク画像
- `-n`: 1枚の画像から何枚のバリエーションを作るか（3なら3倍のデータ）

### 方法2: 実際のウォーターマーク付き画像を使用

ウォーターマーク付き画像とクリーンな画像のペアがある場合:

```bash
# データ構造
data/
├── train/
│   ├── clean/          # クリーン画像
│   └── watermarked/    # ウォーターマーク付き画像
└── val/
    ├── clean/
    └── watermarked/
```

**重要**: ファイル名は同じにしてください（例: `image_001.png` が両方のフォルダに）

## 🚀 学習の実行

### 基本的な学習

```bash
/Users/yoppii/Desktop/code/watermark/.venv/bin/python src/train.py \
  --clean-dir data/train/clean \
  --watermarked-dir data/train/watermarked \
  --epochs 100 \
  --batch-size 4
```

### 合成データセットで学習

```bash
/Users/yoppii/Desktop/code/watermark/.venv/bin/python src/train.py \
  --clean-dir data/original \
  --watermarks data/watermarks/*.png \
  --synthetic \
  --epochs 100 \
  --batch-size 4
```

### GPUを使用する場合

```bash
# 自動検出（推奨）
/Users/yoppii/Desktop/code/watermark/.venv/bin/python src/train.py \
  --clean-dir data/train/clean \
  --watermarked-dir data/train/watermarked \
  --epochs 100 \
  --batch-size 8
```

macOS (Apple Silicon)では、MPSが自動的に使用されます。

### 学習パラメータの説明

| パラメータ | 説明 | デフォルト | 推奨値 |
|----------|------|----------|--------|
| `--epochs` | 学習エポック数 | 100 | 50-200 |
| `--batch-size` | バッチサイズ | 4 | GPU: 8-16, CPU: 2-4 |
| `--val-split` | 検証データの割合 | 0.1 | 0.1-0.2 |
| `--resume` | チェックポイントから再開 | None | 中断時のみ |

### 学習の監視

学習中は以下の情報が表示されます:

```
Epoch 10/100
Train Loss: 0.0234
Val Loss: 0.0198
Learning Rate: 0.000100
```

**TensorBoard**で詳細を確認:

```bash
tensorboard --logdir runs/watermark_removal
```

ブラウザで `http://localhost:6006` を開く

## 💾 学習済みモデルの使用

### モデルの保存場所

学習中、モデルは以下に保存されます:

- `models/pretrained/checkpoint_epoch_X.pth` - 定期的なチェックポイント
- `models/pretrained/checkpoint_epoch_X_best.pth` - 最良モデル

### モデルを使って推論

```bash
/Users/yoppii/Desktop/code/watermark/.venv/bin/python src/main.py \
  -i sample/image.png \
  -o output/result.png \
  --method ai \
  --model models/pretrained/checkpoint_epoch_99_best.pth \
  --auto-detect \
  --save-comparison
```

### 学習の再開

```bash
/Users/yoppii/Desktop/code/watermark/.venv/bin/python src/train.py \
  --clean-dir data/train/clean \
  --watermarked-dir data/train/watermarked \
  --resume models/pretrained/checkpoint_epoch_50.pth \
  --epochs 100
```

## ⚙️ 高度な設定

### config.yaml の編集

`config/config.yaml` でモデル設定をカスタマイズ:

```yaml
model:
  name: "unet_watermark_remover"
  input_channels: 3
  output_channels: 3
  features: [64, 128, 256, 512]  # モデルの容量

training:
  batch_size: 4
  learning_rate: 0.0001
  epochs: 100
  save_interval: 10

image:
  input_size: [512, 512]  # 大きいほど精度up、メモリ消費増
```

### メモリ不足エラーの対処

**エラー**: `CUDA out of memory` または `MPS out of memory`

**解決策**:
1. バッチサイズを減らす: `--batch-size 2`
2. 画像サイズを小さく: `config.yaml` で `input_size: [256, 256]`
3. CPUモードを使用: config.yamlで `device: cpu`

### より良い結果を得るために

1. **データ量を増やす**
   - 最低 1000枚のペア推奨
   - 多様なウォーターマークタイプ

2. **学習時間を長く**
   - 100エポック以上
   - 検証ロスが下がらなくなるまで

3. **データ拡張**
   - デフォルトで有効（反転、回転）
   - `dataset.py` でカスタマイズ可能

4. **ハイパーパラメータ調整**
   - 学習率: 0.0001 が良いスタート
   - L1とMSEの重み調整: `train.py`

## 📊 学習の評価

### 良い学習の兆候

✅ 検証ロスが安定して減少  
✅ 学習ロスと検証ロスが近い（過学習していない）  
✅ 視覚的な結果が良好

### 問題の兆候と対処

❌ **検証ロスが減らない**
- データ量を増やす
- 学習率を下げる
- データ拡張を強化

❌ **過学習（train loss << val loss）**
- データ量を増やす
- 正則化を追加
- モデルサイズを小さく

❌ **学習が不安定**
- 学習率を下げる
- バッチサイズを調整
- グラデーションクリッピング追加

## 🎯 実践例

### 例1: 少量データで素早く試す

```bash
# 1. テキストウォーターマーク作成
/Users/yoppii/Desktop/code/watermark/.venv/bin/python src/prepare_data.py text \
  -t "SAMPLE" -o data/watermarks -n 10

# 2. 50枚の画像で合成データ作成（→150枚に）
/Users/yoppii/Desktop/code/watermark/.venv/bin/python src/prepare_data.py synthetic \
  -i data/original -w data/watermarks/*.png -n 3

# 3. 短期間学習（テスト）
/Users/yoppii/Desktop/code/watermark/.venv/bin/python src/train.py \
  --clean-dir data/train/clean \
  --watermarked-dir data/train/watermarked \
  --epochs 20 \
  --batch-size 4
```

### 例2: 本格的な学習

```bash
# 1. 大量のウォーターマークバリエーション
/Users/yoppii/Desktop/code/watermark/.venv/bin/python src/prepare_data.py text \
  -t "COPYRIGHT" -o data/watermarks/copyright -n 30
/Users/yoppii/Desktop/code/watermark/.venv/bin/python src/prepare_data.py text \
  -t "SAMPLE" -o data/watermarks/sample -n 30

# 2. 1000枚の画像から3000枚のペア生成
/Users/yoppii/Desktop/code/watermark/.venv/bin/python src/prepare_data.py synthetic \
  -i data/original \
  -w data/watermarks/*/*.png \
  -n 3

# 3. 本格学習
/Users/yoppii/Desktop/code/watermark/.venv/bin/python src/train.py \
  --clean-dir data/train/clean \
  --watermarked-dir data/train/watermarked \
  --epochs 150 \
  --batch-size 8
```

## 🔍 トラブルシューティング

### Q: 学習が始まらない
A: データパスを確認してください。`--clean-dir` と `--watermarked-dir` が正しいか。

### Q: メモリエラーが出る
A: `--batch-size 2` に下げて試してください。

### Q: 結果が悪い
A: 
- より多くのデータで学習
- より長く学習（100+ epochs）
- データの品質を確認

### Q: 学習が遅い
A:
- GPUが使われているか確認
- バッチサイズを増やす（メモリが許せば）
- num_workers を調整

## 📚 参考リソース

- [U-Net論文](https://arxiv.org/abs/1505.04597)
- [PyTorch チュートリアル](https://pytorch.org/tutorials/)
- [画像データ拡張のベストプラクティス](https://www.kaggle.com/learn/computer-vision)

---

**次のステップ**: 学習が完了したら、`README.md` の使用方法セクションを参照して、実際の画像でモデルをテストしてください！
