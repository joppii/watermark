# 🚀 Google Colab で学習する方法

Google Colabを使用すると、無料のGPUで学習が大幅に高速化されます（ローカルの5-10倍速い）。

## 📝 Colab Notebookの作成

1. [Google Colab](https://colab.research.google.com/) にアクセス
2. 「ファイル」→「ノートブックを新規作成」
3. 以下のコードセルを順番に実行

## 🔧 セットアップ（Colabで実行）

### Step 1: GPUを有効化

ノートブック上部のメニューから：
- 「ランタイム」→「ランタイムのタイプを変更」
- 「ハードウェア アクセラレータ」を **T4 GPU** に設定
- 「保存」をクリック

### Step 2: リポジトリをクローン

```python
# GitHubリポジトリをクローン
!git clone https://github.com/joppii/watermark.git
%cd watermark
```

### Step 3: 依存関係をインストール

```python
# 必要なパッケージをインストール
!pip install -q torch torchvision numpy opencv-python Pillow scikit-image scipy tqdm tensorboard PyYAML colorama
```

### Step 4: GPUの確認

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")
```

## 📊 データの準備

### 方法1: Google Driveから画像をアップロード（推奨）

```python
# Google Driveをマウント
from google.colab import drive
drive.mount('/content/drive')

# データをコピー（事前にDriveに画像を配置）
!mkdir -p data/original
!cp -r /content/drive/MyDrive/watermark_images/* data/original/

# 画像数を確認
!find data/original -type f \( -iname "*.png" -o -iname "*.jpg" \) | wc -l
```

### 方法2: Zipファイルをアップロード

```python
from google.colab import files
import zipfile

# Zipファイルをアップロード
uploaded = files.upload()

# 解凍
!mkdir -p data/original
!unzip -q *.zip -d data/original/
!find data/original -type f \( -iname "*.png" -o -iname "*.jpg" \) | wc -l
```

### 方法3: サンプルデータで試す（テスト用）

```python
# サンプル画像をダウンロード（テスト用）
!mkdir -p data/original
# 公開データセットからダウンロード例
!wget -q -P data/original https://placekitten.com/800/600
# または自分のサンプル画像URL
```

## 🎯 学習の実行

### Step 1: ウォーターマークを作成

```python
# テキストウォーターマークを作成
!python src/prepare_data.py text -t "SAMPLE" -o data/watermarks/sample -n 15
!python src/prepare_data.py text -t "COPYRIGHT" -o data/watermarks/copyright -n 15
```

### Step 2: 合成学習データを生成

```python
# 合成データセットを生成（3倍に増強）
!python src/prepare_data.py synthetic \
  -i data/original \
  --output-clean data/train/clean \
  --output-watermarked data/train/watermarked \
  -w data/watermarks/*/*.png \
  -n 3
```

### Step 3: 学習を開始

```python
# 学習開始（GPU使用）
# T4 GPU (15GB): batch-size 8 推奨
# メモリエラーが出たら 4 または 2 に下げてください
!python src/train.py \
  --clean-dir data/train/clean \
  --watermarked-dir data/train/watermarked \
  --epochs 100 \
  --batch-size 8 \
  --val-split 0.1
```

**注意**: Colabではバッチサイズを16-32に増やせます（GPU使用時）

### Step 4: TensorBoardで監視（オプション）

```python
# TensorBoardを起動
%load_ext tensorboard
%tensorboard --logdir runs/watermark_removal
```

## 💾 学習済みモデルのダウンロード

### 学習完了後、モデルをダウンロード

```python
# 最良モデルを探す
!ls -lh models/pretrained/*best.pth

# Google Driveに保存
!cp models/pretrained/*best.pth /content/drive/MyDrive/

# またはローカルにダウンロード
from google.colab import files
files.download('models/pretrained/checkpoint_epoch_99_best.pth')
```

## 🧪 学習済みモデルをテスト

```python
# サンプル画像でテスト
!python src/main.py \
  -i sample/image.png \
  -o output/result.png \
  --method ai \
  --model models/pretrained/checkpoint_epoch_99_best.pth \
  --auto-detect \
  --save-comparison

# 結果を表示
from IPython.display import Image, display
display(Image('output/result_comparison.png'))
```

## ⏱️ 推定時間（Colab GPU使用時）

| データ量 | エポック | 予想時間 |
|---------|---------|---------|
| 500枚 (1500ペア) | 50 | 10-15分 |
| 800枚 (2400ペア) | 50 | 15-25分 |
| 800枚 (2400ペア) | 100 | 30-50分 |
| 2000枚 (6000ペア) | 100 | 1-2時間 |

**ローカルの5-10倍速い！** 🚀

## 🎓 完全なColabノートブック例

```python
# ===========================================
# セル1: セットアップ
# ===========================================
# リポジトリをクローン
!git clone https://github.com/joppii/watermark.git
%cd watermark

# 依存関係をインストール
!pip install -q torch torchvision numpy opencv-python Pillow scikit-image scipy tqdm tensorboard PyYAML colorama

# GPU確認
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")

# ===========================================
# セル2: データアップロード
# ===========================================
from google.colab import drive
drive.mount('/content/drive')

# データをコピー（事前にDriveに画像を配置しておく）
!mkdir -p data/original
!cp -r /content/drive/MyDrive/watermark_training_images/* data/original/
!find data/original -type f \( -iname "*.png" -o -iname "*.jpg" \) | wc -l

# ===========================================
# セル3: データ準備
# ===========================================
# ウォーターマーク作成
!python src/prepare_data.py text -t "SAMPLE" -o data/watermarks/sample -n 15
!python src/prepare_data.py text -t "COPYRIGHT" -o data/watermarks/copyright -n 15

# 合成データ生成
!python src/prepare_data.py synthetic \
  -i data/original \
  --output-clean data/train/clean \
  --output-watermarked data/train/watermarked \
  -w data/watermarks/*/*.png \
  -n 3

# ===========================================
# セル4: 学習開始
# ===========================================
!python src/train.py \
  --clean-dir data/train/clean \
  --watermarked-dir data/train/watermarked \
  --epochs 100 \
  --batch-size 16 \
  --val-split 0.1

# ===========================================
# セル5: TensorBoard（オプション）
# ===========================================
%load_ext tensorboard
%tensorboard --logdir runs/watermark_removal

# ===========================================
# セル6: モデルを保存
# ===========================================
# Google Driveに保存
!cp models/pretrained/*best.pth /content/drive/MyDrive/watermark_model_best.pth
print("✓ Model saved to Google Drive!")

# ===========================================
# セル7: テスト
# ===========================================
!python src/main.py \
  -i sample/image.png \
  -o output/result.png \
  --method ai \
  --model models/pretrained/checkpoint_epoch_99_best.pth \
  --auto-detect \
  --save-comparison

# 結果を表示
from IPython.display import Image, display
display(Image('output/result_comparison.png'))
```

## 📌 重要なヒント

### 1. Colabセッションの制限

- 無料版: 最大12時間
- 長時間学習の場合、チェックポイントから再開可能

### 2. セッション切断対策

```python
# 定期的にチェックポイントをDriveに保存
!cp models/pretrained/*.pth /content/drive/MyDrive/checkpoints/
```

### 3. 学習を再開

```python
!python src/train.py \
  --clean-dir data/train/clean \
  --watermarked-dir data/train/watermarked \
  --resume /content/drive/MyDrive/checkpoints/checkpoint_epoch_50.pth \
  --epochs 100 \
  --batch-size 8
```

### 4. バッチサイズの調整（重要！）

**メモリエラーが出た場合の対処法:**

```bash
# Step 1: バッチサイズを下げる
!python src/train.py \
  --clean-dir data/train/clean \
  --watermarked-dir data/train/watermarked \
  --epochs 100 \
  --batch-size 4 \
  --val-split 0.1

# それでもエラーが出る場合: さらに下げる
!python src/train.py \
  --clean-dir data/train/clean \
  --watermarked-dir data/train/watermarked \
  --epochs 100 \
  --batch-size 2 \
  --val-split 0.1
```

**推奨バッチサイズ:**

- **T4 GPU (15GB VRAM)**: batch_size=8（データ量により4-8）
- **メモリ不足時**: batch_size=4 または 2
- **画像サイズを小さく**: config.yamlで `input_size: [256, 256]`

### 5. GPUメモリをクリア

```python
# 学習前にメモリをクリア
import torch
torch.cuda.empty_cache()
import gc
gc.collect()
```

## 🔗 便利なリンク

- [Google Colab](https://colab.research.google.com/)
- [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [GPU使用量の確認](https://colab.research.google.com/notebooks/gpu.ipynb)

## ⚡ 速度比較

| 環境 | 時間（100エポック） |
|------|-------------------|
| MacBook Pro (Apple Silicon) | 60-90分 |
| Google Colab (T4 GPU) | 10-15分 |
| **高速化** | **6-9倍速** 🚀 |

---

**次のステップ**: 
1. Google Colabで新しいノートブックを作成
2. 上記のコードをコピー&ペースト
3. 順番に実行！
