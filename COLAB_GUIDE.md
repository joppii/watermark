# Google Colaboratory での学習ガイド

Google Colaboratoryを使用すると、無料でGPUを使ってモデルを高速に学習できます。

## 🚀 クイックスタート

### Step 1: Colabノートブックを開く

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joppii/watermark/blob/main/train_watermark_removal.ipynb)

上記のバッジをクリックするか、以下のファイルをColabで開いてください：
- `train_watermark_removal.ipynb`

### Step 2: GPU を有効化

1. Colabで: **ランタイム** → **ランタイムのタイプを変更**
2. **ハードウェアアクセラレータ** → **T4 GPU** を選択
3. **保存** をクリック

### Step 3: ノートブックを順番に実行

ノートブックのセルを上から順に実行してください：

1. ✅ Google Drive をマウント
2. ✅ リポジトリをクローン
3. ✅ 依存関係をインストール
4. ✅ データを準備
5. ✅ 学習を開始

## 📁 データの準備方法

### 方法1: Google Drive に画像をアップロード（推奨）

1. Google Drive で以下のフォルダ構造を作成：

```
Google Drive/
└── watermark_data/
    ├── original/          # ← ここにクリーン画像を配置
    └── watermarks/        # 自動生成されます
```

2. `watermark_data/original/` に学習用の画像をアップロード
   - 推奨: 500枚以上
   - 形式: PNG, JPG

3. ノートブックを実行すると、自動的にウォーターマークを追加して学習データを生成

### 方法2: Kaggle Dataset を使用

```python
# Colabノートブックで実行
!pip install kaggle
!mkdir -p ~/.kaggle
# kaggle.json をアップロードして配置
!kaggle datasets download -d <dataset-name>
!unzip <dataset-name>.zip -d data/original/
```

## ⚙️ 学習パラメータ

ノートブック内で調整可能：

```python
EPOCHS = 50          # 学習回数（多いほど精度向上）
BATCH_SIZE = 8       # T4 GPU: 8-16 推奨
```

### GPU別の推奨設定

| GPU | バッチサイズ | 画像サイズ | 学習速度 |
|-----|------------|----------|---------|
| T4 (無料) | 8-12 | 512x512 | 普通 |
| A100 (有料) | 16-32 | 512x512 | 高速 |

## 💾 学習済みモデルの保存

学習完了後、モデルは自動的にGoogle Driveに保存されます：

```
Google Drive/
└── watermark_models/
    ├── checkpoint_epoch_49.pth
    └── checkpoint_epoch_49_best.pth  # ← 最良モデル
```

## 📥 モデルのダウンロード

### 方法1: Google Drive から直接

Google Drive の `watermark_models/` フォルダからダウンロード

### 方法2: Colabから直接ダウンロード

ノートブックの最後のセルを実行すると、zipファイルが作成されます。

## 🏠 ローカルでの使用

ダウンロードしたモデルをローカルで使用：

```bash
# ローカル環境で実行
python src/main.py \
  -i input_image.png \
  -o output_image.png \
  --method ai \
  --model path/to/checkpoint_epoch_49_best.pth \
  --auto-detect \
  --save-comparison
```

## 📊 学習の監視

### TensorBoard を使用

ノートブック内で TensorBoard が起動します：

- 学習ロスの推移
- 検証ロスの推移
- 学習率の変化

### 学習状況の確認

```python
# Colabで実行中に確認
!ls -lh models/pretrained/
```

## ⏸️ 学習の中断と再開

### 中断した場合

学習を中断しても大丈夫！チェックポイントから再開できます：

```python
# ノートブックで実行
!python src/train.py \
  --clean-dir {LOCAL_DATA_DIR}/train/clean \
  --watermarked-dir {LOCAL_DATA_DIR}/train/watermarked \
  --resume models/pretrained/checkpoint_epoch_25.pth \
  --epochs 100
```

## ⚠️ よくある問題

### 「セッション切断」エラー

**原因**: Colabの無料版は12時間でセッションが切れます

**対処**:
1. 定期的にチェックポイントを保存（自動）
2. Google Drive に保存されたモデルから再開
3. ブラウザのタブを開いたままにする

### 「GPU メモリ不足」エラー

**対処**:
```python
# バッチサイズを減らす
BATCH_SIZE = 4  # または 2
```

### 「ランタイムがクラッシュ」

**対処**:
1. ランタイムを再起動
2. 最初のセルから再実行
3. チェックポイントから学習を再開

## 💡 ヒント

### より良い結果を得るために

1. **データ量を増やす**
   - 最低: 500枚
   - 推奨: 2000枚以上
   - 理想: 5000枚以上

2. **多様なウォーターマーク**
   - 異なるテキスト
   - 異なる位置
   - 異なる透明度

3. **十分な学習時間**
   - 小規模データ: 50エポック
   - 中規模データ: 100エポック
   - 大規模データ: 150-200エポック

### 学習を高速化

```python
# num_workers を増やす（データローダー）
# train.py の get_dataloader 関数で設定可能
num_workers=4  # デフォルト
```

## 🔄 ワークフロー例

### 1回目（テスト学習）

```
1. 50枚の画像で試す
2. 20エポック学習
3. 結果を確認
4. パラメータ調整
```

### 2回目（本番学習）

```
1. 2000枚の画像を準備
2. 100エポック学習
3. モデルをダウンロード
4. ローカルで使用
```

## 📞 サポート

問題が発生した場合：

1. エラーメッセージを確認
2. [TRAINING.md](TRAINING.md) の「トラブルシューティング」セクションを参照
3. GitHubのIssueを作成

## 🎓 学習完了後

学習が完了したら：

1. ✅ モデルを Google Drive からダウンロード
2. ✅ ローカル環境でテスト
3. ✅ 実際の画像で効果を確認
4. ✅ 必要に応じて追加学習

---

**Happy Training! 🚀**

詳細な学習ガイドは [TRAINING.md](TRAINING.md) を参照してください。
