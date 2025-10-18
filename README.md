# 🎨 AI Watermark Remover

AIを使用した高度な画像ウォーターマーク除去ツール

## ✨ 特徴

- 🤖 **AI駆動**: U-Netアーキテクチャを使用した深層学習ベースの除去
- 🔍 **自動検出**: 複数の検出アルゴリズムによる自動ウォーターマーク検出
- 🎯 **複数の手法**: AI、インペインティング、ハイブリッド方式をサポート
- ⚡ **GPU対応**: CUDA/MPS対応で高速処理
- 🛠️ **カスタマイズ可能**: YAML設定ファイルで柔軟な調整が可能

## 📋 必要要件

- Python 3.8以上
- PyTorch 2.0以上
- OpenCV
- その他の依存関係は`requirements.txt`を参照

## 🚀 インストール

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd watermark
```

### 2. 仮想環境の作成（推奨）

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# または
venv\Scripts\activate  # Windows
```

### 3. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### GPU対応（オプション）

**CUDA（NVIDIA GPU）の場合:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**MPS（Apple Silicon）の場合:**
PyTorch 2.0以上をインストールすれば自動的に対応

## 📖 使い方

### 基本的な使用方法

```bash
# AI方式でウォーターマーク除去
python src/main.py -i sample/image.png -o output/result.png

# 自動検出を使用
python src/main.py -i sample/image.png -o output/result.png --auto-detect

# 比較画像を保存
python src/main.py -i sample/image.png -o output/result.png --save-comparison
```

### 異なる除去方式

```bash
# インペインティング方式
python src/main.py -i sample/image.png -o output/result.png --method inpainting

# ハイブリッド方式（AI + インペインティング）
python src/main.py -i sample/image.png -o output/result.png --method hybrid
```

### カスタムマスクの使用

```bash
# マスク画像を指定
python src/main.py -i sample/image.png -m mask.png -o output/result.png
```

### 詳細オプション

```bash
# すべてのオプションを表示
python src/main.py --help
```

## 🏗️ プロジェクト構造

```
watermark/
├── README.md                    # このファイル
├── requirements.txt             # Python依存関係
├── .gitignore                  # Git除外設定
├── config/
│   └── config.yaml             # 設定ファイル
├── src/
│   ├── __init__.py
│   ├── main.py                 # メインエントリーポイント
│   ├── watermark_remover.py    # AI除去モデル（U-Net）
│   ├── image_processor.py      # 画像前処理・後処理
│   ├── detector.py             # ウォーターマーク検出
│   └── utils.py                # ユーティリティ関数
├── models/
│   └── pretrained/             # 学習済みモデル
├── sample/
│   └── image.png               # サンプル画像
├── output/                     # 出力ディレクトリ
└── tests/                      # テスト
```

## ⚙️ 設定

`config/config.yaml`で以下の設定が可能：

- モデルアーキテクチャ
- 画像処理パラメータ
- 検出アルゴリズム
- 出力形式
- デバイス設定

## 🎯 検出方式

### 1. Adaptive Threshold（適応的閾値処理）
- シンプルで高速
- 単色・高コントラストのウォーターマークに効果的

### 2. Edge Detection（エッジ検出）
- Cannyエッジ検出を使用
- 輪郭のはっきりしたウォーターマークに適している

### 3. Color-based（色ベース）
- 特定の色のウォーターマークを検出
- HSV色空間を使用

## 🧠 AI除去方式

### U-Net アーキテクチャ

本プロジェクトは、医療画像セグメンテーションで実績のあるU-Netアーキテクチャを採用：

- **エンコーダー**: 画像から特徴を抽出
- **ボトルネック**: 圧縮された表現
- **デコーダー**: 高解像度で再構築
- **スキップ接続**: 詳細情報を保持

### 事前学習済みモデル

現在、モデルはランダム初期化です。より良い結果を得るには：

1. ウォーターマーク付き/なしの画像ペアでモデルを学習
2. 公開されている学習済みモデルをダウンロード
3. `config/config.yaml`で`pretrained_path`を設定

## 🔧 トラブルシューティング

### ImportError: cv2

OpenCVがインストールされていない場合：
```bash
pip install opencv-python
```

### CUDA out of memory

画像サイズを小さくするか、CPUモードを使用：
```bash
python src/main.py -i input.png -o output.png --device cpu
```

### 検出が機能しない

異なる検出方式を試してください：
```bash
python src/main.py -i input.png -o output.png --detection-method edge_detection
```

## 📝 使用例

### 例1: 基本的なウォーターマーク除去

```bash
python src/main.py \
  -i sample/image.png \
  -o output/cleaned.png \
  --auto-detect \
  --save-comparison
```

### 例2: カスタム設定

```bash
python src/main.py \
  -i sample/image.png \
  -o output/cleaned.png \
  --method hybrid \
  --detection-method color_based \
  --save-mask \
  --verbose
```

## 🤝 貢献

プルリクエストを歓迎します！大きな変更の場合は、まずissueを開いて変更内容を議論してください。

## ⚠️ 免責事項

このツールは教育目的のみを目的としています。他人の著作権を侵害しないようにしてください。ウォーターマークを除去する前に、画像の所有者から適切な権限を取得してください。

## 📄 ライセンス

MIT License

## 🙏 謝辞

- U-Netアーキテクチャ: Ronneberger et al.
- PyTorchコミュニティ
- OpenCVプロジェクト

## 📞 サポート

問題が発生した場合は、GitHubのissueを作成してください。

---

**注意**: このプロジェクトは開発中です。最良の結果を得るには、学習済みモデルでモデルを学習またはロードしてください。
