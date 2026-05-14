# Inference Demo App <!-- omit in toc -->

自動運転における経路計画 (trajectory planning) の出力を可視化する Streamlit アプリです。デプロイ済みの SageMaker Endpoint に対して推論リクエストを送信し、結果をリアルタイムに表示します。

モデルは自車の現在の運動状態を入力として受け取り、将来 4 秒間の走行軌跡を予測します。

**入力 (自車の現在状態)**

| パラメータ | 説明 |
|-----------|------|
| 速度 (vx, vy) | 前方向・横方向の速度 (m/s) |
| 加速度 (ax, ay) | 前方向・横方向の加速度 (m/s²) |
| 走行コマンド | FORWARD / LEFT / RIGHT (ナビゲーションの指示に相当) |

**出力 (予測軌跡)**

0.5 秒間隔 × 8 ステップ = 4 秒先までの将来位置 (x, y) を予測します。つまり「0.5 秒後、1.0 秒後、...、4.0 秒後に車がどこにいるか」の 8 点の座標です。

実際の自動運転システムでは、この予測軌跡をもとにステアリング角度やアクセル/ブレーキ量を計算して車両を制御します。

**使用モデル: NAVSIM EgoStatusMLP**

EgoStatusMLP はカメラや LiDAR などのセンサー入力を一切使わず、自車の運動状態のみから将来軌跡を予測する軽量なベースラインモデルです。「センサーなしで車の運動状態だけからどこまで軌跡を予測できるか」を示す位置づけのモデルであり、NAVSIM フレームワークにおけるベースラインとして提供されています。

## 前提条件

- SageMaker Endpoint がデプロイ済みであること (下記「推論エンドポイントのデプロイ」参照)
- AWS 認証情報が設定されていること (`aws configure` または IAM ロール)

> Mock Demo Mode を使えば、エンドポイントなしでもアプリの動作を確認できます。

## 推論エンドポイントのデプロイ

Pipeline で学習したモデルを SageMaker リアルタイム推論エンドポイントとしてデプロイします。

```bash
./infra/sagemaker-ai-inference/scripts/deploy.sh -c navsim-ego-mlp
```

デプロイスクリプトは S3 上の最新モデルアーティファクトを自動検索し、推論スクリプト (`inference.py`) を含む形で再パッケージした後、CloudFormation でエンドポイントを作成します。

## デモアプリの起動

```bash
pip install -r demo-app/requirements.txt
streamlit run demo-app/main.py
```

エンドポイントなしで動作確認する場合は Mock モードで起動します。

```bash
streamlit run demo-app/main.py -- --mock
```

環境変数でデフォルト値を設定できます。

```bash
export AWS_DEFAULT_REGION=us-east-1
streamlit run demo-app/main.py
```

## 使い方

アプリには 2 つのタブがあります。

**🎬 Scenario タブ**: プリセットシナリオ (加速、ブレーキ、左折、右折、車線変更、コーナリング) を選択し、「🔮 Predict」で全フレームの推論を一括実行します。結果は Plotly のアニメーションとして表示され、「▶ Play」で再生、スライダーで任意のフレームに移動できます。

**🎛️ Manual タブ**: スライダーで速度・加速度・走行コマンドを手動設定し、リアルタイムに推論結果を確認します。パラメータを変更するたびに自動で推論が実行されます。

サイドバーの設定:

- **Mock Demo Mode**: エンドポイントなしで物理ベースのシミュレーション予測を使用
- **Show Surrounding Vehicles**: 道路上の周囲車両の表示/非表示

## クリーンアップ

```bash
./infra/sagemaker-ai-inference/scripts/destroy.sh -c navsim-ego-mlp
```
