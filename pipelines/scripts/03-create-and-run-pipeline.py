#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
SageMaker Pipeline 定義スクリプト。

Train → RegisterModel → Evaluate の 3 ステップからなる Pipeline を定義・実行する。

【Pipeline の構成】
  Step 1 - Train        : PyTorch Estimator / 汎用 Estimator で学習ジョブを実行
  Step 2 - RegisterModel: 学習済みモデルを SageMaker Model Registry に登録
  Step 3 - Evaluate     : PyTorchProcessor / ScriptProcessor で評価を実行

【フレームワーク別の Estimator / Processor 選択】
  container-pytorch-dlc      : Train = PyTorch Estimator, Evaluate = PyTorchProcessor
  container-pytorch-dlc-byoc : Train = 汎用 Estimator (BYOC), Evaluate = ScriptProcessor (BYOC)
  その他 (BYOC)              : Train = 汎用 Estimator (BYOC), Evaluate = ScriptProcessor (BYOC)

  PyTorch コンテナは Train も Evaluate も AWS マネージドコンテナを使用する。
  ECR へのイメージプッシュが不要で、依存パッケージは source_dir 内の
  requirements.txt で管理する。

【使い方】
  # Pipeline 定義を確認 (JSON 出力)
  python 03-create-and-run-pipeline.py \\
      --project-name sagemaker-ai-ml-pipeline \\
      --region us-east-1 \\
      --role-arn <SageMakerRoleArn>

  # Pipeline を作成/更新
  python 03-create-and-run-pipeline.py ... --create

  # Pipeline を実行
  python 03-create-and-run-pipeline.py ... --create --start
"""

import argparse
import logging
import os

# SageMaker SDK の不要な警告・INFO ログを抑制
logging.getLogger("sagemaker.config").setLevel(logging.WARNING)
logging.getLogger("sagemaker.jumpstart").setLevel(logging.WARNING)

import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.processing import ScriptProcessor
from sagemaker.estimator import Estimator
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.pytorch.processing import PyTorchProcessor
from sagemaker.inputs import TrainingInput
from sagemaker.network import NetworkConfig
from sagemaker.workflow.pipeline_context import PipelineSession

# コンテナごとのインスタンスタイプマッピング
INSTANCE_TYPE_MAP = {
    "container-navsim-transfuser": ("ml.g6.4xlarge", "ml.g6.4xlarge"),
    "container-navsim-ego-mlp": ("ml.c7i.xlarge", "ml.c7i.xlarge"),
    "container-pytorch-dlc": ("ml.c7i.xlarge", "ml.c7i.xlarge"),
    "container-pytorch-dlc-byoc": ("ml.c7i.xlarge", "ml.c7i.xlarge"),
}

def create_pipeline(
    project_name: str,
    region: str,
    role_arn: str,
    container_dir: str = "pipelines/container-navsim-transfuser",
    train_instance_type: str = "ml.c7i.xlarge",
    eval_instance_type: str = "ml.c7i.xlarge",
    subnet_ids: list[str] | None = None,
    security_group_ids: list[str] | None = None,
) -> Pipeline:
    """SageMaker Pipeline を定義して返す。

    Args:
        project_name:         プロジェクト名。S3 バケット名や Pipeline 名のプレフィックスに使用。
        region:               AWS リージョン。
        role_arn:             SageMaker が使用する IAM ロールの ARN。
        container_dir:        コンテナディレクトリのパス。フレームワーク判定にも使用。
        train_instance_type:  学習ジョブのインスタンスタイプ。
        eval_instance_type:   評価ジョブのインスタンスタイプ。
        subnet_ids:           VPC サブネット ID のリスト (VPC 構成時)。
        security_group_ids:   セキュリティグループ ID のリスト (VPC 構成時)。

    Returns:
        定義済みの Pipeline オブジェクト。upsert() で作成/更新、start() で実行できる。
    """

    account_id = boto3.client("sts").get_caller_identity()["Account"]

    # PipelineSession: Pipeline 定義時に使用する特殊なセッション。
    # 通常の Session と異なり、fit() / run() を呼んでも実際にジョブは起動せず、
    # Pipeline の DAG 定義として記録される。
    pipeline_session = PipelineSession()

    # -------------------------------------------------------------------------
    # S3 パス設定
    # -------------------------------------------------------------------------
    # バケット名は CloudFormation スタックの命名規則に合わせている
    dataset_bucket = f"{project_name}-dataset-{account_id}-{region}"

    # container_dir のベース名をコンテナタグとして使用する。
    # 例: "pipelines/container-navsim-transfuser" → "container-navsim-transfuser"
    CONTAINER_TAG = os.path.basename(os.path.normpath(container_dir))

    # コンテナごとに S3 のデータプレフィックスを分ける。
    # コンテナ名 (container_tag) をそのままプレフィックスに使用する。
    # 例: s3://{bucket}/container-navsim-transfuser/train/
    #     s3://{bucket}/container-navsim-transfuser/train/
    train_data_uri = f"s3://{dataset_bucket}/{CONTAINER_TAG}/train"
    test_data_uri = f"s3://{dataset_bucket}/{CONTAINER_TAG}/test"
    model_output_uri = (
        f"s3://{project_name}-model-{account_id}-{region}/output/{framework_suffix}"
    )
    eval_output_uri = (
        f"s3://{project_name}-eval-{account_id}-{region}/output/{framework_suffix}"
    )

    # BYOC の場合に使用する ECR イメージ URI。
    # PyTorch Estimator の場合はマネージドイメージが使われるため参照されない。
    ecr_image_uri = (
        f"{account_id}.dkr.ecr.{region}.amazonaws.com"
        f"/{project_name}-container:{CONTAINER_TAG}"
    )

    # -------------------------------------------------------------------------
    # モデルグループ名
    # -------------------------------------------------------------------------
    # SageMaker Model Registry のモデルパッケージグループ名。
    # フレームワークごとに別グループに登録することで、モデルの系統を管理しやすくする。
    framework_suffix_map = {
        "container-pytorch-dlc": "pytorch",
        "container-pytorch-dlc-byoc": "pytorch-byoc",
        "container-navsim-ego-mlp": "navsim-ego-mlp",
        "container-navsim-transfuser": "navsim-transfuser",
    }
    framework_suffix = framework_suffix_map.get(CONTAINER_TAG, CONTAINER_TAG)
    model_group_name = f"{project_name}-{framework_suffix}"

    # コンテナごとのインスタンスタイプ (引数のデフォルト値を上書き)
    if CONTAINER_TAG in INSTANCE_TYPE_MAP:
        train_instance_type, eval_instance_type = INSTANCE_TYPE_MAP[CONTAINER_TAG]

    # -------------------------------------------------------------------------
    # MLflow App ARN の取得
    # -------------------------------------------------------------------------
    # MLflow App が存在しない場合は空文字列のままにして、
    # train.py / evaluate.py 側でスキップさせる。
    mlflow_app_arn = ""
    mlflow_app_url = ""
    try:
        # CloudFormationスタックからMLflow App ARNを取得
        cfn_client = boto3.client("cloudformation", region_name=region)
        resp = cfn_client.describe_stacks(
            StackName=f"{project_name}-stack"
        )
        outputs = resp["Stacks"][0]["Outputs"]
        for output in outputs:
            if output["OutputKey"] == "MlflowAppArn":
                mlflow_app_arn = output["OutputValue"]
                # ARN format: arn:aws:sagemaker:region:account:mlflow-app/app-xxxxx
                mlflow_app_url = f"https://{mlflow_app_arn.split('/')[-1]}.mlflow.sagemaker.{region}.amazonaws.com"
                break

        if mlflow_app_arn:
            print(f"MLflow Tracking ARN: {mlflow_app_arn}")
            print(f"MLflow Tracking URL: {mlflow_app_url}")
    except Exception as e:
        print(
            f"Warning: MLflow App not found ({e}). "
            "Metrics will not be recorded to MLflow."
        )

    # -------------------------------------------------------------------------
    # ジョブ名のプレフィックス
    # -------------------------------------------------------------------------
    # Training Job / Processing Job の名前を一貫させるために base_job_name を設定する。
    # 例: sagemaker-ai-ml-pipeline-navsim-transfuser-train, sagemaker-ai-ml-pipeline-navsim-transfuser-eval
    train_job_name = f"{project_name}-{framework_suffix}-train"
    eval_job_name = f"{project_name}-{framework_suffix}-eval"

    # -------------------------------------------------------------------------
    # Step 1: Train
    # -------------------------------------------------------------------------
    # CloudWatch Metrics に送信するメトリクスの定義。
    # train.py の print 出力から Regex でメトリクス値を抽出する。
    # Regex のキャプチャグループ ([0-9.]+) が実際の数値として記録される。
    metric_definitions = [
        {"Name": "train:accuracy",  "Regex": r"Training accuracy: ([0-9.]+)"},
        {"Name": "train:precision", "Regex": r"Training precision: ([0-9.]+)"},
        {"Name": "train:recall",    "Regex": r"Training recall: ([0-9.]+)"},
        {"Name": "train:f1",        "Regex": r"Training f1: ([0-9.]+)"},
    ]

    # PyTorch / BYOC で共通の Estimator 引数をまとめる。
    # **common_estimator_kwargs で各 Estimator に展開して渡す。
    common_estimator_kwargs = dict(
        role=role_arn,
        instance_count=1,
        instance_type=train_instance_type,
        output_path=model_output_uri,
        base_job_name=train_job_name,
        environment={
            # train.py が MLflow に記録するために使用する
            "MLFLOW_APP_ARN": mlflow_app_arn,
            "MLFLOW_APP_URL": mlflow_app_url,
            # MLflow / SageMaker Model Registry のモデルグループ名
            "MODEL_GROUP_NAME": model_group_name,
        },
        metric_definitions=metric_definitions,
        # PipelineSession を渡すことで Pipeline の DAG 定義として記録される
        sagemaker_session=pipeline_session,
    )
    # VPC 構成: サブネットとセキュリティグループを Estimator に渡す
    if subnet_ids:
        common_estimator_kwargs["subnets"] = subnet_ids
    if security_group_ids:
        common_estimator_kwargs["security_group_ids"] = security_group_ids

    if CONTAINER_TAG == "container-pytorch-dlc":
        # PyTorch Estimator: AWS マネージドの PyTorch DLC を使用。
        # framework_version に対応するイメージが自動的に選択される。
        # GPU インスタンス (ml.p3 系など) を使う場合は instance_type を変更する。
        estimator = PyTorch(
            entry_point="train.py",
            source_dir=container_dir,
            framework_version="2.5.1",  # サポートバージョン: https://github.com/aws/deep-learning-containers/blob/master/available_images.md
            py_version="py311",
            hyperparameters={
                "epochs": 20,
                "batch-size": 32,
                "learning-rate": 0.001,
            },
            **common_estimator_kwargs,
        )
    else:
        # BYOC (Bring Your Own Container): カスタム ECR イメージを使用。
        # 02-build-and-push-container.sh で ECR にプッシュしたイメージを参照する。
        # entry_point + source_dir を指定することで、train.py の変更時に
        # コンテナの再ビルドなしで反映できる (SDK が S3 経由で注入する)。
        # container-pytorch-dlc-byoc: DLC ベース BYOC (Train も Evaluate も BYOC イメージ)
        estimator = Estimator(
            image_uri=ecr_image_uri,
            entry_point="train.py",
            source_dir=container_dir,
            **common_estimator_kwargs,
        )

    # TrainingStep: Pipeline 内の学習ステップ。
    # inputs の "train" キーが SM_CHANNEL_TRAIN 環境変数としてコンテナに渡される。
    #
    # input_mode:
    #   "FastFile" - S3 からオンデマンドでストリーミング (ダウンロード待ちなし)
    #   "File"     - S3 から EBS に全量ダウンロード後に学習開始 (デフォルト)
    # どちらも同じファイルパス (/opt/ml/input/data/train) でアクセスできるため
    # train.py の変更は不要。
    train_step = TrainingStep(
        name="Train",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=train_data_uri,
                input_mode="FastFile",
                # input_mode="File",  # 小規模データの場合はこちら
            ),
        },
    )

    # -------------------------------------------------------------------------
    # Step 2: RegisterModel
    # -------------------------------------------------------------------------
    # 学習ステップの出力 (S3ModelArtifacts) を Model Registry に登録する。
    # train_step.properties.ModelArtifacts.S3ModelArtifacts は Pipeline 実行時に
    # 動的に解決される参照 (PipelineVariable) であり、定義時点では値が確定しない。
    register_step = RegisterModel(
        name="RegisterModel",
        estimator=estimator,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m7i.xlarge"],
        transform_instances=["ml.m7i.xlarge"],
        # このグループ名で SageMaker Model Registry にモデルパッケージが作成される
        model_package_group_name=model_group_name,
    )

    # -------------------------------------------------------------------------
    # Step 3: Evaluate
    # -------------------------------------------------------------------------
    # フレームワーク別に Processor を選択する。
    # container-pytorch-dlc: AWS マネージドコンテナを使用。
    #   source_dir 内の requirements.txt で追加パッケージを自動インストールする。
    # container-pytorch-dlc-byoc: BYOC イメージを使用。
    common_eval_env = {
        "MLFLOW_APP_ARN": mlflow_app_arn,
        "MLFLOW_APP_URL": mlflow_app_url,
    }

    # VPC 構成: Processor 用の NetworkConfig
    network_config = None
    if subnet_ids and security_group_ids:
        network_config = NetworkConfig(
            subnets=subnet_ids,
            security_group_ids=security_group_ids,
        )

    if CONTAINER_TAG == "container-pytorch-dlc":
        # PyTorchProcessor: マネージド PyTorch DLC で評価。
        eval_processor = PyTorchProcessor(
            framework_version="2.5.1",
            py_version="py311",
            role=role_arn,
            instance_count=1,
            instance_type=eval_instance_type,
            base_job_name=eval_job_name,
            env=common_eval_env,
            network_config=network_config,
            sagemaker_session=pipeline_session,
        )
    else:
        # BYOC: ScriptProcessor でカスタム ECR イメージを使用。
        eval_processor = ScriptProcessor(
            image_uri=ecr_image_uri,
            role=role_arn,
            instance_count=1,
            instance_type=eval_instance_type,
            command=["python3"],
            base_job_name=eval_job_name,
            env=common_eval_env,
            network_config=network_config,
            sagemaker_session=pipeline_session,
        )

    # ProcessingStep の共通引数
    eval_inputs = [
        # 学習ステップが出力した model.tar.gz をコンテナにマウント
        sagemaker.processing.ProcessingInput(
            source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model",
        ),
        # テストデータをコンテナにマウント
        sagemaker.processing.ProcessingInput(
            source=test_data_uri,
            destination="/opt/ml/processing/test",
        ),
    ]
    eval_outputs = [
        # evaluate.py が出力する evaluation.json を S3 に保存
        sagemaker.processing.ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation",
            destination=eval_output_uri,
        ),
    ]

    if CONTAINER_TAG in ("container-pytorch-dlc",):
        # マネージドコンテナ: processor.run() で source_dir を指定して
        # requirements.txt を自動インストールさせる。
        # ProcessingStep は source_dir を直接受け付けないため、
        # processor.run() で step_args を生成して渡す。
        eval_args = eval_processor.run(
            inputs=eval_inputs,
            outputs=eval_outputs,
            code="evaluate.py",
            source_dir=container_dir,
        )
        eval_step = ProcessingStep(
            name="Evaluate",
            step_args=eval_args,
        )
    elif CONTAINER_TAG == "container-navsim-transfuser":
        # 公式版 Transfuser: evaluate.py が transfuser_config.py / transfuser_model.py を
        # import するため、PyTorchProcessor + source_dir でディレクトリごとアップロードする。
        # ScriptProcessor は source_dir をサポートしないため PyTorchProcessor を使用。
        eval_processor_tf = PyTorchProcessor(
            framework_version="2.5.1",
            image_uri=ecr_image_uri,
            role=role_arn,
            instance_count=1,
            instance_type=eval_instance_type,
            base_job_name=eval_job_name,
            env=common_eval_env,
            network_config=network_config,
            sagemaker_session=pipeline_session,
        )
        eval_args = eval_processor_tf.run(
            inputs=eval_inputs,
            outputs=eval_outputs,
            code="evaluate.py",
            source_dir=container_dir,
        )
        eval_step = ProcessingStep(
            name="Evaluate",
            step_args=eval_args,
        )
    else:
        # BYOC: source_dir は不要 (依存は Dockerfile に含まれている)。
        # code にローカルパスを直接指定する。
        eval_args = eval_processor.run(
            inputs=eval_inputs,
            outputs=eval_outputs,
            code=os.path.join(container_dir, "evaluate.py"),
        )
        eval_step = ProcessingStep(
            name="Evaluate",
            step_args=eval_args,
        )
    # RegisterModel が完了してから Evaluate を実行する依存関係を明示する
    eval_step.add_depends_on([register_step])

    # -------------------------------------------------------------------------
    # Pipeline 組み立て
    # -------------------------------------------------------------------------
    pipeline_name = f"{project_name}-{CONTAINER_TAG}-pipeline"
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[train_step, register_step, eval_step],
        sagemaker_session=pipeline_session,
    )

    return pipeline

def main():
    parser = argparse.ArgumentParser(
        description="SageMaker Pipeline を定義・作成・実行する"
    )
    parser.add_argument("--project-name", required=True, help="プロジェクト名")
    parser.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"), help="AWS リージョン")
    parser.add_argument("--role-arn", required=True, help="SageMaker 実行ロールの ARN")
    parser.add_argument(
        "--container-dir",
        default="pipelines/container-navsim-transfuser",
        help="コンテナディレクトリのパス (フレームワーク判定に使用)",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Pipeline を作成/更新する (upsert)",
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="Pipeline の実行を開始する",
    )
    parser.add_argument(
        "--subnet-ids",
        default=None,
        help="VPC サブネット ID (カンマ区切り)。省略時は CFn スタックから自動取得",
    )
    parser.add_argument(
        "--security-group-ids",
        default=None,
        help="セキュリティグループ ID (カンマ区切り)。省略時は CFn スタックから自動取得",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="確認プロンプトをスキップ",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Pipeline の設定情報を表示して終了",
    )
    args = parser.parse_args()

    # --show-config: インスタンスタイプ等の設定を表示して終了
    if args.show_config:
        container_tag = os.path.basename(os.path.normpath(args.container_dir))
        default_train = "ml.c7i.xlarge"
        default_eval = "ml.c7i.xlarge"
        train_it, eval_it = INSTANCE_TYPE_MAP.get(container_tag, (default_train, default_eval))
        print(f"Train:     {train_it}")
        print(f"Evaluate:  {eval_it}")
        return

    # VPC 設定の自動取得: CLI 引数が未指定の場合、CFn スタック出力から取得
    subnet_ids = args.subnet_ids.split(",") if args.subnet_ids else None
    security_group_ids = args.security_group_ids.split(",") if args.security_group_ids else None
    if subnet_ids is None:
        try:
            cfn = boto3.client("cloudformation", region_name=args.region)
            outputs = cfn.describe_stacks(StackName=f"{args.project_name}-stack")[
                "Stacks"
            ][0].get("Outputs", [])
            out = {o["OutputKey"]: o["OutputValue"] for o in outputs}
            if "VpcSubnetIds" in out:
                subnet_ids = out["VpcSubnetIds"].split(",")
                security_group_ids = [out["VpcSecurityGroupId"]]
                print(f"VPC config detected: subnets={subnet_ids}, sg={security_group_ids}")
        except Exception:
            pass  # VPC なし、またはスタック未検出

    pipeline = create_pipeline(
        project_name=args.project_name,
        region=args.region,
        role_arn=args.role_arn,
        container_dir=args.container_dir,
        subnet_ids=subnet_ids,
        security_group_ids=security_group_ids,
    )

    if args.create:
        if not args.auto_approve:
            confirm = input("Pipeline を作成/更新して実行しますか？ [y/N]: ")
            if confirm.lower() != "y":
                print("中止しました。")
                return
        # upsert: Pipeline が存在しない場合は作成、存在する場合は更新する
        pipeline.upsert(role_arn=args.role_arn)
        print(f"Pipeline '{pipeline.name}' created/updated.")

    if args.start:
        execution = pipeline.start()
        print(f"Pipeline execution started: {execution.arn}")

    if not args.create and not args.start:
        # --create / --start を指定しない場合は Pipeline 定義の JSON を標準出力に表示する
        definition = pipeline.definition()
        print(definition)

if __name__ == "__main__":
    main()
