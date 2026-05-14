# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#!/usr/bin/env python3
"""
SageMaker リソースに AmazonDataZoneProject タグを付与 / 削除するスクリプト。

AWS 公式 GitHub スクリプト (bring_your_own_sagemaker_ai_resources.py) を参考に、
本リポジトリ (sample-sagemaker-ai-navsim-e2e-driving) で作成されたリソースに特化した実装。

公式スクリプトは SageMaker AI ドメインの sagemaker:domain-arn タグで検索するが、
本リポジトリはドメインを使用しないため、プロジェクト名プレフィックスで検索する。

参考:
  https://github.com/aws/Unified-Studio-for-Amazon-Sagemaker/blob/main/migration/sagemaker-ai/bring_your_own_sagemaker_ai_resources.py
  https://docs.aws.amazon.com/sagemaker-unified-studio/latest/userguide/bring-resources-scripts.html

使い方:
  # タグ付与
  python3 tag-resources-for-unified-studio.py \
    --project-id <DataZone プロジェクト ID> \
    --region us-east-1 \
    --project-name sagemaker-ai-ml-pipeline

  # タグ削除
  python3 tag-resources-for-unified-studio.py \
    --project-id <DataZone プロジェクト ID> \
    --region us-east-1 \
    --project-name sagemaker-ai-ml-pipeline \
    --unlink
"""

import argparse
import os
import sys

import boto3
from botocore.exceptions import ClientError

TAG_KEY = "AmazonDataZoneProject"

# タグ付与対象外のリソースタイプ (ドメイン移行スクリプトと同様)
SKIP_RESOURCE_TYPES = {"user-profile", "space", "app", "domain"}


def get_project_resources(tagging_client, sagemaker_client, project_name, region, account_id):
    """プロジェクト名プレフィックスに一致する SageMaker リソースを取得する。

    AWS 公式スクリプトは tag:GetResources で sagemaker:domain-arn タグを検索するが、
    本リポジトリはドメインを使用しないため、SageMaker API で直接検索する。
    """
    resources = []
    prefix = f"{project_name}-"

    # Pipeline
    try:
        paginator = sagemaker_client.get_paginator("list_pipelines")
        for page in paginator.paginate():
            for p in page.get("PipelineSummaries", []):
                if p["PipelineName"].startswith(prefix):
                    resources.append({
                        "type": "pipeline",
                        "name": p["PipelineName"],
                        "arn": p["PipelineArn"],
                    })
    except ClientError as e:
        print(f"  Pipeline の取得に失敗: {e}")

    # Training Job (直近 100 件)
    try:
        paginator = sagemaker_client.get_paginator("list_training_jobs")
        for page in paginator.paginate(
            NameContains=project_name,
            SortBy="CreationTime",
            SortOrder="Descending",
            PaginationConfig={"MaxItems": 100},
        ):
            for job in page.get("TrainingJobSummaries", []):
                resources.append({
                    "type": "training-job",
                    "name": job["TrainingJobName"],
                    "arn": job["TrainingJobArn"],
                })
    except ClientError as e:
        print(f"  Training Job の取得に失敗: {e}")

    # Processing Job (直近 100 件)
    try:
        paginator = sagemaker_client.get_paginator("list_processing_jobs")
        for page in paginator.paginate(
            NameContains=project_name,
            SortBy="CreationTime",
            SortOrder="Descending",
            PaginationConfig={"MaxItems": 100},
        ):
            for job in page.get("ProcessingJobSummaries", []):
                resources.append({
                    "type": "processing-job",
                    "name": job["ProcessingJobName"],
                    "arn": job["ProcessingJobArn"],
                })
    except ClientError as e:
        print(f"  Processing Job の取得に失敗: {e}")

    # Model Package Group
    try:
        paginator = sagemaker_client.get_paginator("list_model_package_groups")
        for page in paginator.paginate(NameContains=project_name):
            for group in page.get("ModelPackageGroupSummaryList", []):
                if group["ModelPackageGroupName"].startswith(prefix):
                    resources.append({
                        "type": "model-package-group",
                        "name": group["ModelPackageGroupName"],
                        "arn": group["ModelPackageGroupArn"],
                    })
    except ClientError as e:
        print(f"  Model Package Group の取得に失敗: {e}")

    # MLflow App
    try:
        mlflow_name = f"{prefix}-mlflow"
        resp = sagemaker_client.list_mlflow_apps()
        for app in resp.get("Summaries", []):
            if app.get("Name") == mlflow_name and app.get("Status") not in ("Deleted", "Deleting"):
                resources.append({
                    "type": "mlflow-app",
                    "name": mlflow_name,
                    "arn": app["Arn"],
                })
                break
    except ClientError as e:
        print(f"  MLflow App の取得に失敗: {e}")

    # Model (直近 100 件)
    try:
        paginator = sagemaker_client.get_paginator("list_models")
        for page in paginator.paginate(
            NameContains=project_name,
            SortBy="CreationTime",
            SortOrder="Descending",
            PaginationConfig={"MaxItems": 100},
        ):
            for model in page.get("Models", []):
                resources.append({
                    "type": "model",
                    "name": model["ModelName"],
                    "arn": model["ModelArn"],
                })
    except ClientError as e:
        print(f"  Model の取得に失敗: {e}")

    # Endpoint
    try:
        paginator = sagemaker_client.get_paginator("list_endpoints")
        for page in paginator.paginate(NameContains=project_name):
            for ep in page.get("Endpoints", []):
                resources.append({
                    "type": "endpoint",
                    "name": ep["EndpointName"],
                    "arn": ep["EndpointArn"],
                })
    except ClientError as e:
        print(f"  Endpoint の取得に失敗: {e}")

    return resources


def get_ecr_repositories(ecr_client, project_name):
    """プロジェクト名プレフィックスに一致する ECR リポジトリを取得する。"""
    repositories = []
    prefix = f"{project_name}-"
    try:
        paginator = ecr_client.get_paginator("describe_repositories")
        for page in paginator.paginate():
            for repo in page.get("repositories", []):
                if repo["repositoryName"].startswith(prefix):
                    repositories.append({
                        "type": "ecr-repository",
                        "name": repo["repositoryName"],
                        "arn": repo["repositoryArn"],
                    })
    except ClientError as e:
        print(f"  ECR リポジトリの取得に失敗: {e}")
    return repositories


def add_tags(sagemaker_client, ecr_client, resources, ecr_repos, project_id):
    """AmazonDataZoneProject タグを付与する。"""
    success = 0
    failed = 0

    # SageMaker リソース
    for r in resources:
        try:
            sagemaker_client.add_tags(
                ResourceArn=r["arn"],
                Tags=[{"Key": TAG_KEY, "Value": project_id}],
            )
            print(f"  ✔ {r['type']}: {r['name']}")
            success += 1
        except Exception as e:
            print(f"  ✘ {r['type']}: {r['name']} - {e}")
            failed += 1

    # ECR リポジトリ
    for r in ecr_repos:
        try:
            ecr_client.tag_resource(
                resourceArn=r["arn"],
                tags=[{"Key": TAG_KEY, "Value": project_id}],
            )
            print(f"  ✔ {r['type']}: {r['name']}")
            success += 1
        except Exception as e:
            print(f"  ✘ {r['type']}: {r['name']} - {e}")
            failed += 1

    return success, failed


def remove_tags(sagemaker_client, ecr_client, resources, ecr_repos):
    """AmazonDataZoneProject タグを削除する。"""
    success = 0
    failed = 0

    # SageMaker リソース
    for r in resources:
        try:
            sagemaker_client.delete_tags(
                ResourceArn=r["arn"],
                TagKeys=[TAG_KEY],
            )
            print(f"  ✔ {r['type']}: {r['name']}")
            success += 1
        except Exception as e:
            print(f"  ✘ {r['type']}: {r['name']} - {e}")
            failed += 1

    # ECR リポジトリ
    # 注意: ECR の AmazonDataZoneProject タグは削除できない場合がある (公式ドキュメント参照)
    for r in ecr_repos:
        try:
            ecr_client.untag_resource(
                resourceArn=r["arn"],
                tagKeys=[TAG_KEY],
            )
            print(f"  ✔ {r['type']}: {r['name']}")
            success += 1
        except Exception as e:
            print(f"  ✘ {r['type']}: {r['name']} - {e}")
            failed += 1

    return success, failed


def main():
    parser = argparse.ArgumentParser(
        description="SageMaker リソースに AmazonDataZoneProject タグを付与/削除する"
    )
    parser.add_argument("--project-id", required=True, help="DataZone プロジェクト ID")
    parser.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"), help="AWS リージョン")
    parser.add_argument("--project-name", default="sagemaker-ai-ml-pipeline", help="リソース名プレフィックス")
    parser.add_argument("--unlink", action="store_true", help="タグを削除する")
    args = parser.parse_args()

    session = boto3.Session(region_name=args.region)
    sagemaker_client = session.client("sagemaker")
    ecr_client = session.client("ecr")
    tagging_client = session.client("resourcegroupstaggingapi")
    account_id = session.client("sts").get_caller_identity()["Account"]

    action = "削除" if args.unlink else "付与"
    print(f"\n{'=' * 60}")
    print(f"AmazonDataZoneProject タグの{action}")
    print(f"{'=' * 60}")
    print(f"  プロジェクト ID : {args.project_id}")
    print(f"  リージョン      : {args.region}")
    print(f"  リソースプレフィックス : {args.project_name}")
    print(f"  アカウント ID   : {account_id}")
    print()

    # リソース検索
    print("リソースを検索中...")
    resources = get_project_resources(
        tagging_client, sagemaker_client, args.project_name, args.region, account_id
    )
    ecr_repos = get_ecr_repositories(ecr_client, args.project_name)

    all_resources = resources + ecr_repos
    if not all_resources:
        print("対象リソースが見つかりませんでした。")
        sys.exit(0)

    print(f"\n対象リソース ({len(all_resources)} 件):")
    for r in all_resources:
        print(f"  - {r['type']}: {r['name']}")

    # タグ付与 / 削除
    print(f"\nタグを{action}中...")
    if args.unlink:
        success, failed = remove_tags(sagemaker_client, ecr_client, resources, ecr_repos)
    else:
        success, failed = add_tags(sagemaker_client, ecr_client, resources, ecr_repos, args.project_id)

    print(f"\n完了: 成功 {success} 件, 失敗 {failed} 件")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
