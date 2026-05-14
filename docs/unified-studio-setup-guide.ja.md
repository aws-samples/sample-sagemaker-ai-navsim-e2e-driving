# SageMaker Unified Studio セットアップガイド

🌐 **Language**: 🇺🇸 [English](unified-studio-setup-guide.md) | 🇯🇵 [日本語](unified-studio-setup-guide.ja.md)

CloudFormation とスクリプトを使った SageMaker Unified Studio のドメイン・プロジェクト作成について説明します。

## 目次

- [デプロイフロー](#デプロイフロー)
- [リソース管理](#リソース管理)
  - [CloudFormation で管理できるリソース](#cloudformation-で管理できるリソース)
  - [CloudFormation で管理できないリソース](#cloudformation-で管理できないリソース)
- [設計上の考慮点](#設計上の考慮点)
  - [IAM ロールの設計](#iam-ロールの設計)
  - [ブループリント設定の注意点](#ブループリント設定の注意点)
  - [認証方式 (IDC ベース vs IAM ベース)](#認証方式-idc-ベース-vs-iam-ベース)
- [トラブルシューティング](#トラブルシューティング)

## デプロイフロー

Unified Studio 環境は以下の 3 つのスクリプトで構成されます。管理コンソールの手動操作は不要です。

| ステップ | 方法 | 内容 |
|---------|------|------|
| Step 1: Foundation | CFn (`foundation.yaml`) | Domain + DomainExecutionRole + DomainServiceRole |
| Step 2: Project | CFn (`project.yaml`) + API/CLI | IAM ロール作成 + ブループリント設定 + Authorization ポリシー + ProjectProfile + Project + メンバー追加 |
| Step 3: Integration | CFn (`integration.yaml`) + API/CLI | Model Registry 連携 (RAM share + DataSource) + MLflow 接続 + タグ付与 |

以下の順序でデプロイします。

```
deploy-foundation.sh
  └─ CFn: Domain + IAM Roles (DomainExecutionRole, DomainServiceRole)
       │
       ▼
deploy-project.sh
  ├─ API: IAM ロール作成 (Provisioning / ManageAccess、存在しない場合のみ)
  ├─ API: put-environment-blueprint-configuration (regionalParameters 付き)
  ├─ API: add-policy-grant (Authorization ポリシー)
  ├─ CFn: ProjectProfile + Project (Tooling ON_CREATE)
  └─ API: create-project-membership (IAM / SSO ユーザーをオーナーに追加)
       │
       ▼
setup-integration.sh
  ├─ CFn: RAM share + DataZone DataSource (Model Registry 連携)
  ├─ API: create-connection (MLflow App 接続)
  └─ API: AmazonDataZoneProject タグ付与 (Pipeline, Training Job, MLflow App 等)
```

削除は逆順で行います。`deploy-foundation.sh --delete` はドメイン配下の全プロジェクト削除、ManageAccess ロール削除、Lake Formation Data Lake Admin のクリーンアップも自動で行います。

## リソース管理

### CloudFormation で管理できるリソース

以下のリソースは CloudFormation で作成・管理できます。

| リソースタイプ | CFn リソース | 備考 |
|--------------|-------------|------|
| DataZone Domain (V2) | `AWS::DataZone::Domain` | `DomainVersion: V2` と `ServiceRole` が必須 |
| Domain Execution Role | `AWS::IAM::Role` | `SageMakerStudioAdminIAMDefaultExecutionPolicy` をアタッチ |
| Domain Service Role | `AWS::IAM::Role` | `SageMakerStudioDomainServiceRolePolicy` をアタッチ、Path は `/service-role/` |
| Project Profile | `AWS::DataZone::ProjectProfile` | `EnvironmentConfigurations` でブループリントを指定 |
| Project | `AWS::DataZone::Project` | `ProjectProfileId` でプロファイルを指定可能 |

参考:

- [AWS::DataZone::Domain](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-datazone-domain.html)
- [AWS::DataZone::ProjectProfile](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-datazone-projectprofile.html)
- [AWS::DataZone::Project](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-datazone-project.html)

### CloudFormation で管理できないリソース

以下のリソースは CFn では管理できず、API/CLI での操作が必要です。`deploy-project.sh` がこれらを自動で設定します。

#### EnvironmentBlueprintConfiguration (V2 ドメインの managed ブループリント)

`AWS::DataZone::EnvironmentBlueprintConfiguration` は CFn リソースとして存在しますが、V2 ドメインの managed ブループリント (Tooling、MLExperiments 等) を指定すると以下のエラーが発生します。

```
Managed Environment Blueprint with <blueprint-id> doesn't exist.
```

CFn ドキュメントにも「In the current release, only the following values are supported: DefaultDataLake and DefaultDataWarehouse」と記載されています。

代替手段として `put-environment-blueprint-configuration` API を使用します。`regionalParameters` には以下の 4 つのパラメータが必要です。`deploy-project.sh` はデフォルト VPC から自動検出して設定します。

| パラメータ | 説明 | 例 |
|-----------|------|-----|
| `VpcId` | VPC ID | `vpc-xxxxxxxxxxxxxxxxx` |
| `Subnets` | サブネット ID (カンマ区切り) | `subnet-aaa,subnet-bbb` |
| `AZs` | アベイラビリティゾーン (カンマ区切り) | `<az-1>,<az-2>` |
| `S3Location` | S3 バケット URI | `s3://amazon-sagemaker-{account}-{region}-{hash}` |

`regionalParameters` を含めずに設定した場合、`get-environment-blueprint-configuration` API では `enabledRegions` が正しく返りますが、Tooling (ON_CREATE) のプロビジョニング時に `Environment blueprint configuration needs to enable atleast one region` エラーが発生します。

設定の確認には `list-environment-blueprint-configurations` を使用します。

```bash
aws datazone put-environment-blueprint-configuration \
  --domain-identifier <domain-id> \
  --environment-blueprint-identifier <blueprint-id> \
  --enabled-regions <region> \
  --provisioning-role-arn <provisioning-role-arn> \
  --manage-access-role-arn <manage-access-role-arn> \
  --regional-parameters '{"<region>":{"VpcId":"...","Subnets":"...","S3Location":"...","AZs":"..."}}' \
  --region <region>
```

参考:

- [PutEnvironmentBlueprintConfiguration](https://docs.aws.amazon.com/datazone/latest/APIReference/API_PutEnvironmentBlueprintConfiguration.html)
- [ListEnvironmentBlueprintConfigurations](https://docs.aws.amazon.com/datazone/latest/APIReference/API_ListEnvironmentBlueprintConfigurations.html)
- [AWS::DataZone::EnvironmentBlueprintConfiguration](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-datazone-environmentblueprintconfiguration.html)

#### ブループリントの Authorization ポリシー

ブループリントの使用を特定のドメインユニットに許可する Authorization ポリシーは、`add-policy-grant` API で設定できます。`ENVIRONMENT_BLUEPRINT_CONFIGURATION` の `entityIdentifier` のフォーマットは `{AWSアカウントID}:{environmentBlueprintId}` です (例: `123456789012:abcdef1234ghij`)。このフォーマットは API ドキュメントに明記されていませんが、管理コンソールの Configure 実行時の CloudTrail イベント (`AddPolicyGrant`) から確認しました。

`deploy-project.sh` は各ブループリントに対して以下の 2 つのポリシーを自動設定します。

- `CREATE_ENVIRONMENT_PROFILE` - ルートドメインユニットの CONTRIBUTOR プロジェクトに環境プロファイル作成を許可
- `CREATE_ENVIRONMENT_FROM_BLUEPRINT` - ルートドメインユニットの CONTRIBUTOR プロジェクトにブループリントからの環境作成を許可

API で同等のポリシーを設定する例です。

```bash
# ルートドメインユニット ID の取得
ROOT_DOMAIN_UNIT_ID=$(aws datazone get-domain \
  --identifier <domain-id> \
  --region <region> \
  --query 'rootDomainUnitId' --output text)

# CREATE_ENVIRONMENT_FROM_BLUEPRINT ポリシーの追加
aws datazone add-policy-grant \
  --domain-identifier <domain-id> \
  --entity-type ENVIRONMENT_BLUEPRINT_CONFIGURATION \
  --entity-identifier "<account-id>:<blueprint-id>" \
  --policy-type CREATE_ENVIRONMENT_FROM_BLUEPRINT \
  --principal "{\"project\":{\"projectDesignation\":\"CONTRIBUTOR\",\"projectGrantFilter\":{\"domainUnitFilter\":{\"domainUnit\":\"${ROOT_DOMAIN_UNIT_ID}\",\"includeChildDomainUnits\":true}}}}" \
  --detail '{"createEnvironmentFromBlueprint":{}}' \
  --region <region>

# CREATE_ENVIRONMENT_PROFILE ポリシーの追加
aws datazone add-policy-grant \
  --domain-identifier <domain-id> \
  --entity-type ENVIRONMENT_BLUEPRINT_CONFIGURATION \
  --entity-identifier "<account-id>:<blueprint-id>" \
  --policy-type CREATE_ENVIRONMENT_PROFILE \
  --principal "{\"project\":{\"projectDesignation\":\"CONTRIBUTOR\",\"projectGrantFilter\":{\"domainUnitFilter\":{\"domainUnit\":\"${ROOT_DOMAIN_UNIT_ID}\",\"includeChildDomainUnits\":false}}}}" \
  --detail "{\"createEnvironmentProfile\":{\"domainUnitId\":\"${ROOT_DOMAIN_UNIT_ID}\"}}" \
  --region <region>
```

既存のポリシーは `list-policy-grants` で確認できます。

```bash
aws datazone list-policy-grants \
  --domain-identifier <domain-id> \
  --entity-type ENVIRONMENT_BLUEPRINT_CONFIGURATION \
  --entity-identifier "<account-id>:<blueprint-id>" \
  --policy-type CREATE_ENVIRONMENT_FROM_BLUEPRINT \
  --region <region>
```

参考:

- [AddPolicyGrant](https://docs.aws.amazon.com/datazone/latest/APIReference/API_AddPolicyGrant.html)
- [ListPolicyGrants](https://docs.aws.amazon.com/datazone/latest/APIReference/API_ListPolicyGrants.html)
- [Assign authorization policies within blueprint configurations](https://docs.aws.amazon.com/datazone/latest/userguide/assign-authorization-policies-in-blueprint-config.html)

#### プロジェクトメンバーシップ

CFn で `AWS::DataZone::Project` を作成した場合、作成者が CloudFormation サービスになるため、ユーザーが自動的にプロジェクトメンバーに追加されません。Unified Studio の UI にアクセスすると「No project access」と表示されます。

`deploy-project.sh` は以下の順序でメンバー追加を試みます。

1. 現在の IAM プリンシパルの DataZone プロファイルを検索し、見つかればプロジェクトオーナーとして追加
2. `.env` の `UNIFIED_STUDIO_SSO_USERS` で指定された SSO ユーザーのプロファイルを検索し、見つかればプロジェクトオーナーとして追加

SSO ユーザーの DataZone プロファイルは、そのユーザーが Unified Studio ポータルに初めてログインした時に作成されます。`deploy-project.sh` 実行時点でまだログインしていない場合はプロファイルが見つからず、メンバー追加がスキップされます。この場合、ログイン後に `deploy-project.sh` を再実行すると、既存リソースはスキップされてメンバー追加のみが実行されます。

API で手動追加する場合は以下のコマンドを使用します。

```bash
# SSO ユーザーのプロファイル ID を取得
aws datazone search-user-profiles \
  --domain-identifier <domain-id> \
  --user-type DATAZONE_SSO_USER \
  --region <region>

# プロジェクトオーナーとして追加
aws datazone create-project-membership \
  --domain-identifier <domain-id> \
  --project-identifier <project-id> \
  --member '{"userIdentifier":"<user-profile-id>"}' \
  --designation PROJECT_OWNER \
  --region <region>
```

参考:

- [CreateProjectMembership](https://docs.aws.amazon.com/datazone/latest/APIReference/API_CreateProjectMembership.html)
- [SearchUserProfiles](https://docs.aws.amazon.com/datazone/latest/APIReference/API_SearchUserProfiles.html)

## 設計上の考慮点

### IAM ロールの設計

Unified Studio では複数の IAM ロールが使用されます。ロールの管理方法と命名規則を理解することが重要です。

#### ロール一覧

| ロール名 | 作成方法 | スコープ | 用途 |
|---------|---------|---------|------|
| `{project}-unified-studio-domain-role` | CFn (foundation) | ドメイン固有 | ドメイン実行ロール |
| `{project}-unified-studio-service-role` | CFn (foundation) | ドメイン固有 | ドメインサービスロール (V2 必須) |
| `AmazonSageMakerProvisioning-{AccountId}` | `deploy-project.sh` (存在しない場合のみ作成) | アカウント共有 | ブループリントのプロビジョニング |
| `AmazonSageMakerManageAccess-{Region}-{DomainId}` | `deploy-project.sh` (存在しない場合のみ作成) | ドメイン固有 | ブループリントのアクセス管理 |

#### Provisioning ロールの共有設計

`AmazonSageMakerProvisioning-{AccountId}` はアカウントに 1 つだけ存在する設計です。

- ロール名にドメイン ID が含まれないため、同一アカウントの全 V2 ドメインで共有される
- 信頼ポリシーの Condition は `aws:SourceAccount` のみ (ドメイン固有の制限なし)
- ロールが存在しない場合、`deploy-project.sh` が自動作成する。管理コンソールの Configure でも作成される
- CFn スタックで管理すべきではない (スタック削除時に他のドメインに影響するため)

参考: [AmazonSageMakerProvisioning-\<domainAccountId\> role](https://docs.aws.amazon.com/sagemaker-unified-studio/latest/adminguide/AmazonSageMakerProvisioning.html)

#### ManageAccess ロールのドメイン固有設計

`AmazonSageMakerManageAccess-{Region}-{DomainId}` はドメインごとに作成されます。

- ロール名にドメイン ID が含まれるため、ドメイン間で衝突しない
- 信頼ポリシーの Condition に `ArnEquals: aws:SourceArn` でドメイン ARN が指定される
- ロールが存在しない場合、`deploy-project.sh` が自動作成する。`deploy-foundation.sh --delete` で自動削除される
- 削除時に Lake Formation の Data Lake Admin からも自動的にクリーンアップされる (削除済みロールが残っていると新しいドメインのプロビジョニングが失敗するため)

#### カスタム名ロールの制約

公式ドキュメントではカスタム名の Provisioning ロールがサポートされていると記載されていますが、実際にはカスタム名ロールで Tooling ブループリントの ON_CREATE プロビジョニングを行うと以下のエラーが発生しました。

```
Caller is not authorized to create environment using blueprintId <blueprint-id>
```

このエラーは Authorization ポリシーが未設定の場合にも発生します。カスタム名ロール自体の問題か、Authorization ポリシーの問題かは切り分けできていません。現在の `deploy-project.sh` では標準名のロールと Authorization ポリシーの両方を設定しているため、この問題は発生しません。

### ブループリント設定の注意点

#### ブループリント ID の取得

managed ブループリントの ID は `list-environment-blueprints --managed` API で取得できます。`deploy-project.sh` はこの API でブループリント ID を動的に取得しているため、ハードコードは不要です。

```bash
aws datazone list-environment-blueprints \
  --domain-identifier <domain-id> \
  --managed \
  --region <region> \
  --query 'items[].{id:id,name:name}' --output table
```

### 認証方式 (IDC ベース vs IAM ベース)

Unified Studio のドメインは、作成時に「Identity Center (IDC) ベース」か「IAM ベース」のどちらかを選択します。1 つのドメインで両方を併用することはできません。

| 項目 | IDC ベース | IAM ベース |
|------|-----------|-----------|
| 認証方式 | AWS IAM Identity Center (SSO) | IAM ロール |
| ユーザー管理 | 個別ユーザー ID を保持 | プロジェクト内で同一ロールを共有 |
| ガバナンス | きめ細かいアクセス制御、カタログ管理 | 開発者の生産性重視 |
| 制限 | なし | 1 アカウント 1 リージョンに 1 つだけ |
| ポータルアクセス | SSO ログイン | IAM ロールでログイン |

同じアカウント・リージョンに IDC ベースのドメインと IAM ベースのドメインを別々に作成して併用することは可能です。

参考:

- [Domains in Amazon SageMaker Unified Studio](https://docs.aws.amazon.com/sagemaker-unified-studio/latest/adminguide/working-with-domains.html)
- [Using Amazon SageMaker Unified Studio Identity center (IDC) and IAM-based domains together](https://aws.amazon.com/blogs/big-data/using-amazon-sagemaker-unified-studio-identity-center-idc-and-iam-based-domains-together/)

#### ポータルアクセスの注意点

IDC ベースのドメインのポータル URL に IAM ロールでアクセスすると「IAM roles are not permitted to access the portal」エラーが表示されます。ポータルのログイン画面に「Sign in with SSO」と「Sign in with AWS IAM」の 2 つの選択肢が表示されますが、IDC ベースのドメインでは「Sign in with AWS IAM」は使用できません。

本プロジェクトでは IDC ベースのドメインを使用しています。SSO ユーザーでアクセスしてください。

#### 左メニューの AI/ML セクションについて

左メニューの AI/ML セクション (MLflow、Models、Training jobs、Inference endpoints) は IAM ベースのドメインでのみ表示されます。IDC ベースのドメインでは、これらの機能には Build メニューからアクセスします。

公式ドキュメントに「For SageMaker Unified Studio domains configured with IAM roles, you will be able to access the following components」と明記されています。

参考: [Navigating Amazon SageMaker Unified Studio](https://docs.aws.amazon.com/sagemaker-unified-studio/latest/userguide/navigating-sagemaker-unified-studio.html)

## トラブルシューティング

### V2 ドメインの ServiceRole 必須

`AWS::DataZone::Domain` で `DomainVersion: V2` を指定する場合、`ServiceRole` プロパティが必須です。CFn ドキュメントでは Required: No と記載されていますが、実際には以下のエラーが発生します。

```
ServiceRole is required for creating a V2 domain.
```

参考: [AWS::DataZone::Domain - ServiceRole](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-datazone-domain.html#cfn-datazone-domain-servicerole)

### ProjectProfile の ON_CREATE と Authorization

`ProjectProfile` の `EnvironmentConfigurations` で `DeploymentMode: ON_CREATE` を指定すると、プロジェクト作成時にブループリント環境が自動プロビジョニングされます。この際、ブループリントの Authorization ポリシーと `regionalParameters` が設定されていないとプロビジョニングが失敗します。

`DeploymentMode: ON_DEMAND` にすると、プロジェクト作成時にはプロビジョニングされず、Compute ページから手動で追加する形になります。ただし、Tooling を ON_DEMAND にすると Spaces (JupyterLab / VS Code) が有効化されず、「Spaces not enabled in this project profile」と表示されます。

### Lake Formation の無効なプリンシパル

ドメイン削除時に ManageAccess ロールを IAM から削除しても、Lake Formation の Data Lake Admin にそのロールの ARN が残ります。この状態で新しいドメインを作成すると、Tooling のプロビジョニング時に以下のエラーが発生します。

```
Failed to add arn:aws:iam::<account>:role/service-role/AmazonSageMakerManageAccess-... as data lake administrator: invalid principal detected.
```

`deploy-foundation.sh --delete` は Lake Formation からの自動クリーンアップを行いますが、手動でロールを削除した場合は以下のコマンドで確認・修正してください。

```bash
# 現在の Data Lake Admin を確認
aws lakeformation get-data-lake-settings --region <region> \
  --query 'DataLakeSettings.DataLakeAdmins'

# 無効なプリンシパルを除外して再設定
aws lakeformation put-data-lake-settings --region <region> \
  --data-lake-settings '{"DataLakeAdmins":[{"DataLakePrincipalIdentifier":"<有効なロール ARN>"}]}'
```

### Configure 時に「Role already exists」エラー

管理コンソールの Configure 実行時に `AmazonSageMakerProvisioning-{AccountId}` が既に存在する場合、エラーが表示されます。ただし、Configure 自体は成功し、プロジェクトプロファイルは正常に作成されます。既存のロールがそのまま使用されます。
