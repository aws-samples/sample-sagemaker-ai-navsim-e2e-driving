# SageMaker Unified Studio Integration Guide <!-- omit in toc -->

🌐 **Language**: 🇺🇸 [English](unified-studio-integration-guide.md) | 🇯🇵 [日本語](unified-studio-integration-guide.ja.md)

This guide describes how to integrate this project with Amazon SageMaker Unified Studio.

Unified Studio displays only resources associated with a project. To use the SageMaker resources created in this project (Model Package Group, MLflow App, Pipeline, etc.) from Unified Studio, each resource requires its own integration setup.

## Table of Contents <!-- omit in toc -->

- [1. Integration Overview](#1-integration-overview)
- [2. Quick Start: Automation with the Setup Script](#2-quick-start-automation-with-the-setup-script)
  - [Prerequisites](#prerequisites)
  - [Usage](#usage)
  - [What the Script Does](#what-the-script-does)
  - [Scope of the Script](#scope-of-the-script)
  - [Verification After Setup](#verification-after-setup)
  - [Running the Tagging Script Standalone](#running-the-tagging-script-standalone)
- [3. Model Registry Integration (DataSource Method)](#3-model-registry-integration-datasource-method)
  - [Overview](#overview)
  - [How It Works](#how-it-works)
  - [Integration Steps](#integration-steps)
    - [Step 1: Create a RAM Share](#step-1-create-a-ram-share)
    - [Step 2: Create a DataZone Data Source](#step-2-create-a-datazone-data-source)
    - [Step 3: Verify Automatic Registration](#step-3-verify-automatic-registration)
- [4. MLflow App Integration](#4-mlflow-app-integration)
  - [Overview](#overview-1)
  - [Method A: Connect via UI](#method-a-connect-via-ui)
  - [Method B: Connect via API (for Automation)](#method-b-connect-via-api-for-automation)
  - [Notes](#notes)
- [5. SageMaker Pipeline Integration](#5-sagemaker-pipeline-integration)
  - [Overview](#overview-2)
  - [Manual Tagging](#manual-tagging)
  - [Verification](#verification)
  - [Notes](#notes-1)
- [6. Training Job / Processing Job Integration](#6-training-job--processing-job-integration)
  - [Overview](#overview-3)
  - [Manual Tagging](#manual-tagging-1)
  - [Notes](#notes-2)
- [7. ECR Repository Integration](#7-ecr-repository-integration)
  - [Overview](#overview-4)
  - [Manual Tagging](#manual-tagging-2)
  - [Notes](#notes-3)
- [8. Notes on Custom Tags](#8-notes-on-custom-tags)
  - [About the AmazonDataZoneProject Tag](#about-the-amazondatazoneproject-tag)
  - [Other DataZone-Related Tags](#other-datazone-related-tags)


## 1. Integration Overview

The integration method differs depending on the resource type. The table below summarizes them.

| Resource | Integration Method | Automation |
|---------|---------|--------|
| Model Package Group | DataZone DataSource (API/CLI) | ✅ `setup-integration.sh` |
| MLflow App | DataZone `create-connection` API + `AmazonDataZoneProject` tag (both required) | ✅ `setup-integration.sh` |
| SageMaker Pipeline | `AmazonDataZoneProject` tag (value = project ID) | ✅ `setup-integration.sh` |
| Training Job | `AmazonDataZoneProject` tag (value = project ID) | ✅ `setup-integration.sh` |
| Processing Job | `AmazonDataZoneProject` tag (value = project ID) | ✅ `setup-integration.sh` |
| ECR repository | `AmazonDataZoneProject` tag (value = project ID) | ✅ `setup-integration.sh` |


The integration mechanism can be broadly divided into two patterns.

1. **DataSource method** (Model Package Group): Create a RAM share + DataZone DataSource, and DataZone scans and syncs the resources
2. **Tag method** (Pipeline, Training Job, Processing Job, ECR): Attach the `AmazonDataZoneProject` tag to the resource with the project ID as the value, and it will be displayed in that project

Reference:

- [Machine learning - Amazon SageMaker Unified Studio](https://docs.aws.amazon.com/sagemaker-unified-studio/latest/userguide/sagemaker.html)
- [Bringing existing resources into Amazon SageMaker Unified Studio](https://docs.aws.amazon.com/sagemaker-unified-studio/latest/userguide/bring-resources-scripts.html)


## 2. Quick Start: Automation with the Setup Script

This repository provides a script that automates the Unified Studio integration. A single script performs Model Registry integration (RAM share + DataSource), MLflow App connection, and `AmazonDataZoneProject` tagging on existing SageMaker resources. For details on how each resource is integrated, see Section 3 and onward.

### Prerequisites

Before running the setup script, the following are required.

- The main stack (`sagemaker-ai-ml-pipeline-stack`) is already deployed
- The Unified Studio domain and project are already created (created by `deploy-foundation.sh` + `deploy-project.sh`)

The domain ID and project ID are shown in the output of `deploy-foundation.sh` / `deploy-project.sh`. To check them via the CLI, use the following commands.

```bash
# List domains
aws datazone list-domains --region <region>

# List projects
aws datazone list-projects \
  --domain-identifier <domain ID> \
  --region <region>
```

### Usage

```bash
# Setup
./infra/unified-studio/scripts/setup-integration.sh \
  --domain-id <domain ID> \
  --project-id <project ID>

# Unlink
./infra/unified-studio/scripts/setup-integration.sh \
  --unlink \
  --domain-id <domain ID> \
  --project-id <project ID>
```

### What the Script Does

At setup time, the following 5 steps are executed.

1. Automatic retrieval of the DataZone connection ID
2. CloudFormation stack deployment (RAM share + DataZone DataSource)
3. Initial DataSource run (first sync to Unified Studio)
4. MLflow App connection (DataZone `create-connection` API)
5. `AmazonDataZoneProject` tagging on existing SageMaker resources

In Step 4, the `datazone create-connection` API connects the MLflow App to the project. The connection is created associated with the Tooling environment, and only when a connection with the same name does not yet exist. Displaying the MLflow App UI requires both the `AmazonDataZoneProject` tag and the connection.

In Step 5, `tag-resources.py` searches for resources matching the project name prefix (`sagemaker-ai-ml-pipeline-`) and attaches the `AmazonDataZoneProject` tag (value = project ID). The target resources are Pipeline, Training Job, Processing Job, Model Package Group, MLflow App, Model, Endpoint, and ECR repository.

At unlink time, the following 3 steps are executed.

1. Remove the `AmazonDataZoneProject` tag from existing SageMaker resources
2. Delete the MLflow App connection
3. Delete the CloudFormation stack (RAM share + DataZone DataSource)

### Scope of the Script

| Resource | Coverage |
|---------|--------|
| Model Package Group (DataSource) | ✅ Automated |
| MLflow App | ✅ Automated (`create-connection` API) |
| Pipeline / Training Job / Processing Job | ✅ Tagging automated |
| Model / Endpoint | ✅ Tagging automated |
| ECR repository | ✅ Tagging automated |

### Verification After Setup

After setup completes, you can verify the following from the Unified Studio **Build** menu.

- **Model Registry → Registered Models**: Models synced via DataSource
- **ML Pipelines**: Pipelines displayed via tag
- **MLflow**: Experiments and runs on the connected MLflow App

The initial sync may take a few minutes to complete.

### Running the Tagging Script Standalone

`tag-resources.py` is called from the setup script, but it can also be run standalone.

```bash
# Add tags
python3 infra/unified-studio/scripts/tag-resources.py \
  --project-id <project ID> \
  --region <region>

# Remove tags
python3 infra/unified-studio/scripts/tag-resources.py \
  --project-id <project ID> \
  --region <region> \
  --unlink
```


Reference:

- [CloudFormation template](../infra/unified-studio/cfn/integration.yaml)
- [Setup script](../infra/unified-studio/scripts/setup-integration.sh)
- [Tagging script](../infra/unified-studio/scripts/tag-resources.py)


## 3. Model Registry Integration (DataSource Method)

### Overview

The models displayed in the Unified Studio Model Registry are limited to those associated with a project. To make Model Package Groups visible from Unified Studio, create a DataSource (not available from the UI).

Reference:

- [Create a data source for SageMaker AI](https://docs.aws.amazon.com/sagemaker-unified-studio/latest/userguide/create-sagemaker-data-source.html)

### How It Works

Model Groups created in Unified Studio are automatically tagged with the following.

```
AmazonDataZoneProject: <project ID>
AmazonDataZoneDomain: <domain ID>
AmazonDataZoneScopeName: dev
AmazonDataZoneUser: <user ID>
```

Since the Model Package Group in this repository does not have these tags, it is not displayed in the Unified Studio Model Registry. Creating a DataSource allows DataZone to scan the Model Package Group and sync it to Unified Studio.

### Integration Steps

Automated by `setup-integration.sh`. If you want to perform this manually, follow the steps below.

#### Step 1: Create a RAM Share

Establish a trust relationship between SageMaker and DataZone.

1. Open the [RAM console](https://console.aws.amazon.com/ram/home)
2. Select **Create resource share**
3. Name: `DataZone-<domain ID>-SageMaker`
4. Resources: Select the target domain from **DataZone Domains**
5. Managed Permissions: `AWSRAMSageMakerServicePrincipalPermissionAmazonDataZoneDomain`
6. Principals: Service principal = `sagemaker.amazonaws.com`
7. Sources: Specify your own account ID (required when sharing with a service principal)
8. **Create resource share**

#### Step 2: Create a DataZone Data Source

Save the following JSON as `create-sagemaker-datasource.json`.

```json
{
  "name": "sagemaker-ai-ml-pipeline-datasource",
  "projectIdentifier": "<project ID>",
  "type": "SAGEMAKER",
  "description": "Integrate the sagemaker-ai-ml-pipeline Model Package Group with Unified Studio",
  "connectionIdentifier": "<connection ID>",
  "configuration": {
    "sageMakerRunConfiguration": {
      "trackingAssets": {
        "SageMakerModelPackageGroupAssetType": [
          "arn:aws:sagemaker:<region>:<account-id>:model-package-group/sagemaker-ai-ml-pipeline-pytorch"
        ]
      }
    }
  },
  "enableSetting": "ENABLED",
  "publishOnImport": "True"
}
```

```bash
aws datazone create-data-source \
  --domain-identifier <domain ID> \
  --cli-input-json file://create-sagemaker-datasource.json \
  --region <region>
```

#### Step 3: Verify Automatic Registration

Once integrated, every Pipeline execution automatically registers models via the following flow.

1. Pipeline execution → Train → RegisterModel step
2. A new version is registered in `sagemaker-ai-ml-pipeline-pytorch`
3. The DataZone data source schedule (or a manual run) syncs it to Unified Studio
4. It appears in Unified Studio under **Build → Model Registry → Registered Models**

Reference:

- [Create a data source for SageMaker AI](https://docs.aws.amazon.com/sagemaker-unified-studio/latest/userguide/create-sagemaker-data-source.html)
- [Model registry - Amazon SageMaker Unified Studio](https://docs.aws.amazon.com/sagemaker-unified-studio/latest/userguide/sagemaker-register-models.xml.html)


## 4. MLflow App Integration

### Overview

This section describes how to connect the MLflow App (`sagemaker-ai-ml-pipeline-mlflow`) in this repository to a Unified Studio project. Unified Studio does not allow you to create a new MLflow App — you "connect" a server that has already been created in SageMaker AI.

Reference:

- [Track experiments using MLflow - Amazon SageMaker Unified Studio](https://docs.aws.amazon.com/sagemaker-unified-studio/latest/userguide/use-mlflow-experiments.html)

### Method A: Connect via UI

You can connect an existing MLflow App from the Unified Studio UI.

1. Sign in to the Unified Studio project
2. Select **MLflow** from the left menu
3. Select **Connect MLflow App**
4. Enter the MLflow App Name
5. Enter the Connection name
6. Enter the MLflow App ARN: `arn:aws:sagemaker:<region>:<account-id>:mlflow-app/app-XXXXXXXXXXXX` (the suffix is a service-generated resource ID)
7. Select **Connect to server**

After connecting, select **Open MLflow** to open the MLflow UI and view experiments, models, and traces.

### Method B: Connect via API (for Automation)

The DataZone `create-connection` API lets you programmatically create an MLflow connection. `setup-integration.sh` uses this method.

Reference:

- [CreateConnection - Amazon DataZone API Reference](https://docs.aws.amazon.com/datazone/latest/APIReference/API_CreateConnection.html)
- [MlflowPropertiesInput](https://docs.aws.amazon.com/datazone/latest/APIReference/API_MlflowPropertiesInput.html)

```bash
aws datazone create-connection \
  --domain-identifier <domain ID> \
  --environment-identifier <environment ID> \
  --name "sagemaker-ai-ml-pipeline-mlflow" \
  --props '{
    "mlflowProperties": {
      "mlflowAppArn": "arn:aws:sagemaker:<region>:<account-id>:mlflow-app/app-XXXXXXXXXXXX"
    }
  }' \
  --region <region>
```

`ConnectionPropertiesInput` is a union type; specify `mlflowAppArn` in the `mlflowProperties` member. Use the Tooling environment ID for `environment-identifier` (auto-detected by `setup-integration.sh`).

### Notes

- To display the MLflow App under Build → MLflow in Unified Studio, both the `AmazonDataZoneProject` tag and the `create-connection` API call (or UI connection) are required. Neither tag alone nor connection alone is sufficient
- Use the Tooling environment ID for `environment-identifier` in `create-connection`. The MLExperiments environment is not needed
- Unified Studio does not allow you to create a new MLflow App. You connect a server already created on the SageMaker AI side
- MLflow experiments and runs are tied to an MLflow App, so using a different MLflow App will split the experiment history
- Connecting the existing MLflow App from this repository lets you reference all previous experiment history from Unified Studio

Reference:

- [Track experiments using MLflow - Amazon SageMaker Unified Studio](https://docs.aws.amazon.com/sagemaker-unified-studio/latest/userguide/use-mlflow-experiments.html)
- [CreateConnection - Amazon DataZone API Reference](https://docs.aws.amazon.com/datazone/latest/APIReference/API_CreateConnection.html)
- [MLflow Experiment Management Guide](mlflow-guide.md)


## 5. SageMaker Pipeline Integration

### Overview

To display a SageMaker Pipeline in Unified Studio, attach the `AmazonDataZoneProject` tag to the Pipeline resource. Set the project ID as the tag value.

Reference:

- [Machine learning - Amazon SageMaker Unified Studio](https://docs.aws.amazon.com/sagemaker-unified-studio/latest/userguide/sagemaker.html)

### Manual Tagging

```bash
aws sagemaker add-tags \
  --resource-arn arn:aws:sagemaker:<region>:<account-id>:pipeline/sagemaker-ai-ml-pipeline-container-pytorch-dlc-pipeline \
  --tags Key=AmazonDataZoneProject,Value=<project ID> \
  --region <region>
```

### Verification

After tagging, the Pipeline appears under **Build → ML Pipelines** in Unified Studio. You can also view execution history and step details.

### Notes

- Pipeline executions created after tagging are automatically displayed in Unified Studio
- Pipeline executions run before tagging are also displayed (since the tag is attached to the Pipeline resource itself)
- Using `setup-integration.sh` automatically attaches the tag (see Section 2)


## 6. Training Job / Processing Job Integration

### Overview

To display Training Jobs and Processing Jobs in Unified Studio, attach the `AmazonDataZoneProject` tag to each job. This applies to both jobs created via Pipeline and jobs created individually.

Reference:

- [Machine learning - Amazon SageMaker Unified Studio](https://docs.aws.amazon.com/sagemaker-unified-studio/latest/userguide/sagemaker.html)

### Manual Tagging

```bash
# Training Job
aws sagemaker add-tags \
  --resource-arn arn:aws:sagemaker:<region>:<account-id>:training-job/<job name> \
  --tags Key=AmazonDataZoneProject,Value=<project ID> \
  --region <region>

# Processing Job
aws sagemaker add-tags \
  --resource-arn arn:aws:sagemaker:<region>:<account-id>:processing-job/<job name> \
  --tags Key=AmazonDataZoneProject,Value=<project ID> \
  --region <region>
```

### Notes

- Training Jobs / Processing Jobs tend to be numerous, so bulk tagging with `tag-resources.py` is convenient (see Section 2)
- The tagging script automatically searches for jobs matching the project name prefix (`sagemaker-ai-ml-pipeline-`)
- If new jobs are created via Pipeline, you need to run the tagging script again


## 7. ECR Repository Integration

### Overview

To display ECR repositories used for BYOC (Bring Your Own Container) in Unified Studio, attach the `AmazonDataZoneProject` tag to the repository.

### Manual Tagging

```bash
aws ecr tag-resource \
  --resource-arn arn:aws:ecr:<region>:<account-id>:repository/sagemaker-ai-ml-pipeline-byoc \
  --tags Key=AmazonDataZoneProject,Value=<project ID> \
  --region <region>
```

### Notes

- Use the `ecr:tag-resource` API for tagging ECR repositories (not the SageMaker `add-tags`)
- `setup-integration.sh` automatically attaches the tag via `tag-resources.py` (see Section 2)


## 8. Notes on Custom Tags

### About the AmazonDataZoneProject Tag

The `AmazonDataZoneProject` tag is a reserved tag used by Unified Studio (DataZone). Note the following.

- Specify the project ID as the tag value (e.g., `abc1defgh2ijkl`)
- Only one project ID can be specified per resource (simultaneous display across multiple projects is not supported)
- Resources created in Unified Studio are automatically tagged with this tag
- If you manually delete the tag, the resource becomes hidden from Unified Studio

### Other DataZone-Related Tags

Resources created in Unified Studio are also automatically tagged with the following.

```
AmazonDataZoneDomain: <domain ID>
AmazonDataZoneScopeName: dev
AmazonDataZoneUser: <user ID>
```

You do not need to attach these tags manually. The `AmazonDataZoneProject` tag alone is sufficient for display in Unified Studio.
