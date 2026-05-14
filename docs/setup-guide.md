# Getting Started Guide <!-- omit in toc -->

🌐 **Language**: 🇺🇸 [English](setup-guide.md) | 🇯🇵 [日本語](setup-guide.ja.md)

Guide for building and running NAVSIM autonomous driving model training pipelines on Amazon SageMaker AI. Covers AWS infrastructure deployment (S3, ECR, SageMaker AI Notebook, MLflow), pipeline execution, inference endpoint deployment, demo app launch, and SageMaker Unified Studio integration.

- [Initial Setup (One-time)](#initial-setup-one-time)
  - [Step 1: Configure .env File](#step-1-configure-env-file)
    - [GitHub Repository Integration](#github-repository-integration)
    - [VPC Configuration](#vpc-configuration)
  - [Step 2: Run Deployment](#step-2-run-deployment)
- [Using the Development Environment](#using-the-development-environment)
  - [JupyterLab](#jupyterlab)
  - [MLflow UI](#mlflow-ui)
- [Pipeline Execution (Repeatable)](#pipeline-execution-repeatable)
  - [Step 1: Prepare Source Code](#step-1-prepare-source-code)
  - [Step 2: Upload Dataset](#step-2-upload-dataset)
  - [Step 3: Container Build & ECR Push](#step-3-container-build--ecr-push)
  - [Step 4: Create & Run Pipeline](#step-4-create--run-pipeline)
  - [Step 5: Check Execution Status](#step-5-check-execution-status)
  - [Running with Jupyter Notebooks](#running-with-jupyter-notebooks)
- [CARLA Simulation Demo](#carla-simulation-demo)
- [Inference Endpoint and Demo App](#inference-endpoint-and-demo-app)
  - [Step 1: Deploy Inference Endpoint](#step-1-deploy-inference-endpoint)
  - [Step 2: Launch Demo App](#step-2-launch-demo-app)
  - [Delete Inference Endpoint](#delete-inference-endpoint)
- [SageMaker Unified Studio Integration (Optional)](#sagemaker-unified-studio-integration-optional)
  - [Step 1: Create Unified Studio Domain](#step-1-create-unified-studio-domain)
  - [Step 2: Create Project](#step-2-create-project)
  - [Step 3: Link SageMaker Resources](#step-3-link-sagemaker-resources)
  - [Unlinking](#unlinking)
- [Testing](#testing)
  - [Lint Check (run-lint.sh)](#lint-check-run-lintsh)
  - [Integration Test (run-tests.sh)](#integration-test-run-testssh)
- [Cleanup](#cleanup)
- [Troubleshooting](#troubleshooting)
- [Appendix](#appendix)
  - [AWS Resources Created](#aws-resources-created)
  - [Related Information](#related-information)
    - [AWS Documentation](#aws-documentation)
    - [Related Workshops and Samples](#related-workshops-and-samples)

## Initial Setup (One-time)

CloudFormation creates S3 buckets, an ECR repository, a SageMaker AI Notebook, an MLflow App, and other AWS resources in a single deployment. Sample datasets are automatically uploaded to S3 during deployment. Stack creation takes approximately 10-15 minutes.

> ⚠️ [Configure AWS CLI credentials](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-authentication.html) before deployment.
>
> **IAM Identity Center (SSO)**:
> ```bash
> aws sso login --profile <your-profile>
> export AWS_PROFILE=<your-profile>
> ```
>
> **IAM user access keys**:
> ```bash
> aws configure
> ```
>
> **Verify authentication**:
> ```bash
> aws sts get-caller-identity
> ```

### Step 1: Configure .env File

Create a `.env` file and verify/set the deployment region. GitHub repository integration and VPC configuration can also be set here.

```bash
cp .env.example .env
# Edit .env to set region and other configuration
```

Settings configurable via `.env` file or environment variables:

| Variable | Default | Description |
|---------|---------|-------------|
| `AWS_DEFAULT_REGION` | `us-east-1` | Deployment region |
| `NOTEBOOK_IDLE_TIMEOUT_MIN` | `60` | Minutes of idle time before Notebook auto-stop (minimum 5) |
| `GITHUB_REPO` | (none) | GitHub repository URL to link with Notebook (optional). Username is auto-extracted from URL |
| `GITHUB_PAT` | (none) | GitHub Personal Access Token (for private repositories) |
| `ENABLE_VPC` | `false` | Enable VPC configuration. Set `true` to place all components inside a VPC |
| `VPC_ID` | (none) | Existing VPC ID (creates a new VPC if empty) |
| `SUBNET_IDS` | (none) | Existing subnet IDs (comma-separated, required when `VPC_ID` is specified) |
| `SECURITY_GROUP_ID` | (none) | Existing security group ID (required when `VPC_ID` is specified) |
| `CREATE_VPC_ENDPOINTS` | `true` | Create VPC Endpoints. Set `false` if existing VPC already has endpoints |

Example `.env` configuration:

```bash
# GitHub repository integration (optional)
GITHUB_REPO="https://github.com/your-username/your-repo.git"
GITHUB_PAT="your-github-personal-access-token"

# VPC configuration (optional)
ENABLE_VPC=true
```

> ⚠️ The `.env` file is included in `.gitignore` and will not be committed to the repository. Be careful not to commit PATs directly.

##### GitHub Repository Integration

When `GITHUB_REPO` is set, the deploy script automatically performs the following before CloudFormation deployment:

1. Checks for repository existence using `gh` command, creates it if not found
2. Pushes repository contents to the remote repository

During CloudFormation deployment, Secrets Manager secret creation and SageMaker CodeRepository linking are also performed automatically.

For GitHub Personal Access Token (PAT) creation, see the [GitHub documentation](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens). Classic PATs require the `repo` scope; Fine-grained PATs require `Contents` permission (Read-only) for the target repository.

##### VPC Configuration

To enable VPC, set `ENABLE_VPC=true` in `.env`. VPC, subnets, security groups, NAT Gateway, and VPC Endpoints are automatically created.

```bash
# Add to .env
ENABLE_VPC=true
```

To use an existing VPC, specify the following resource IDs:

- `VPC_ID`: Target VPC ID. `EnableDnsSupport` and `EnableDnsHostnames` must be enabled
- `SUBNET_IDS`: Private subnet IDs for Notebook / Training Job / Processing Job placement (comma-separated, 2 or more in different AZs). Must have routes to NAT Gateway and S3 Gateway Endpoint in route table
- `SECURITY_GROUP_ID`: Security group ID shared by Notebook and Training / Processing Jobs. For distributed training, a self-referencing ingress rule allowing all traffic within the same SG is required

```bash
# Add to .env
ENABLE_VPC=true
VPC_ID="vpc-xxxxxxxxxxxxxxxxx"
SUBNET_IDS="subnet-xxxxxxxxxxxxxxxxx,subnet-yyyyyyyyyyyyyyyyy"
SECURITY_GROUP_ID="sg-xxxxxxxxxxxxxxxxx"
CREATE_VPC_ENDPOINTS=false  # If existing VPC already has endpoints
```

For VPC configuration details, see [VPC Configuration Implementation](vpc-implementation.ja.md).

### Step 2: Run Deployment

```bash
./infra/sagemaker-ai-ml-pipeline/scripts/deploy.sh [STACK_NAME] [PROJECT_NAME]
```

All parameters are optional.

| Parameter | Default | Description |
|-----------|---------|-------------|
| STACK_NAME | `sagemaker-ai-ml-pipeline-stack` | CloudFormation stack name |
| PROJECT_NAME | `sagemaker-ai-ml-pipeline` | Resource naming prefix |

## Using the Development Environment

After deployment, JupyterLab and MLflow UI are accessible via browser. Both use presigned URL authentication, and browser sessions are valid for 4 hours.

### JupyterLab

The JupyterLab environment on the Notebook instance. Used for code editing, pipeline execution, and data inspection.

```bash
./infra/sagemaker-ai-ml-pipeline/scripts/open-jupyterlab.sh [PROJECT_NAME]
```

The Notebook automatically stops after a period of idle time (default: 60 minutes). If you run the script after the Notebook has stopped, it will automatically wait for restart before opening JupyterLab.

Kiro CLI is automatically installed via Lifecycle Config, allowing you to use AI coding tools from the JupyterLab terminal. Log in with device flow on first use:

```bash
kiro-cli login --use-device-flow
```

Open the displayed device code and URL in your local PC browser to complete authentication, then `kiro-cli` commands become available.

### MLflow UI

Web UI for comparing experiment metrics and managing model versions.

```bash
./infra/sagemaker-ai-ml-pipeline/scripts/open-mlflow.sh [PROJECT_NAME]
```

For details on recording metrics and registering models with the MLflow SDK, see the [MLflow Experiment Management Guide](mlflow-guide.ja.md).

## Pipeline Execution (Repeatable)

> ⚠️ **Run all commands in this section and below from the JupyterLab terminal.**

Use `run-pipeline.sh` to execute Steps 1-4 in one command.

PyTorch containers (`container-pytorch-dlc`, etc.) include sample data, so you can run `run-pipeline.sh` directly. NAVSIM containers (`container-navsim-ego-mlp`, etc.) require data preparation with `prepare_dataset.sh` first, then run with `--skip-upload`.

Available containers:

| Container | Description | Build | GPU | Data Prep | Pipeline |
|-----------|-------------|-------|-----|-----------|----------|
| `container-navsim-transfuser` | NAVSIM Transfuser | Required | Required | ~140 min | ~10 min |
| `container-navsim-ego-mlp` | NAVSIM EgoStatusMLP baseline | Required | Not required | ~60 min | ~15 min |
| `container-pytorch-dlc` | PyTorch DLC managed container (generic template) | Not required | Supported | Not required | ~8 min |
| `container-pytorch-dlc-byoc` | PyTorch DLC-based BYOC | Required | Supported | Not required | ~20 min |

```bash
# Run with PyTorch managed container (default, no build required)
./pipelines/scripts/run-pipeline.sh

# Run with PyTorch managed container (no build required)
./pipelines/scripts/run-pipeline.sh -c container-pytorch-dlc

# Run with PyTorch DLC BYOC (Dockerfile build required)
./pipelines/scripts/run-pipeline.sh -c container-pytorch-dlc-byoc

# Run with NAVSIM EgoStatusMLP (data preparation → pipeline execution)
./pipelines/container-navsim-ego-mlp/scripts/prepare_dataset.sh
./pipelines/scripts/run-pipeline.sh -c container-navsim-ego-mlp --skip-upload

# Run with NAVSIM Transfuser (data preparation → pipeline execution)
./pipelines/container-navsim-transfuser/scripts/prepare_dataset.sh
./pipelines/scripts/run-pipeline.sh -c container-navsim-transfuser --skip-upload

# Re-run after changing only train.py / evaluate.py (skip build)
./pipelines/scripts/run-pipeline.sh --skip-upload --skip-build
```

To run each step individually, follow the steps below.

### Step 1: Prepare Source Code

Place source code on the Notebook instance to run pipeline scripts from the JupyterLab terminal.

- **With GitHub repository integration**: The repository is automatically cloned when the Notebook starts. After pushing changes from your local PC, run `git pull` in the Notebook terminal to fetch the latest changes.
- **Without GitHub repository integration**: The repository is automatically placed via S3 during deployment. If you modify files locally, re-run `deploy.sh` to update S3, then restart the Notebook (stop → start) to automatically download the latest files.

The entire repository is automatically placed at `~/SageMaker/{project-name}/` on the Notebook instance, regardless of GitHub integration.

```bash
cd ~/SageMaker/{project-name}
```

### Step 2: Upload Dataset

Data preparation differs by container type.

**For PyTorch containers**:

`container-pytorch-dlc` / `container-pytorch-dlc-byoc` upload sample data from the container directory to S3.

```bash
./pipelines/scripts/01-upload-dataset.sh [PROJECT_NAME]

# To specify a container
./pipelines/scripts/01-upload-dataset.sh -c container-pytorch-dlc [PROJECT_NAME]
```

Data files are in each container directory's `data/` subdirectory (e.g., `pipelines/container-pytorch-dlc/data/{train,test}.csv`). To use your own dataset, replace the CSV files and run the command above.

**For NAVSIM containers**:

`container-navsim-ego-mlp` / `container-navsim-transfuser` use a dedicated `prepare_dataset.sh` to prepare and upload data to S3. Use `--skip-upload` when running the pipeline. For details on dataset preparation (prerequisites, disk space, feature extraction), see the [NAVSIM Guide - Dataset Preparation](navsim-guide.md#dataset-preparation).

```bash
# 1. Prepare dataset and upload to S3
./pipelines/container-navsim-ego-mlp/scripts/prepare_dataset.sh

# 2. Run pipeline with --skip-upload (data already prepared)
./pipelines/scripts/run-pipeline.sh -c container-navsim-ego-mlp --skip-upload
```

**Automatic data-pipeline mapping**: Data is stored in S3 using the container name as a prefix. When a pipeline runs, the same container name is used to generate the S3 path, so data and containers are automatically linked.

```
s3://{project}-dataset-{account}-{region}/
  ├── container-navsim-ego-mlp/train/        ← training data for container-navsim-ego-mlp
  └── container-navsim-transfuser/train/     ← training data for container-navsim-transfuser
```

### Step 3: Container Build & ECR Push

Build Docker images for BYOC containers and push to ECR. `container-pytorch-dlc` uses AWS managed containers, so no build is required (it will be skipped). Run this when you change dependencies (`pip install`) or the base image. For logic changes in train.py / evaluate.py only, rebuilding is not necessary as the SDK injects scripts via S3 (see `docs/sagemaker-python-sdk-guide.ja.md` Section 3.3 for details).

```bash
./pipelines/scripts/02-build-and-push-container.sh -c container-pytorch-dlc-byoc [PROJECT_NAME]
```

| Container | Description | Build | ECR Tag |
|-----------|-------------|-------|---------|
| `container-pytorch-dlc` | PyTorch DLC-based, managed container | Not required | - |
| `container-pytorch-dlc-byoc` | PyTorch DLC-based BYOC (Train also BYOC) | Required (10 GB+) | `container-pytorch-dlc-byoc` |
| `container-navsim-ego-mlp` | NAVSIM EgoStatusMLP (CPU) | Required | `container-navsim-ego-mlp` |
| `container-navsim-transfuser` | NAVSIM Transfuser (GPU) | Required | `container-navsim-transfuser` |

Each container is pushed to a single ECR repository (`{project}-container`) with the container directory name as the tag. The directory name specified with `-c` becomes the ECR tag, allowing multiple containers to coexist.

> ⚠️ `container-pytorch-dlc-byoc` includes GPU libraries (CUDA, cuDNN, NCCL, etc.), so the initial build download takes time. Ensure sufficient free disk space (30 GB+ recommended).

### Step 4: Create & Run Pipeline

Create and run a SageMaker Pipeline. The pipeline consists of 3 steps — Train → RegisterModel → Evaluate — which automatically execute model training, registration, and evaluation in sequence.

To run from the terminal:

For `container-pytorch-dlc`:

```bash
ROLE_ARN=$(aws cloudformation describe-stacks \
  --stack-name sagemaker-ai-ml-pipeline-stack \
  --query 'Stacks[0].Outputs[?OutputKey==`SageMakerRoleArn`].OutputValue' \
  --output text)

python pipelines/scripts/03-create-and-run-pipeline.py \
  --project-name sagemaker-ai-ml-pipeline \
  --role-arn "$ROLE_ARN" \
  --container-dir pipelines/container-pytorch-dlc \
  --create --start
```

Key options:

| Option | Description |
|--------|-------------|
| `--project-name` | Project name (required) |
| `--role-arn` | SageMaker execution role ARN (required) |
| `--region` | AWS region (default: `us-east-1`) |
| `--container-dir` | Container directory path (default: `pipelines/container-navsim-transfuser`) |
| `--create` | Create/update pipeline |
| `--start` | Start pipeline execution |
| `--subnet-ids` | VPC subnet IDs (comma-separated). Auto-detected from CFn stack if omitted |
| `--security-group-ids` | Security group IDs (comma-separated). Auto-detected from CFn stack if omitted |

If neither `--create` nor `--start` is specified, the pipeline definition JSON is printed to stdout.

> 💡 With VPC configuration, omitting `--subnet-ids` / `--security-group-ids` auto-detects them from CloudFormation stack outputs. Explicitly specified values take precedence.

### Step 5: Check Execution Status

After pipeline execution, check the status of each step from the terminal:

```bash
./pipelines/scripts/04-check-pipeline-status.sh [PROJECT_NAME]
```

Example output:

```
=== Pipeline Execution Status ===
Pipeline:  sagemaker-ai-ml-pipeline-container-pytorch-dlc-pipeline
Execution: ooj49xv2k8fc
Status:    🔄 Executing
Started:   1771677200.017

Steps:
🔄 Evaluate: Executing
└─ [Console]  [CW Instance Metrics]
✅ RegisterModel-RegisterModel: Succeeded
✅ Train: Succeeded
└─ [Console]  [CW Instance Metrics]  [CW Algorithm Metrics]
```

`[Console]` / `[CW Instance Metrics]` / `[CW Algorithm Metrics]` are clickable links in the terminal.

### Running with Jupyter Notebooks

`notebooks/` contains Notebooks corresponding to each phase of the ML workflow.

| Notebook | Purpose |
|----------|---------|
| `pytorch-pipeline.ipynb` | PyTorch DLC: Data inspection → Local training/evaluation → SageMaker Job → Pipeline |
| `pytorch-byoc-pipeline.ipynb` | PyTorch BYOC: Data inspection → Local training/evaluation → Docker Build → SageMaker Job → Pipeline |
| `navsim-ego-mlp-pipeline.ipynb` | NAVSIM EgoStatusMLP training and evaluation |
| `navsim-transfuser-pipeline.ipynb` | NAVSIM Transfuser / LTF training and evaluation (GPU) |

Open and run each Notebook from JupyterLab. See individual Notebooks for details.

## CARLA Simulation Demo

Run the TransFuser model trained with the Pipeline on the [CARLA](https://carla.org/) autonomous driving simulator. This lets you observe how a model trained on NAVSIM's offline dataset behaves in a real-time simulation environment.

The demo runs the following flow end-to-end:

- Launch the CARLA server
- Download the trained TransFuser model from S3
- Capture sensor data in real time from 3 RGB cameras and LiDAR
- Convert the model's predicted 4-second future trajectory into steering, throttle, and brake via Pure Pursuit + Lane-Keeping control
- Record a driving video

A GPU instance (`ml.g4dn.2xlarge` or larger) is required. The default SageMaker AI Notebook instance (`ml.g4dn.2xlarge`) can run this as is.

For detailed instructions, architecture, and customization options, see the [CARLA Simulation Demo README](../demo-carla/transfuser/README.md).

## Inference Endpoint and Demo App

Deploy trained models as SageMaker real-time inference endpoints and send inference requests from a demo app.

### Step 1: Deploy Inference Endpoint

```bash
# For EgoStatusMLP (CPU)
./infra/sagemaker-ai-inference/scripts/deploy.sh -c navsim-ego-mlp

# For Transfuser (GPU)
./infra/sagemaker-ai-inference/scripts/deploy.sh -c navsim-transfuser
```

The deploy script automatically performs the following:

1. Searches for the latest model artifact on S3
2. Repackages `model.tar.gz` including the inference script (`inference.py`)
3. Creates the endpoint via CloudFormation

### Step 2: Launch Demo App

```bash
pip install -r demo-app/requirements.txt
streamlit run demo-app/main.py
```

You can specify the endpoint name and region via environment variables:

```bash
export AWS_DEFAULT_REGION=us-east-1
export SAGEMAKER_ENDPOINT=my-endpoint-name
streamlit run demo-app/main.py
```

For details, see [demo-app/README.md](../demo-app/README.md).

### Delete Inference Endpoint

```bash
./infra/sagemaker-ai-inference/scripts/destroy.sh -c navsim-ego-mlp
```

## SageMaker Unified Studio Integration (Optional)

Integration feature to view and manage SageMaker resources (Model Registry, Pipeline, etc.) created by this repository from [Amazon SageMaker Unified Studio](https://docs.aws.amazon.com/sagemaker-unified-studio/latest/userguide/what-is-sagemaker-unified-studio.html). Can be added at any time after deploying the ML pipeline environment. Skip this section if you do not use Unified Studio.

> ⚠️ Add Unified Studio settings to your `.env` file beforehand.
>
> ```bash
> # Add to .env
> UNIFIED_STUDIO_IDC_INSTANCE_ARN="arn:aws:sso:::instance/ssoins-xxxxxxxxxxxx"
> UNIFIED_STUDIO_SSO_USERS="user1@example.com,user2@example.com"
> ```
>
> Find `UNIFIED_STUDIO_IDC_INSTANCE_ARN` at AWS Console → IAM Identity Center → Settings → Instance ARN. `UNIFIED_STUDIO_SSO_USERS` is the email address of SSO users to register as domain administrators / project owners (comma-separated for multiple users).

### Step 1: Create Unified Studio Domain

Creates a DataZone V2 domain and IAM roles (DomainExecutionRole, DomainServiceRole). If `UNIFIED_STUDIO_IDC_INSTANCE_ARN` is set in `.env`, SSO authentication is automatically enabled.

```bash
./infra/unified-studio/scripts/deploy-foundation.sh
```

| Option | Default | Description |
|--------|---------|-------------|
| `--domain-name` | `sagemaker-ai-ml-pipeline` | Domain name |
| `--project-name` | `sagemaker-ai-ml-pipeline` | Resource naming prefix |
| `--region` | `us-east-1` | AWS region |

Upon completion, the domain ID and next step command are displayed.

### Step 2: Create Project

Performs blueprint activation (including VPC/Subnet configuration), IAM role creation (Provisioning / ManageAccess), Authorization policy setup, project profile creation (Tooling + LakehouseCatalog + MLExperiments), and project creation in one step. The script auto-detects blueprint IDs and default VPC, and auto-creates required roles if they do not exist. If `UNIFIED_STUDIO_SSO_USERS` is set in `.env`, SSO users are automatically added as project owners. SSO user profiles are created after the first login to Unified Studio, so member addition is skipped before login. Re-run the script after login.

```bash
./infra/unified-studio/scripts/deploy-project.sh \
  --domain-id <domain ID>
```

The domain ID is shown in Step 1 output.

| Option | Default | Description |
|--------|---------|-------------|
| `--domain-id` | (required) | Domain ID created in Step 1 |
| `--us-project-name` | `ml-pipeline` | Unified Studio project name |
| `--project-name` | `sagemaker-ai-ml-pipeline` | Resource naming prefix |

### Step 3: Link SageMaker Resources

Links existing SageMaker resources from the ML pipeline to the Unified Studio project.

```bash
./infra/unified-studio/scripts/setup-integration.sh \
  --domain-id <domain ID> \
  --project-id <project ID>
```

The setup script automatically performs:

- Model Registry sync (RAM share + DataZone DataSource)
- MLflow App connection (DataZone connection + `AmazonDataZoneProject` tag)
- `AmazonDataZoneProject` tag assignment to Pipeline / Training Job / Processing Job / ECR repository

The project ID is shown in Step 2 output. To check via CLI:

```bash
# List domains
aws datazone list-domains --region us-east-1

# List projects
aws datazone list-projects \
  --domain-identifier <domain ID> \
  --region us-east-1
```

### Unlinking

To unlink, delete in reverse order:

```bash
# Unlink Step 3: Remove SageMaker resource integration
./infra/unified-studio/scripts/setup-integration.sh \
  --unlink \
  --domain-id <domain ID> \
  --project-id <project ID>

# Delete Step 2: Project + profile
./infra/unified-studio/scripts/deploy-project.sh --delete --domain-id <domain ID>

# Delete Step 1: Domain + IAM roles
./infra/unified-studio/scripts/deploy-foundation.sh --delete
```

For integration details and resource descriptions, see the [SageMaker Unified Studio Integration Guide](unified-studio-integration-guide.ja.md). For setup constraints and troubleshooting, see the [SageMaker Unified Studio Setup Guide](unified-studio-setup-guide.ja.md).

## Testing

Two test scripts are provided in the `tests/` directory for verifying behavior after infrastructure deployment or changes to scripts / Notebooks.

- **`run-lint.sh`** — Statically validates script syntax and notebook structure. **No AWS resources are used and no charges are incurred.** Can be run on a local PC as well, making it suitable for pre-commit checks.
- **`run-tests.sh`** — Runs the Pipeline and Notebooks end-to-end to verify actual behavior. **Uses AWS resources and incurs charges.** Can only be run from the JupyterLab terminal on the SageMaker AI Notebook instance.

### Lint Check (run-lint.sh)

Validates script permissions, syntax, configuration consistency, and notebook structure. Does not use AWS resources or incur costs. Can be run locally or on the Notebook instance.

```bash
./tests/run-lint.sh
```

Tests performed:

| Check | Description |
|-------|-------------|
| Script permissions | All `.sh` and `.py` scripts have execute permission |
| Shell syntax | `bash -n` syntax validation |
| Python syntax | `py_compile` validation |
| `--help` option | All scripts respond to `--help` |
| `--show-config` | Instance type mapping is correct per container |
| `_common.sh` path | All scripts reference `infra/_common.sh` correctly |
| Notebook | JSON validity + `papermill --prepare-only` (kernel and parameter validation) |

### Integration Test (run-tests.sh)

Runs Pipeline and Notebook executions to verify end-to-end behavior. Uses AWS resources and incurs costs.

> ⚠️ This script must be run from the **JupyterLab terminal (on the SageMaker AI Notebook instance)**. It does not work on a local PC.

```bash
# Test a specific container (fastest: pytorch-dlc, no build required)
./tests/run-tests.sh -c container-pytorch-dlc

# Test PyTorch BYOC only
./tests/run-tests.sh -c container-pytorch-dlc-byoc

# Test NAVSIM EgoStatusMLP only
./tests/run-tests.sh -c container-navsim-ego-mlp

# Test NAVSIM Transfuser only (GPU, takes longer)
./tests/run-tests.sh -c container-navsim-transfuser

# All containers (takes longest)
./tests/run-tests.sh

# Skip Pipeline, run Notebook tests only
./tests/run-tests.sh --skip-pipeline

# Skip Notebook tests
./tests/run-tests.sh --skip-notebook -c container-pytorch-dlc
```

Tests performed:

| Check | Description |
|-------|-------------|
| Pipeline (`run-pipeline.sh`) | Dataset upload → container build → Pipeline execution complete without errors |
| Notebook (papermill) | Corresponding Pipeline notebook runs end-to-end (training, evaluation, model registration) |

| Option | Description |
|--------|-------------|
| `-c, --container DIR` | Container to test (can specify multiple times) |
| `--skip-pipeline` | Skip Pipeline execution |
| `--skip-notebook` | Skip Notebook execution |
| `--auto-approve` | Skip the confirmation prompt before running tests |

## Cleanup

Empties S3 buckets, deletes ECR images and SageMaker model package groups, then deletes the stack. `destroy.sh` handles everything automatically.

```bash
./infra/sagemaker-ai-ml-pipeline/scripts/destroy.sh [STACK_NAME] [PROJECT_NAME]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| STACK_NAME | `sagemaker-ai-ml-pipeline-stack` | CloudFormation stack name |
| PROJECT_NAME | `sagemaker-ai-ml-pipeline` | Resource naming prefix |

MLflow App deletion may take several minutes.

## Troubleshooting

For deployment errors (such as Service Quotas) and pipeline execution errors, see the [Troubleshooting Guide](troubleshooting-guide.md) for investigation and resolution steps.

## Appendix

### AWS Resources Created

The following resources are created by the CloudFormation stack:

| Resource | Name Pattern | Purpose |
|----------|-------------|---------|
| Amazon S3 Bucket | `{project}-dataset-{account}-{region}` | Training and test datasets (`train/`, `test/`) |
| Amazon S3 Bucket | `{project}-model-{account}-{region}` | Model artifacts |
| Amazon S3 Bucket | `{project}-eval-{account}-{region}` | Evaluation results |
| Amazon S3 Bucket | `{project}-mlflow-{account}-{region}` | MLflow artifacts |
| Amazon ECR Repository | `{project}-container` | Training/evaluation containers |
| SageMaker AI Notebook | `{project}-notebook` | Development notebook |
| SageMaker Lifecycle Config | `{project}-notebook-lcc` | Notebook startup configuration |
| SageMaker Code Repository | `{project}` | GitHub repository integration (only when GitHub URL is set) |
| SageMaker Model Package Group | `{project}-model-group` | Model registry |
| SageMaker MLflow App | `{project}-mlflow` | Experiment management |
| AWS Secrets Manager Secret | `{project}-sagemaker-github-credentials` | GitHub credentials (only when PAT is set) |
| IAM Role | `{project}-sagemaker-role` | SageMaker execution role |
| VPC | `{project}-vpc` | VPC (only when `ENABLE_VPC=true`) |
| Private Subnets ×2 | `{project}-private-1`, `{project}-private-2` | Notebook / Training / Processing Job placement |
| Public Subnets ×2 | `{project}-public-1`, `{project}-public-2` | NAT Gateway placement |
| NAT Gateway | `{project}-nat` | Internet access from private subnets |
| VPC Endpoints | - | S3 Gateway Endpoint + SageMaker API / MLflow / Notebook / ECR / CW Logs / STS (Interface) |

### Related Information

#### AWS Documentation

Official documentation for AWS services used in this project.

**SageMaker Pipelines**:

- [Amazon SageMaker Pipelines Overview](https://aws.amazon.com/sagemaker-ai/pipelines/)
- [Pipelines Overview - Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-overview.html)
- [Define a Pipeline](https://docs.aws.amazon.com/sagemaker/latest/dg/define-pipeline.html)
- [Create a Pipeline with @step Decorator](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-step-decorator-create-pipeline.html)

**SageMaker Model Registry**:

- [Model Registry Overview](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html)
- [Model Registry - Models, Versions, and Groups](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry-models.html)

**MLflow on SageMaker**:

- [Create an MLflow App](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-app-create-app-cli.html)
- [Launch the MLflow UI](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-launch-ui.html)
- [MLflow Tutorials - Example Notebooks](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-tutorials.html)
- [Auto-register Models with Model Registry via MLflow](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-track-experiments-model-registration.html)
- [MLflow Integration with SageMaker Pipelines](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps-integration.html)
- [Managed MLflow 3.0 on SageMaker (Blog)](https://aws.amazon.com/blogs/machine-learning/accelerating-generative-ai-development-with-fully-managed-mlflow-3-0-on-amazon-sagemaker-ai/)

**SageMaker MLOps**:

- [Amazon SageMaker MLOps](https://aws.amazon.com/sagemaker/ai/mlops/)

**CloudFormation Reference**:

- [AWS::SageMaker::Pipeline](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-sagemaker-pipeline.html)
- [AWS::SageMaker::NotebookInstance](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-sagemaker-notebookinstance.html)
- [AWS::SageMaker::ModelPackageGroup](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-sagemaker-modelpackagegroup.html)

**MLflow App API Reference**:

- [CreateMlflowApp](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateMlflowApp.html)
- [DescribeMlflowApp](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_DescribeMlflowApp.html)
- [DeleteMlflowApp](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_DeleteMlflowApp.html)
- [CreatePresignedMlflowAppUrl](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreatePresignedMlflowAppUrl.html)
- [Boto3 create_presigned_mlflow_app_url](https://docs.aws.amazon.com/boto3/latest/reference/services/sagemaker/client/create_presigned_mlflow_app_url.html)

#### Related Workshops and Samples

Workshops and sample repositories for learning more about ML workflows with SageMaker AI.

- [Amazon SageMaker AI Immersion Day](https://catalog.us-east-1.prod.workshops.aws/workshops/63069e26-921c-4ce1-9cc7-dd882ff62575/en-US) - Hands-on workshop covering SageMaker AI key features: data preparation, model training, deployment, and MLOps
- [amazon-sagemaker-from-idea-to-production](https://github.com/aws-samples/amazon-sagemaker-from-idea-to-production) - End-to-end sample building ML workflows from idea to production with SageMaker. Uses Studio, Pipelines, Model Registry, and Feature Store
- [sagemaker-end-to-end-workshop](https://github.com/aws-samples/sagemaker-end-to-end-workshop) - SageMaker end-to-end workshop. Hands-on learning of the complete ML lifecycle from data exploration to model deployment and monitoring
