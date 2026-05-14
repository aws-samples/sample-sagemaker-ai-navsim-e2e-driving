# Troubleshooting Guide

🌐 **Language**: 🇺🇸 [English](troubleshooting-guide.md) | 🇯🇵 [日本語](troubleshooting-guide.ja.md)

This guide summarizes errors that can occur when deploying or running the SageMaker AI ML Pipeline, along with methods for investigating and resolving them.

- [Deployment Errors](#deployment-errors)
  - [ResourceLimitExceeded (Service Quotas)](#resourcelimitexceeded-service-quotas)
- [Pipeline Execution Errors](#pipeline-execution-errors)
  - [How to Investigate Errors](#how-to-investigate-errors)
    - [Step 1: Check the Pipeline execution status](#step-1-check-the-pipeline-execution-status)
    - [Step 2: Identify the failed Job](#step-2-identify-the-failed-job)
    - [Step 3: Check detailed error logs in CloudWatch Logs](#step-3-check-detailed-error-logs-in-cloudwatch-logs)
  - [AlgorithmError: ExecuteUserScriptError (exit code 1)](#algorithmerror-executeuserscripterror-exit-code-1)
  - [FileNotFoundError: No CSV files found](#filenotfounderror-no-csv-files-found)
- [Notebook Environment Errors](#notebook-environment-errors)
  - [Python Package Version Compatibility](#python-package-version-compatibility)
    - [NumPy 2.x and PyTorch Incompatibility](#numpy-2x-and-pytorch-incompatibility)
    - [MLflow 3.x and PyTorch 2.0 Incompatibility](#mlflow-3x-and-pytorch-20-incompatibility)
    - [Recommended pip install Commands](#recommended-pip-install-commands)
  - [Authentication Error on git pull](#authentication-error-on-git-pull)

## Deployment Errors

Errors that can occur during CloudFormation stack deployment (`deploy.sh`).

### ResourceLimitExceeded (Service Quotas)

SageMaker AI GPU instances may have a default quota of 0. If you encounter a `ResourceLimitExceeded` error, request a quota increase through AWS Service Quotas. The `deploy.sh` script automatically requests increases for common instance types, but approval may take from a few minutes to several hours.

The following quotas are commonly needed.

| Quota Name | Quota Code | Purpose | Recommended Value |
|-----------|-------------|------|--------|
| `ml.g6.xlarge for training job usage` | `L-1B43CB89` | Training Job (reserved) | 5 |
| `ml.g6.xlarge for processing job usage` | `L-B0D09498` | Processing Job (reserved) | 5 |
| `ml.g6.4xlarge for training job usage` | `L-07B51BA0` | Training Job (Transfuser) | 5 |
| `ml.g6.4xlarge for processing job usage` | `L-8BF5F502` | Processing Job (Transfuser) | 5 |

You can request quota increases via AWS CLI.

```bash
aws service-quotas request-service-quota-increase \
    --service-code sagemaker \
    --quota-code L-07B51BA0 \
    --desired-value 5 \
    --region us-east-1
```

Check the status in the [Service Quotas console](https://console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas).

CPU instances (`ml.c7i.xlarge`) used by EgoStatusMLP typically have sufficient default quotas and usually do not require increases.

## Pipeline Execution Errors

How to investigate failures in SageMaker Pipeline Training Jobs or Processing Jobs, along with common errors and their resolutions.

### How to Investigate Errors

The error message shown in the SageMaker console alone may not reveal the root cause. You can retrieve detailed error logs from CloudWatch Logs using the following steps.

#### Step 1: Check the Pipeline execution status

Check the Pipeline execution status via the SageMaker console or the AWS CLI.

```bash
aws sagemaker list-pipeline-executions \
  --pipeline-name sagemaker-ai-ml-pipeline-container-pytorch-dlc-pipeline \
  --sort-by CreationTime \
  --sort-order Descending \
  --max-results 5 \
  --region us-east-1
```

#### Step 2: Identify the failed Job

Use the Pipeline execution ARN to identify the Job name of the failed step.

```bash
aws sagemaker list-pipeline-execution-steps \
  --pipeline-execution-arn <execution-arn> \
  --region us-east-1
```

For a Training Job, the `FailureReason` field shows a summary of the error, but it may be empty like `ErrorMessage ""`. In that case, proceed to the next step and check CloudWatch Logs.

#### Step 3: Check detailed error logs in CloudWatch Logs

Logs for Training Jobs and Processing Jobs are output to CloudWatch Logs.

Here are the steps to check Training Job logs.

1. Retrieve the list of log streams.

```bash
aws logs describe-log-streams \
  --log-group-name "/aws/sagemaker/TrainingJobs" \
  --region us-east-1 \
  --order-by LastEventTime \
  --descending \
  --max-items 5 \
  --query 'logStreams[].logStreamName' \
  --output json
```

2. Specify the log stream name for the relevant Job and retrieve the log events. The log stream name follows the format `{job-name}/algo-1-{timestamp}`.

```bash
aws logs get-log-events \
  --log-group-name "/aws/sagemaker/TrainingJobs" \
  --log-stream-name "<log-stream-name>" \
  --region us-east-1 \
  --query 'events[].message' \
  --output json
```

For Processing Jobs, the log group name is different.

```bash
aws logs describe-log-streams \
  --log-group-name "/aws/sagemaker/ProcessingJobs" \
  --region us-east-1 \
  --order-by LastEventTime \
  --descending \
  --max-items 5 \
  --query 'logStreams[].logStreamName' \
  --output json
```

> 💡 You can also access CloudWatch Logs directly via the "View logs" link on the Training Job detail page in the SageMaker console.

### AlgorithmError: ExecuteUserScriptError (exit code 1)

You may see an error like the following in a Training Job or Processing Job.

```
AlgorithmError: ExecuteUserScriptError:
ExitCode 1
ErrorMessage ""
Command "/usr/local/bin/python3.10 train.py"
```

Because `ErrorMessage` is empty, this message alone does not reveal the cause. Follow the steps in [Step 3: Check detailed error logs in CloudWatch Logs](#step-3-check-detailed-error-logs-in-cloudwatch-logs) to check CloudWatch Logs.

Common causes include:

- Missing Python dependency packages (need to be added to the Dockerfile)
- Incorrect input data path (data has not been uploaded to S3)
- Insufficient IAM permissions (access to external services such as MLflow or S3)
- Bugs in the code (cases where it works locally but fails inside the container)

### FileNotFoundError: No CSV files found

```
FileNotFoundError: No CSV files found in /opt/ml/input/data/train
```

The data may not have been uploaded to S3. Upload the data with the following command.

```bash
./pipelines/scripts/01-upload-dataset.sh
```

After uploading, verify that the data exists in the S3 bucket.

```bash
aws s3 ls s3://sagemaker-ai-ml-pipeline-dataset-<account-id>/train/
aws s3 ls s3://sagemaker-ai-ml-pipeline-dataset-<account-id>/test/
```

## Notebook Environment Errors

Errors related to environment setup when working on the SageMaker AI Notebook instance.

### Python Package Version Compatibility

In the pytorch environment of the SageMaker AI Notebook Instance, you need to be mindful of version compatibility between preinstalled packages. Running `pip install -U` to update some packages may break compatibility with other packages due to changes in dependencies.

#### NumPy 2.x and PyTorch Incompatibility

When you run commands such as `pip install -U scikit-learn`, NumPy may be upgraded to 2.x as a dependency. However, since the preinstalled PyTorch (2.0.1) was compiled against NumPy 1.x, the following error occurs.

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6 as it may crash.
RuntimeError: Could not infer dtype of numpy.float32
```

Pin NumPy to 1.x by specifying `"numpy<2"` during pip install.

```bash
pip install -U scikit-learn pandas "numpy<2"
```

#### MLflow 3.x and PyTorch 2.0 Incompatibility

From MLflow 3.9 onward, `mlflow.pytorch.save_model()` internally imports the `torch.export` module (added in PyTorch 2.1). In PyTorch 2.0.x environments, the following error occurs.

```
ModuleNotFoundError: No module named 'torch.export'
```

MLflow App runs on MLflow 3.x. Install with the `sagemaker-mlflow` package.

```bash
pip install -U mlflow sagemaker-mlflow
```

#### Recommended pip install Commands

The pip install cell in each notebook uses the following version constraints.

For PyTorch notebooks:

```bash
pip install -U scikit-learn pandas mlflow sagemaker-mlflow matplotlib seaborn "numpy<2"
```

> ⚠️ Do not include PyTorch itself in the upgrade targets. Use the preinstalled version. Running `pip install -U torch` will pull in NumPy 2.x and cause the compatibility issues described above.

### Authentication Error on git pull

When running `git pull` from the terminal on the Notebook instance, you may encounter an error like the following.

```
Username for 'https://github.com/...':
Password for 'https://...@github.com/...':
remote: Invalid username or token. Password authentication is not supported for Git operations.
fatal: Authentication failed for 'https://github.com/...'
```

**Cause**: SageMaker CodeRepository clones the repository at Notebook startup using the GitHub Personal Access Token (PAT) stored in AWS Secrets Manager, but subsequent `git pull` / `git push` from the terminal may not automatically use the same credentials.

**Resolution**: Retrieve the GitHub PAT from Secrets Manager and set it in the repository's remote URL. The PAT is stored in Secrets Manager (`{project-name}-sagemaker-github-credentials`) from the `GITHUB_PAT` in `.env` during infrastructure deployment.

```bash
# Retrieve the PAT from Secrets Manager
PAT=$(aws secretsmanager get-secret-value \
  --secret-id sagemaker-ai-ml-pipeline-sagemaker-github-credentials \
  --query 'SecretString' --output text \
  | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['password'])")

# Set the PAT in the remote URL
REPO_URL=$(git remote get-url origin | sed 's|https://[^@]*@|https://|' | sed 's|https://|https://'"${PAT}"'@|')
git remote set-url origin "${REPO_URL}"

# Verify
git pull --all
```

> **Note**: This configuration is saved in `.git/config`, so it persists even if the Notebook instance is stopped and restarted. However, if you reissue the PAT, you need to configure it again.
