# SageMaker Unified Studio Setup Guide

🌐 **Language**: 🇺🇸 [English](unified-studio-setup-guide.md) | 🇯🇵 [日本語](unified-studio-setup-guide.ja.md)

This guide covers creating SageMaker Unified Studio domains and projects using CloudFormation and scripts.

## Table of Contents

- [Deployment Flow](#deployment-flow)
- [Resource Management](#resource-management)
  - [Resources That Can Be Managed with CloudFormation](#resources-that-can-be-managed-with-cloudformation)
  - [Resources That Cannot Be Managed with CloudFormation](#resources-that-cannot-be-managed-with-cloudformation)
- [Design Considerations](#design-considerations)
  - [IAM Role Design](#iam-role-design)
  - [Notes on Blueprint Configuration](#notes-on-blueprint-configuration)
  - [Authentication Methods (IDC-Based vs IAM-Based)](#authentication-methods-idc-based-vs-iam-based)
- [Troubleshooting](#troubleshooting)

## Deployment Flow

The Unified Studio environment consists of the following three scripts. No manual operations on the management console are required.

| Step | Method | Content |
|---------|------|------|
| Step 1: Foundation | CFn (`foundation.yaml`) | Domain + DomainExecutionRole + DomainServiceRole |
| Step 2: Project | CFn (`project.yaml`) + API/CLI | IAM role creation + blueprint configuration + Authorization policy + ProjectProfile + Project + member addition |
| Step 3: Integration | CFn (`integration.yaml`) + API/CLI | Model Registry integration (RAM share + DataSource) + MLflow connection + tagging |

Deploy in the following order.

```
deploy-foundation.sh
  └─ CFn: Domain + IAM Roles (DomainExecutionRole, DomainServiceRole)
       │
       ▼
deploy-project.sh
  ├─ API: Create IAM roles (Provisioning / ManageAccess, only if they do not exist)
  ├─ API: put-environment-blueprint-configuration (with regionalParameters)
  ├─ API: add-policy-grant (Authorization policies)
  ├─ CFn: ProjectProfile + Project (Tooling ON_CREATE)
  └─ API: create-project-membership (add IAM / SSO users as owners)
       │
       ▼
setup-integration.sh
  ├─ CFn: RAM share + DataZone DataSource (Model Registry integration)
  ├─ API: create-connection (MLflow App connection)
  └─ API: Apply AmazonDataZoneProject tags (Pipeline, Training Job, MLflow App, etc.)
```

Deletion is performed in reverse order. `deploy-foundation.sh --delete` automatically handles deletion of all projects under the domain, deletion of the ManageAccess role, and cleanup of the Lake Formation Data Lake Admin.

## Resource Management

### Resources That Can Be Managed with CloudFormation

The following resources can be created and managed with CloudFormation.

| Resource Type | CFn Resource | Notes |
|--------------|-------------|------|
| DataZone Domain (V2) | `AWS::DataZone::Domain` | `DomainVersion: V2` and `ServiceRole` are required |
| Domain Execution Role | `AWS::IAM::Role` | Attach `SageMakerStudioAdminIAMDefaultExecutionPolicy` |
| Domain Service Role | `AWS::IAM::Role` | Attach `SageMakerStudioDomainServiceRolePolicy`, Path is `/service-role/` |
| Project Profile | `AWS::DataZone::ProjectProfile` | Specify blueprints in `EnvironmentConfigurations` |
| Project | `AWS::DataZone::Project` | Blueprint profile can be specified with `ProjectProfileId` |

Reference:

- [AWS::DataZone::Domain](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-datazone-domain.html)
- [AWS::DataZone::ProjectProfile](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-datazone-projectprofile.html)
- [AWS::DataZone::Project](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-datazone-project.html)

### Resources That Cannot Be Managed with CloudFormation

The following resources cannot be managed via CFn and require API/CLI operations. `deploy-project.sh` configures these automatically.

#### EnvironmentBlueprintConfiguration (Managed Blueprints for V2 Domains)

While `AWS::DataZone::EnvironmentBlueprintConfiguration` exists as a CFn resource, specifying a managed blueprint (Tooling, MLExperiments, etc.) for a V2 domain results in the following error.

```
Managed Environment Blueprint with <blueprint-id> doesn't exist.
```

The CFn documentation also states: "In the current release, only the following values are supported: DefaultDataLake and DefaultDataWarehouse."

As an alternative, use the `put-environment-blueprint-configuration` API. The `regionalParameters` require the following four parameters. `deploy-project.sh` automatically detects and configures these from the default VPC.

| Parameter | Description | Example |
|-----------|------|-----|
| `VpcId` | VPC ID | `vpc-xxxxxxxxxxxxxxxxx` |
| `Subnets` | Subnet IDs (comma-separated) | `subnet-aaa,subnet-bbb` |
| `AZs` | Availability Zones (comma-separated) | `<az-1>,<az-2>` |
| `S3Location` | S3 bucket URI | `s3://amazon-sagemaker-{account}-{region}-{hash}` |

If configured without `regionalParameters`, the `get-environment-blueprint-configuration` API will return `enabledRegions` correctly, but during Tooling (ON_CREATE) provisioning, the error `Environment blueprint configuration needs to enable atleast one region` occurs.

Use `list-environment-blueprint-configurations` to verify the configuration.

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

Reference:

- [PutEnvironmentBlueprintConfiguration](https://docs.aws.amazon.com/datazone/latest/APIReference/API_PutEnvironmentBlueprintConfiguration.html)
- [ListEnvironmentBlueprintConfigurations](https://docs.aws.amazon.com/datazone/latest/APIReference/API_ListEnvironmentBlueprintConfigurations.html)
- [AWS::DataZone::EnvironmentBlueprintConfiguration](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-datazone-environmentblueprintconfiguration.html)

#### Blueprint Authorization Policies

Authorization policies that grant blueprint usage to specific domain units can be configured with the `add-policy-grant` API. The format of `entityIdentifier` for `ENVIRONMENT_BLUEPRINT_CONFIGURATION` is `{AWSAccountID}:{environmentBlueprintId}` (e.g., `123456789012:abcdef1234ghij`). While this format is not explicitly documented in the API reference, it was confirmed from the CloudTrail event (`AddPolicyGrant`) generated when Configure is executed via the management console.

`deploy-project.sh` automatically configures the following two policies for each blueprint.

- `CREATE_ENVIRONMENT_PROFILE` - Grants CONTRIBUTOR projects in the root domain unit permission to create environment profiles
- `CREATE_ENVIRONMENT_FROM_BLUEPRINT` - Grants CONTRIBUTOR projects in the root domain unit permission to create environments from blueprints

Example of configuring equivalent policies via the API:

```bash
# Get the root domain unit ID
ROOT_DOMAIN_UNIT_ID=$(aws datazone get-domain \
  --identifier <domain-id> \
  --region <region> \
  --query 'rootDomainUnitId' --output text)

# Add the CREATE_ENVIRONMENT_FROM_BLUEPRINT policy
aws datazone add-policy-grant \
  --domain-identifier <domain-id> \
  --entity-type ENVIRONMENT_BLUEPRINT_CONFIGURATION \
  --entity-identifier "<account-id>:<blueprint-id>" \
  --policy-type CREATE_ENVIRONMENT_FROM_BLUEPRINT \
  --principal "{\"project\":{\"projectDesignation\":\"CONTRIBUTOR\",\"projectGrantFilter\":{\"domainUnitFilter\":{\"domainUnit\":\"${ROOT_DOMAIN_UNIT_ID}\",\"includeChildDomainUnits\":true}}}}" \
  --detail '{"createEnvironmentFromBlueprint":{}}' \
  --region <region>

# Add the CREATE_ENVIRONMENT_PROFILE policy
aws datazone add-policy-grant \
  --domain-identifier <domain-id> \
  --entity-type ENVIRONMENT_BLUEPRINT_CONFIGURATION \
  --entity-identifier "<account-id>:<blueprint-id>" \
  --policy-type CREATE_ENVIRONMENT_PROFILE \
  --principal "{\"project\":{\"projectDesignation\":\"CONTRIBUTOR\",\"projectGrantFilter\":{\"domainUnitFilter\":{\"domainUnit\":\"${ROOT_DOMAIN_UNIT_ID}\",\"includeChildDomainUnits\":false}}}}" \
  --detail "{\"createEnvironmentProfile\":{\"domainUnitId\":\"${ROOT_DOMAIN_UNIT_ID}\"}}" \
  --region <region>
```

Existing policies can be verified with `list-policy-grants`.

```bash
aws datazone list-policy-grants \
  --domain-identifier <domain-id> \
  --entity-type ENVIRONMENT_BLUEPRINT_CONFIGURATION \
  --entity-identifier "<account-id>:<blueprint-id>" \
  --policy-type CREATE_ENVIRONMENT_FROM_BLUEPRINT \
  --region <region>
```

Reference:

- [AddPolicyGrant](https://docs.aws.amazon.com/datazone/latest/APIReference/API_AddPolicyGrant.html)
- [ListPolicyGrants](https://docs.aws.amazon.com/datazone/latest/APIReference/API_ListPolicyGrants.html)
- [Assign authorization policies within blueprint configurations](https://docs.aws.amazon.com/datazone/latest/userguide/assign-authorization-policies-in-blueprint-config.html)

#### Project Membership

When `AWS::DataZone::Project` is created via CFn, the creator is the CloudFormation service, so users are not automatically added as project members. Accessing the Unified Studio UI will display "No project access".

`deploy-project.sh` attempts to add members in the following order.

1. Search for the DataZone profile of the current IAM principal, and add as project owner if found
2. Search for the DataZone profile of SSO users specified in `UNIFIED_STUDIO_SSO_USERS` in `.env`, and add as project owner if found

A user's DataZone profile is created when that user first logs in to the Unified Studio portal. If the user has not yet logged in at the time `deploy-project.sh` is executed, the profile will not be found, and member addition will be skipped. In this case, re-running `deploy-project.sh` after login will skip existing resources and execute only the member addition.

To add members manually via the API, use the following commands.

```bash
# Get the SSO user's profile ID
aws datazone search-user-profiles \
  --domain-identifier <domain-id> \
  --user-type DATAZONE_SSO_USER \
  --region <region>

# Add as project owner
aws datazone create-project-membership \
  --domain-identifier <domain-id> \
  --project-identifier <project-id> \
  --member '{"userIdentifier":"<user-profile-id>"}' \
  --designation PROJECT_OWNER \
  --region <region>
```

Reference:

- [CreateProjectMembership](https://docs.aws.amazon.com/datazone/latest/APIReference/API_CreateProjectMembership.html)
- [SearchUserProfiles](https://docs.aws.amazon.com/datazone/latest/APIReference/API_SearchUserProfiles.html)

## Design Considerations

### IAM Role Design

Unified Studio uses multiple IAM roles. It is important to understand the management methods and naming conventions for these roles.

#### Role List

| Role Name | Creation Method | Scope | Purpose |
|---------|---------|---------|------|
| `{project}-unified-studio-domain-role` | CFn (foundation) | Domain-specific | Domain execution role |
| `{project}-unified-studio-service-role` | CFn (foundation) | Domain-specific | Domain service role (required for V2) |
| `AmazonSageMakerProvisioning-{AccountId}` | `deploy-project.sh` (created only if it does not exist) | Account-shared | Blueprint provisioning |
| `AmazonSageMakerManageAccess-{Region}-{DomainId}` | `deploy-project.sh` (created only if it does not exist) | Domain-specific | Blueprint access management |

#### Provisioning Role Shared Design

`AmazonSageMakerProvisioning-{AccountId}` is designed to exist only once per account.

- Since the role name does not include a domain ID, it is shared across all V2 domains in the same account
- The trust policy Condition is only `aws:SourceAccount` (no domain-specific restrictions)
- If the role does not exist, `deploy-project.sh` creates it automatically. It is also created by Configure via the management console
- Should not be managed in a CFn stack (because it affects other domains when the stack is deleted)

Reference: [AmazonSageMakerProvisioning-\<domainAccountId\> role](https://docs.aws.amazon.com/sagemaker-unified-studio/latest/adminguide/AmazonSageMakerProvisioning.html)

#### ManageAccess Role Domain-Specific Design

`AmazonSageMakerManageAccess-{Region}-{DomainId}` is created per domain.

- Since the role name includes the domain ID, there are no collisions between domains
- The trust policy Condition specifies the domain ARN with `ArnEquals: aws:SourceArn`
- If the role does not exist, `deploy-project.sh` creates it automatically. `deploy-foundation.sh --delete` deletes it automatically
- On deletion, it is also automatically cleaned up from Lake Formation's Data Lake Admin (because leaving deleted roles can cause provisioning of new domains to fail)

#### Custom Name Role Constraints

The official documentation states that custom-named Provisioning roles are supported, but in practice, performing ON_CREATE provisioning of the Tooling blueprint with a custom-named role produced the following error.

```
Caller is not authorized to create environment using blueprintId <blueprint-id>
```

This error also occurs when the Authorization policy is not configured. It has not been possible to determine whether this is an issue with the custom-named role itself or with the Authorization policy. The current `deploy-project.sh` configures both the standard-named role and the Authorization policy, so this issue does not occur.

### Notes on Blueprint Configuration

#### Retrieving Blueprint IDs

The IDs of managed blueprints can be retrieved with the `list-environment-blueprints --managed` API. `deploy-project.sh` dynamically retrieves blueprint IDs via this API, so hard-coding is not required.

```bash
aws datazone list-environment-blueprints \
  --domain-identifier <domain-id> \
  --managed \
  --region <region> \
  --query 'items[].{id:id,name:name}' --output table
```

### Authentication Methods (IDC-Based vs IAM-Based)

Unified Studio domains require choosing either "Identity Center (IDC)-based" or "IAM-based" authentication at creation time. A single domain cannot use both simultaneously.

| Item | IDC-Based | IAM-Based |
|------|-----------|-----------|
| Authentication method | AWS IAM Identity Center (SSO) | IAM role |
| User management | Retains individual user IDs | Shares the same role within the project |
| Governance | Fine-grained access control, catalog management | Prioritizes developer productivity |
| Limitations | None | Only one per account per region |
| Portal access | SSO login | Login with IAM role |

It is possible to create and use both IDC-based and IAM-based domains separately in the same account and region.

Reference:

- [Domains in Amazon SageMaker Unified Studio](https://docs.aws.amazon.com/sagemaker-unified-studio/latest/adminguide/working-with-domains.html)
- [Using Amazon SageMaker Unified Studio Identity center (IDC) and IAM-based domains together](https://aws.amazon.com/blogs/big-data/using-amazon-sagemaker-unified-studio-identity-center-idc-and-iam-based-domains-together/)

#### Notes on Portal Access

Accessing an IDC-based domain portal URL with an IAM role will display the error "IAM roles are not permitted to access the portal". Although the portal login screen displays two options, "Sign in with SSO" and "Sign in with AWS IAM", IDC-based domains cannot use "Sign in with AWS IAM".

This project uses an IDC-based domain. Please access it with an SSO user.

#### About the AI/ML Section in the Left Menu

The AI/ML section in the left menu (MLflow, Models, Training jobs, Inference endpoints) is displayed only in IAM-based domains. In IDC-based domains, these features are accessed from the Build menu.

The official documentation explicitly states: "For SageMaker Unified Studio domains configured with IAM roles, you will be able to access the following components".

Reference: [Navigating Amazon SageMaker Unified Studio](https://docs.aws.amazon.com/sagemaker-unified-studio/latest/userguide/navigating-sagemaker-unified-studio.html)

## Troubleshooting

### ServiceRole Required for V2 Domains

When specifying `DomainVersion: V2` in `AWS::DataZone::Domain`, the `ServiceRole` property is required. Although the CFn documentation states Required: No, the following error occurs in practice.

```
ServiceRole is required for creating a V2 domain.
```

Reference: [AWS::DataZone::Domain - ServiceRole](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-datazone-domain.html#cfn-datazone-domain-servicerole)

### ProjectProfile ON_CREATE and Authorization

Specifying `DeploymentMode: ON_CREATE` in `EnvironmentConfigurations` of `ProjectProfile` causes the blueprint environment to be automatically provisioned when the project is created. At this time, if the blueprint's Authorization policy and `regionalParameters` are not configured, provisioning will fail.

With `DeploymentMode: ON_DEMAND`, provisioning is not performed at project creation time, and must be added manually from the Compute page. However, setting Tooling to ON_DEMAND will not enable Spaces (JupyterLab / VS Code), and "Spaces not enabled in this project profile" will be displayed.

### Lake Formation Invalid Principal

Even if the ManageAccess role is deleted from IAM when deleting a domain, the ARN of that role remains in Lake Formation's Data Lake Admin. Creating a new domain in this state will cause the following error during Tooling provisioning.

```
Failed to add arn:aws:iam::<account>:role/service-role/AmazonSageMakerManageAccess-... as data lake administrator: invalid principal detected.
```

`deploy-foundation.sh --delete` performs automatic cleanup from Lake Formation, but if the role was deleted manually, verify and correct it with the following commands.

```bash
# Check current Data Lake Admins
aws lakeformation get-data-lake-settings --region <region> \
  --query 'DataLakeSettings.DataLakeAdmins'

# Reconfigure, excluding invalid principals
aws lakeformation put-data-lake-settings --region <region> \
  --data-lake-settings '{"DataLakeAdmins":[{"DataLakePrincipalIdentifier":"<valid role ARN>"}]}'
```

### "Role already exists" Error During Configure

When executing Configure via the management console, if `AmazonSageMakerProvisioning-{AccountId}` already exists, an error will be displayed. However, Configure itself succeeds, and the project profile is created normally. The existing role is used as-is.
