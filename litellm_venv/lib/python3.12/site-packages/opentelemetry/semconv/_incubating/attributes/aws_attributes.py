# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from enum import Enum

AWS_DYNAMODB_ATTRIBUTE_DEFINITIONS = "aws.dynamodb.attribute_definitions"
"""
The JSON-serialized value of each item in the `AttributeDefinitions` request field.
"""

AWS_DYNAMODB_ATTRIBUTES_TO_GET = "aws.dynamodb.attributes_to_get"
"""
The value of the `AttributesToGet` request parameter.
"""

AWS_DYNAMODB_CONSISTENT_READ = "aws.dynamodb.consistent_read"
"""
The value of the `ConsistentRead` request parameter.
"""

AWS_DYNAMODB_CONSUMED_CAPACITY = "aws.dynamodb.consumed_capacity"
"""
The JSON-serialized value of each item in the `ConsumedCapacity` response field.
"""

AWS_DYNAMODB_COUNT = "aws.dynamodb.count"
"""
The value of the `Count` response parameter.
"""

AWS_DYNAMODB_EXCLUSIVE_START_TABLE = "aws.dynamodb.exclusive_start_table"
"""
The value of the `ExclusiveStartTableName` request parameter.
"""

AWS_DYNAMODB_GLOBAL_SECONDARY_INDEX_UPDATES = (
    "aws.dynamodb.global_secondary_index_updates"
)
"""
The JSON-serialized value of each item in the `GlobalSecondaryIndexUpdates` request field.
"""

AWS_DYNAMODB_GLOBAL_SECONDARY_INDEXES = "aws.dynamodb.global_secondary_indexes"
"""
The JSON-serialized value of each item of the `GlobalSecondaryIndexes` request field.
"""

AWS_DYNAMODB_INDEX_NAME = "aws.dynamodb.index_name"
"""
The value of the `IndexName` request parameter.
"""

AWS_DYNAMODB_ITEM_COLLECTION_METRICS = "aws.dynamodb.item_collection_metrics"
"""
The JSON-serialized value of the `ItemCollectionMetrics` response field.
"""

AWS_DYNAMODB_LIMIT = "aws.dynamodb.limit"
"""
The value of the `Limit` request parameter.
"""

AWS_DYNAMODB_LOCAL_SECONDARY_INDEXES = "aws.dynamodb.local_secondary_indexes"
"""
The JSON-serialized value of each item of the `LocalSecondaryIndexes` request field.
"""

AWS_DYNAMODB_PROJECTION = "aws.dynamodb.projection"
"""
The value of the `ProjectionExpression` request parameter.
"""

AWS_DYNAMODB_PROVISIONED_READ_CAPACITY = "aws.dynamodb.provisioned_read_capacity"
"""
The value of the `ProvisionedThroughput.ReadCapacityUnits` request parameter.
"""

AWS_DYNAMODB_PROVISIONED_WRITE_CAPACITY = "aws.dynamodb.provisioned_write_capacity"
"""
The value of the `ProvisionedThroughput.WriteCapacityUnits` request parameter.
"""

AWS_DYNAMODB_SCAN_FORWARD = "aws.dynamodb.scan_forward"
"""
The value of the `ScanIndexForward` request parameter.
"""

AWS_DYNAMODB_SCANNED_COUNT = "aws.dynamodb.scanned_count"
"""
The value of the `ScannedCount` response parameter.
"""

AWS_DYNAMODB_SEGMENT = "aws.dynamodb.segment"
"""
The value of the `Segment` request parameter.
"""

AWS_DYNAMODB_SELECT = "aws.dynamodb.select"
"""
The value of the `Select` request parameter.
"""

AWS_DYNAMODB_TABLE_COUNT = "aws.dynamodb.table_count"
"""
The number of items in the `TableNames` response parameter.
"""

AWS_DYNAMODB_TABLE_NAMES = "aws.dynamodb.table_names"
"""
The keys in the `RequestItems` object field.
"""

AWS_DYNAMODB_TOTAL_SEGMENTS = "aws.dynamodb.total_segments"
"""
The value of the `TotalSegments` request parameter.
"""

AWS_ECS_CLUSTER_ARN = "aws.ecs.cluster.arn"
"""
The ARN of an [ECS cluster](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/clusters.html).
"""

AWS_ECS_CONTAINER_ARN = "aws.ecs.container.arn"
"""
The Amazon Resource Name (ARN) of an [ECS container instance](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ECS_instances.html).
"""

AWS_ECS_LAUNCHTYPE = "aws.ecs.launchtype"
"""
The [launch type](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/launch_types.html) for an ECS task.
"""

AWS_ECS_TASK_ARN = "aws.ecs.task.arn"
"""
The ARN of a running [ECS task](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-account-settings.html#ecs-resource-ids).
"""

AWS_ECS_TASK_FAMILY = "aws.ecs.task.family"
"""
The family name of the [ECS task definition](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task_definitions.html) used to create the ECS task.
"""

AWS_ECS_TASK_ID = "aws.ecs.task.id"
"""
The ID of a running ECS task. The ID MUST be extracted from `task.arn`.
"""

AWS_ECS_TASK_REVISION = "aws.ecs.task.revision"
"""
The revision for the task definition used to create the ECS task.
"""

AWS_EKS_CLUSTER_ARN = "aws.eks.cluster.arn"
"""
The ARN of an EKS cluster.
"""

AWS_LAMBDA_INVOKED_ARN = "aws.lambda.invoked_arn"
"""
The full invoked ARN as provided on the `Context` passed to the function (`Lambda-Runtime-Invoked-Function-Arn` header on the `/runtime/invocation/next` applicable).
Note: This may be different from `cloud.resource_id` if an alias is involved.
"""

AWS_LOG_GROUP_ARNS = "aws.log.group.arns"
"""
The Amazon Resource Name(s) (ARN) of the AWS log group(s).
Note: See the [log group ARN format documentation](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/iam-access-control-overview-cwl.html#CWL_ARN_Format).
"""

AWS_LOG_GROUP_NAMES = "aws.log.group.names"
"""
The name(s) of the AWS log group(s) an application is writing to.
Note: Multiple log groups must be supported for cases like multi-container applications, where a single application has sidecar containers, and each write to their own log group.
"""

AWS_LOG_STREAM_ARNS = "aws.log.stream.arns"
"""
The ARN(s) of the AWS log stream(s).
Note: See the [log stream ARN format documentation](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/iam-access-control-overview-cwl.html#CWL_ARN_Format). One log group can contain several log streams, so these ARNs necessarily identify both a log group and a log stream.
"""

AWS_LOG_STREAM_NAMES = "aws.log.stream.names"
"""
The name(s) of the AWS log stream(s) an application is writing to.
"""

AWS_REQUEST_ID = "aws.request_id"
"""
The AWS request ID as returned in the response headers `x-amz-request-id` or `x-amz-requestid`.
"""

AWS_S3_BUCKET = "aws.s3.bucket"
"""
The S3 bucket name the request refers to. Corresponds to the `--bucket` parameter of the [S3 API](https://docs.aws.amazon.com/cli/latest/reference/s3api/index.html) operations.
Note: The `bucket` attribute is applicable to all S3 operations that reference a bucket, i.e. that require the bucket name as a mandatory parameter.
    This applies to almost all S3 operations except `list-buckets`.
"""

AWS_S3_COPY_SOURCE = "aws.s3.copy_source"
"""
The source object (in the form `bucket`/`key`) for the copy operation.
Note: The `copy_source` attribute applies to S3 copy operations and corresponds to the `--copy-source` parameter
    of the [copy-object operation within the S3 API](https://docs.aws.amazon.com/cli/latest/reference/s3api/copy-object.html).
    This applies in particular to the following operations:

    - [copy-object](https://docs.aws.amazon.com/cli/latest/reference/s3api/copy-object.html)
    - [upload-part-copy](https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part-copy.html).
"""

AWS_S3_DELETE = "aws.s3.delete"
"""
The delete request container that specifies the objects to be deleted.
Note: The `delete` attribute is only applicable to the [delete-object](https://docs.aws.amazon.com/cli/latest/reference/s3api/delete-object.html) operation.
    The `delete` attribute corresponds to the `--delete` parameter of the
    [delete-objects operation within the S3 API](https://docs.aws.amazon.com/cli/latest/reference/s3api/delete-objects.html).
"""

AWS_S3_KEY = "aws.s3.key"
"""
The S3 object key the request refers to. Corresponds to the `--key` parameter of the [S3 API](https://docs.aws.amazon.com/cli/latest/reference/s3api/index.html) operations.
Note: The `key` attribute is applicable to all object-related S3 operations, i.e. that require the object key as a mandatory parameter.
    This applies in particular to the following operations:

    - [copy-object](https://docs.aws.amazon.com/cli/latest/reference/s3api/copy-object.html)
    - [delete-object](https://docs.aws.amazon.com/cli/latest/reference/s3api/delete-object.html)
    - [get-object](https://docs.aws.amazon.com/cli/latest/reference/s3api/get-object.html)
    - [head-object](https://docs.aws.amazon.com/cli/latest/reference/s3api/head-object.html)
    - [put-object](https://docs.aws.amazon.com/cli/latest/reference/s3api/put-object.html)
    - [restore-object](https://docs.aws.amazon.com/cli/latest/reference/s3api/restore-object.html)
    - [select-object-content](https://docs.aws.amazon.com/cli/latest/reference/s3api/select-object-content.html)
    - [abort-multipart-upload](https://docs.aws.amazon.com/cli/latest/reference/s3api/abort-multipart-upload.html)
    - [complete-multipart-upload](https://docs.aws.amazon.com/cli/latest/reference/s3api/complete-multipart-upload.html)
    - [create-multipart-upload](https://docs.aws.amazon.com/cli/latest/reference/s3api/create-multipart-upload.html)
    - [list-parts](https://docs.aws.amazon.com/cli/latest/reference/s3api/list-parts.html)
    - [upload-part](https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part.html)
    - [upload-part-copy](https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part-copy.html).
"""

AWS_S3_PART_NUMBER = "aws.s3.part_number"
"""
The part number of the part being uploaded in a multipart-upload operation. This is a positive integer between 1 and 10,000.
Note: The `part_number` attribute is only applicable to the [upload-part](https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part.html)
    and [upload-part-copy](https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part-copy.html) operations.
    The `part_number` attribute corresponds to the `--part-number` parameter of the
    [upload-part operation within the S3 API](https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part.html).
"""

AWS_S3_UPLOAD_ID = "aws.s3.upload_id"
"""
Upload ID that identifies the multipart upload.
Note: The `upload_id` attribute applies to S3 multipart-upload operations and corresponds to the `--upload-id` parameter
    of the [S3 API](https://docs.aws.amazon.com/cli/latest/reference/s3api/index.html) multipart operations.
    This applies in particular to the following operations:

    - [abort-multipart-upload](https://docs.aws.amazon.com/cli/latest/reference/s3api/abort-multipart-upload.html)
    - [complete-multipart-upload](https://docs.aws.amazon.com/cli/latest/reference/s3api/complete-multipart-upload.html)
    - [list-parts](https://docs.aws.amazon.com/cli/latest/reference/s3api/list-parts.html)
    - [upload-part](https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part.html)
    - [upload-part-copy](https://docs.aws.amazon.com/cli/latest/reference/s3api/upload-part-copy.html).
"""


class AwsEcsLaunchtypeValues(Enum):
    EC2 = "ec2"
    """ec2."""
    FARGATE = "fargate"
    """fargate."""
