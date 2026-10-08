# RDS IAM authentication across regions

Set `IAM_TOKEN_DB_AUTH=true` to use AWS IAM database tokens. LiteLLM selects a signing region independently for the writer and read replica:

| Connection | Optional override |
|---|---|
| Writer | `AWS_RDS_REGION` |
| Read replica | `AWS_RDS_READ_REPLICA_REGION` |

Each connection uses its override first, then the region in its RDS hostname, then the existing AWS RDS client region (`AWS_REGION_NAME`, falling back to `AWS_REGION`). Surrounding whitespace is stripped; empty values are treated as unset. The reader never inherits `AWS_RDS_REGION`.

For example:

```bash
IAM_TOKEN_DB_AUTH=true
DATABASE_HOST=writer.abc123.us-east-1.rds.amazonaws.com
DATABASE_PORT=5432
DATABASE_USER=iam_user
DATABASE_NAME=litellm
DATABASE_URL_READ_REPLICA=postgresql://iam_user@reader.abc123.ap-northeast-1.rds.amazonaws.com:5432/litellm?sslmode=require
AWS_RDS_REGION=us-east-1
AWS_RDS_READ_REPLICA_REGION=ap-northeast-1
```

The overrides are optional for these standard RDS endpoints. They apply to both initial authentication and renewed tokens, including readers configured through `DATABASE_HOST_READ_REPLICA` and writer connections through the in-container PgBouncer. Restart the proxy after changing them. They do not change Bedrock or other AWS service regions, password authentication, or Azure Entra authentication.

An explicit override wins even when it disagrees with the hostname. A wrong region can cause database authentication to fail; it does not fall back to a different signing region after an authentication rejection.

## Custom DNS

An opaque hostname cannot supply a region automatically. Region overrides only select the signing region; they do not rewrite the signing hostname or establish custom-domain support. AWS requires the actual RDS endpoint when generating an IAM authentication token, rather than a custom Route 53 record. TLS hostname verification is also independent of region selection. Use the canonical RDS endpoint in the connection configuration.

See [AWS IAM database authentication](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/UsingWithRDS.IAMDBAuth.Connecting.html).

## Verify a deployment

Use a test writer and cross-region replica with an IAM-enabled database user. Check both initial connections and fresh connections after token renewal. Verify replica queries return `pg_is_in_recovery() = true` and writer queries return `false`; a successful proxy response alone could conceal fallback to the writer. Check writes on the primary and allow for replication lag before checking them on the reader.

Test with both overrides unset, both set correctly, just one set, and an intentionally wrong region for each connection separately. The wrong-region cases should fail authentication for the affected database. Restore the correct configuration afterwards.
