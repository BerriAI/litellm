def generate_iam_auth_token(db_host, db_port, db_user) -> str:
    from urllib.parse import quote

    import boto3

    client = boto3.client("rds")
    token = client.generate_db_auth_token(
        DBHostname=db_host, Port=db_port, DBUsername=db_user
    )
    cleaned_token = quote(token, safe="")
    return cleaned_token
