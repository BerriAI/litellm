pub fn role_identity(arn: &str) -> Option<(&str, &str, &str)> {
    let mut parts = arn.splitn(6, ':');
    let ("arn", partition, _, _, account, resource) = (
        parts.next()?,
        parts.next()?,
        parts.next()?,
        parts.next()?,
        parts.next()?,
        parts.next()?,
    ) else {
        return None;
    };
    let role = if let Some(role) = resource.strip_prefix("role/") {
        role.rsplit('/').next()?
    } else {
        resource.strip_prefix("assumed-role/")?.split('/').next()?
    };
    Some((partition, account, role))
}

pub fn same_role_arns(target: &str, caller: &str) -> bool {
    match (role_identity(target), role_identity(caller)) {
        (Some(target), Some(caller)) => target == caller,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::same_role_arns;

    #[test]
    fn role_matching_is_partition_account_and_role_aware() {
        assert!(same_role_arns(
            "arn:aws:iam::123456789012:role/path/demo",
            "arn:aws:sts::123456789012:assumed-role/demo/session"
        ));
        assert!(!same_role_arns(
            "arn:aws:iam::123456789012:role/demo",
            "arn:aws-cn:iam::123456789012:role/demo"
        ));
    }
}
