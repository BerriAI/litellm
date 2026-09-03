from litellm.proxy.auth.litellm_license import AUTO_ROUTER_LICENSE_FEATURE, LicenseCheck


def test_read_public_key_loads_successfully():
    """Ensure public_key.pem is valid PEM with no leading whitespace."""
    license_check = LicenseCheck()
    assert license_check.public_key is not None, (
        "public_key.pem could not be loaded — check for leading whitespace or malformed PEM header"
    )


def test_is_over_limit():
    license_check = LicenseCheck()
    license_check.airgapped_license_data = {"max_users": 100}
    assert license_check.is_over_limit(101) is True
    assert license_check.is_over_limit(100) is False
    assert license_check.is_over_limit(99) is False

    license_check.airgapped_license_data = {}
    assert license_check.is_over_limit(101) is False
    assert license_check.is_over_limit(100) is False
    assert license_check.is_over_limit(99) is False

    license_check.airgapped_license_data = None
    assert license_check.is_over_limit(101) is False
    assert license_check.is_over_limit(100) is False
    assert license_check.is_over_limit(99) is False


def test_allows_feature_requires_signed_license_claim():
    license_check = LicenseCheck()

    license_check.airgapped_license_data = {
        "allowed_features": [AUTO_ROUTER_LICENSE_FEATURE],
        "expiration_date": "2999-01-01",
    }
    assert license_check.allows_feature(AUTO_ROUTER_LICENSE_FEATURE) is True

    license_check.airgapped_license_data = {"allowed_features": ["other_feature"], "expiration_date": "2999-01-01"}
    assert license_check.allows_feature(AUTO_ROUTER_LICENSE_FEATURE) is False

    license_check.airgapped_license_data = {
        "allowed_features": [AUTO_ROUTER_LICENSE_FEATURE],
        "expiration_date": "2000-01-01",
    }
    assert license_check.allows_feature(AUTO_ROUTER_LICENSE_FEATURE) is False
    assert license_check.airgapped_license_data is None

    license_check.airgapped_license_data = None
    assert license_check.allows_feature(AUTO_ROUTER_LICENSE_FEATURE) is False
