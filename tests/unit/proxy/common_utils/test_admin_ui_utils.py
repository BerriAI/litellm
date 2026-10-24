import re

from litellm.proxy.common_utils.admin_ui_utils import missing_keys_form


def test_missing_keys_form_shows_generate_command_instead_of_a_literal_master_key():
    html = missing_keys_form(missing_key_names="DATABASE_URL, LITELLM_MASTER_KEY")

    assert "DATABASE_URL, LITELLM_MASTER_KEY" in html
    assert 'echo "LITELLM_MASTER_KEY=sk-$(openssl rand -hex 32)"' in html
    suggested_master_key_values = re.findall(r'LITELLM_MASTER_KEY="([^"]*)"', html)
    assert suggested_master_key_values == [""]
