# Copyright 2015 Google LLC
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

"""Define class for the custom encryption configuration."""

import copy


class EncryptionConfiguration(object):
    """Custom encryption configuration (e.g., Cloud KMS keys).

    Args:
        kms_key_name (str): resource ID of Cloud KMS key used for encryption
    """

    def __init__(self, kms_key_name=None) -> None:
        self._properties = {}
        if kms_key_name is not None:
            self._properties["kmsKeyName"] = kms_key_name

    @property
    def kms_key_name(self):
        """str: Resource ID of Cloud KMS key

        Resource ID of Cloud KMS key or :data:`None` if using default
        encryption.
        """
        return self._properties.get("kmsKeyName")

    @kms_key_name.setter
    def kms_key_name(self, value):
        self._properties["kmsKeyName"] = value

    @classmethod
    def from_api_repr(cls, resource):
        """Construct an encryption configuration from its API representation

        Args:
            resource (Dict[str, object]):
                An encryption configuration representation as returned from
                the API.

        Returns:
            google.cloud.bigquery.table.EncryptionConfiguration:
                An encryption configuration parsed from ``resource``.
        """
        config = cls()
        config._properties = copy.deepcopy(resource)
        return config

    def to_api_repr(self):
        """Construct the API resource representation of this encryption
        configuration.

        Returns:
            Dict[str, object]:
                Encryption configuration as represented as an API resource
        """
        return copy.deepcopy(self._properties)

    def __eq__(self, other):
        if not isinstance(other, EncryptionConfiguration):
            return NotImplemented
        return self.kms_key_name == other.kms_key_name

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.kms_key_name)

    def __repr__(self):
        return "EncryptionConfiguration({})".format(self.kms_key_name)
