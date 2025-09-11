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


DEVICE_ID = "device.id"
"""
A unique identifier representing the device.
Note: The device identifier MUST only be defined using the values outlined below. This value is not an advertising identifier and MUST NOT be used as such. On iOS (Swift or Objective-C), this value MUST be equal to the [vendor identifier](https://developer.apple.com/documentation/uikit/uidevice/1620059-identifierforvendor). On Android (Java or Kotlin), this value MUST be equal to the Firebase Installation ID or a globally unique UUID which is persisted across sessions in your application. More information can be found [here](https://developer.android.com/training/articles/user-data-ids) on best practices and exact implementation details. Caution should be taken when storing personal data or anything which can identify a user. GDPR and data protection laws may apply, ensure you do your own due diligence.
"""

DEVICE_MANUFACTURER = "device.manufacturer"
"""
The name of the device manufacturer.
Note: The Android OS provides this field via [Build](https://developer.android.com/reference/android/os/Build#MANUFACTURER). iOS apps SHOULD hardcode the value `Apple`.
"""

DEVICE_MODEL_IDENTIFIER = "device.model.identifier"
"""
The model identifier for the device.
Note: It's recommended this value represents a machine-readable version of the model identifier rather than the market or consumer-friendly name of the device.
"""

DEVICE_MODEL_NAME = "device.model.name"
"""
The marketing name for the device model.
Note: It's recommended this value represents a human-readable version of the device model rather than a machine-readable alternative.
"""
