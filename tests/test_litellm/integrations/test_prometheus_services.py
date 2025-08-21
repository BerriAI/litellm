import json
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from litellm.integrations.prometheus_services import (
    PrometheusServicesLogger,
    ServiceMetrics,
    ServiceTypes,
)

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


def test_create_gauge_new():
    """Test creating a new gauge"""
    pl = PrometheusServicesLogger()

    # Create new gauge
    gauge = pl.create_gauge(service="test_service", type_of_request="size")

    assert gauge is not None
    assert pl._get_metric("litellm_test_service_size") is gauge


def test_update_gauge():
    """Test updating a gauge's value"""
    pl = PrometheusServicesLogger()

    # Create a gauge to test with
    gauge = pl.create_gauge(service="test_service", type_of_request="size")

    # Mock the labels method to verify it's called correctly
    with patch.object(gauge, "labels") as mock_labels:
        mock_gauge = AsyncMock()
        mock_labels.return_value = mock_gauge

        # Call update_gauge
        pl.update_gauge(gauge=gauge, labels="test_label", amount=42.5)

        # Verify correct methods were called
        mock_labels.assert_called_once_with("test_label")
        mock_gauge.set.assert_called_once_with(42.5)
