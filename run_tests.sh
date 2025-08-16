#!/bin/bash
echo "================================================"
echo "OLLAMA TURBO AUTH HEADER TEST RESULTS"
echo "================================================"
echo ""
poetry run pytest tests/local_testing/test_ollama_turbo_integration.py::TestOllamaTurboAuthHeaders -v --tb=short
echo ""
echo "================================================"
echo "TEST SUMMARY: Check above for 4 PASSED"
echo "================================================"