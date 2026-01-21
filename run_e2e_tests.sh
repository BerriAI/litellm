#!/bin/bash
# E2E Test Script for Admin User Usage Endpoint
# Run this after: poetry install --with dev

set -e

BASE_URL="${BASE_URL:-http://localhost:4000}"
AUTH_TOKEN="${AUTH_TOKEN:-sk-test-master-key-12345}"

echo "======================================================================="
echo "E2E TESTS: Admin User Usage Endpoint"
echo "======================================================================="
echo ""
echo "Testing endpoint: $BASE_URL/admin/users/daily/activity"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

passed=0
failed=0

run_test() {
    local test_name="$1"
    local url="$2"
    local expected_status="$3"

    echo -n "Test: $test_name ... "

    response=$(curl -s -w "\n%{http_code}" -X GET "$url" \
        -H "Authorization: Bearer $AUTH_TOKEN" \
        -H "Content-Type: application/json" 2>&1)

    status=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)

    if [ "$status" == "$expected_status" ]; then
        echo -e "${GREEN}✓ PASS${NC} (Status: $status)"
        ((passed++))

        # Additional validation for 200 responses
        if [ "$status" == "200" ]; then
            # Check if response contains required keys
            if echo "$body" | grep -q '"summary"' && \
               echo "$body" | grep -q '"top_users"' && \
               echo "$body" | grep -q '"users"' && \
               echo "$body" | grep -q '"pagination"'; then
                echo "     ✓ Response structure valid"
            else
                echo -e "     ${YELLOW}⚠ Warning: Response structure may be incomplete${NC}"
            fi
        fi
    else
        echo -e "${RED}✗ FAIL${NC} (Status: $status, expected: $expected_status)"
        ((failed++))
        if [ ! -z "$body" ]; then
            echo "     Response: $body" | head -c 200
        fi
    fi
}

echo "Running tests..."
echo ""

# Test 1: Basic request
run_test "Basic request (all users)" \
    "$BASE_URL/admin/users/daily/activity?start_date=2026-01-01&end_date=2026-01-21" \
    "200"

# Test 2: Tag filter
run_test "Filter by tag (Claude Code)" \
    "$BASE_URL/admin/users/daily/activity?start_date=2026-01-01&end_date=2026-01-21&tag_filters=User-Agent:claude-code" \
    "200"

# Test 3: Min spend filter
run_test "Filter by min spend (power users)" \
    "$BASE_URL/admin/users/daily/activity?start_date=2026-01-01&end_date=2026-01-21&min_spend=200" \
    "200"

# Test 4: Sort by requests
run_test "Sort by requests" \
    "$BASE_URL/admin/users/daily/activity?start_date=2026-01-01&end_date=2026-01-21&sort_by=requests&sort_order=desc" \
    "200"

# Test 5: Pagination
run_test "Pagination (page 2)" \
    "$BASE_URL/admin/users/daily/activity?start_date=2026-01-01&end_date=2026-01-21&page=2&page_size=25" \
    "200"

# Test 6: Combined filters
run_test "Combined filters (tag + spend + sort)" \
    "$BASE_URL/admin/users/daily/activity?start_date=2026-01-01&end_date=2026-01-21&tag_filters=User-Agent:claude-code&min_spend=200&sort_by=spend" \
    "200"

# Test 7: Invalid date format (should fail)
run_test "Invalid date format (error case)" \
    "$BASE_URL/admin/users/daily/activity?start_date=01-01-2026&end_date=2026-01-21" \
    "400"

# Test 8: Missing required parameters (should fail)
run_test "Missing required parameters (error case)" \
    "$BASE_URL/admin/users/daily/activity?start_date=2026-01-01" \
    "400"

echo ""
echo "======================================================================="
echo "TEST RESULTS"
echo "======================================================================="
echo ""
echo -e "Passed: ${GREEN}$passed${NC}"
echo -e "Failed: ${RED}$failed${NC}"
echo "Total:  $((passed + failed))"
echo ""

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED${NC}"
    echo ""
    echo "The endpoint is working correctly!"
    exit 0
else
    echo -e "${RED}❌ SOME TESTS FAILED${NC}"
    echo ""
    echo "Please check the errors above."
    exit 1
fi
