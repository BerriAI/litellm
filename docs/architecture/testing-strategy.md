# Testing Strategy

## Testing Pyramid
```text
        E2E Tests
        /        \
    Integration Tests
    /            \
Frontend Unit  Backend Unit
```

## Test Organization

### Frontend Tests
```text
ui/tests/
├── unit/              # Component tests
├── integration/       # API integration
└── e2e/              # User flows
```

### Backend Tests
```text
tests/
├── litellm_tests/     # Core library units
├── proxy_unit_tests/  # Proxy server units
├── router_unit_tests/ # Router logic
└── integration_tests/ # Provider integration
```

### E2E Tests
```text
tests/e2e/
├── auth_flows.py
├── api_workflows.py
└── admin_operations.py
```

## Test Examples

### Frontend Component Test
```typescript
describe('KeyList', () => {
  it('displays API keys', async () => {
    const keys = [
      { token: 'sk-123', key_alias: 'Test Key' }
    ];
    
    render(<KeyList keys={keys} />);
    
    expect(screen.getByText('Test Key')).toBeInTheDocument();
  });
});
```

### Backend API Test
```python
@pytest.mark.asyncio
async def test_generate_key():
    """Test API key generation"""
    response = await client.post(
        "/key/generate",
        json={"key_alias": "Test Key"},
        headers={"Authorization": f"Bearer {MASTER_KEY}"}
    )
    
    assert response.status_code == 200
    assert "key" in response.json()
    assert response.json()["key"].startswith("sk-")
```

### E2E Test
```python
@pytest.mark.asyncio
async def test_complete_flow():
    """Test complete request flow"""
    # Generate key
    key_response = await generate_test_key()
    api_key = key_response["key"]
    
    # Make completion request
    completion = await client.post(
        "/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}]
        },
        headers={"Authorization": f"Bearer {api_key}"}
    )
    
    assert completion.status_code == 200
    assert "choices" in completion.json()
```
