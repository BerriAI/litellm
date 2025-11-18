#!/bin/bash

echo "🚀 Skypiea Gateway Local Test Script"
echo "=================================="

# Test health endpoint
echo "📊 Testing health endpoint..."
curl -s http://localhost:4000/health | jq '.healthy_count, .unhealthy_count' 2>/dev/null || echo "Health check response"

# Test models endpoint
echo -e "\n🤖 Testing models endpoint..."
curl -s http://localhost:4000/v1/models | jq '.data[0].id' 2>/dev/null || echo "Models response"

# Test chat completion
echo -e "\n💬 Testing chat completion..."
curl -s -X POST http://localhost:4000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openrouter/sherlock-think-alpha",
    "messages": [{"role": "user", "content": "Halo bro, test Skypiea Gateway! Siapa namamu?"}]
  }' | jq '.choices[0].message.content' 2>/dev/null || echo "Chat completion response"

# Test vision (if you have an image)
echo -e "\n👁️  Vision test example (replace with actual base64):"
echo "curl -X POST http://localhost:4000/chat/completions \\
  -H \"Content-Type: application/json\" \\
  -d '{
    \"model\": \"openrouter/sherlock-think-alpha\",
    \"messages\": [{\"role\": \"user\", \"content\": [
      {\"type\": \"text\", \"text\": \"What do you see in this image?\"},
      {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,YOUR_BASE64_HERE\"}}
    ]}]
  }'"

echo -e "\n✅ Test complete! Dashboard: http://localhost:4000/ui"
