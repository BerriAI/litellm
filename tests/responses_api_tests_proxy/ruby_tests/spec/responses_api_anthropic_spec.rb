require 'openai'
require 'rspec'

RSpec.describe 'OpenAI Responses API with Anthropic via LiteLLM' do
  let(:client) do
    OpenAI::Client.new(
      access_token: "sk-1234",
      uri_base: "http://0.0.0.0:4000"
    )
  end

  describe 'basic responses API' do
    it 'should create a basic response with anthropic model' do
      puts "\n=== Testing basic Responses API with Anthropic ==="

      response = client.responses.create(
        parameters: {
          model: "anthropic/claude-sonnet-4-5-20250929",
          input: "Say hello in one word",
          max_output_tokens: 50
        }
      )

      puts "Response: #{response.inspect}"

      # Validate response structure
      expect(response).to include('id')
      expect(response).to include('output')
      expect(response).to include('status')
      expect(response['status']).to eq('completed')

      # Validate usage is present
      expect(response).to include('usage')
      usage = response['usage']
      expect(usage).to include('input_tokens')
      expect(usage).to include('output_tokens')
      expect(usage).to include('total_tokens')
      expect(usage['input_tokens']).to be > 0
      expect(usage['output_tokens']).to be > 0

      # CRITICAL: Validate output_tokens_details for Ruby SDK compatibility
      # This is the key fix being tested - the Ruby SDK requires output_tokens_details
      # to be an object with reasoning_tokens, NOT null
      expect(usage).to include('output_tokens_details')
      output_tokens_details = usage['output_tokens_details']
      expect(output_tokens_details).not_to be_nil,
        "output_tokens_details should not be nil - Ruby SDK requires it to be an object"
      expect(output_tokens_details).to include('reasoning_tokens')
      expect(output_tokens_details['reasoning_tokens']).to be_a(Integer),
        "reasoning_tokens should be an Integer (can be 0 for non-reasoning models)"

      puts "✓ output_tokens_details validated: reasoning_tokens=#{output_tokens_details['reasoning_tokens']}"
      puts "=== Basic Responses API test passed ==="
    end

    it 'should create a streaming response with anthropic model' do
      puts "\n=== Testing streaming Responses API with Anthropic ==="

      collected_content = ""
      response_completed_event = nil

      client.responses.create(
        parameters: {
          model: "anthropic/claude-sonnet-4-5-20250929",
          input: "Say hello in one word",
          max_output_tokens: 50,
          stream: proc do |chunk, _bytesize|
            puts "Received chunk type: #{chunk['type']}" if chunk['type']

            if chunk['type'] == 'response.output_text.delta' && chunk['delta']
              collected_content += chunk['delta']
            elsif chunk['type'] == 'response.completed' && chunk['response']
              response_completed_event = chunk
            end
          end
        }
      )

      puts "Collected content: #{collected_content}"

      # Validate we received the completed event
      expect(response_completed_event).not_to be_nil,
        "Expected to receive response.completed event"

      response = response_completed_event['response']
      expect(response).not_to be_nil

      # For streaming, validate the final response
      expect(response).to include('id')
      expect(response).to include('status')

      # Validate usage in streaming response
      usage = response['usage']
      expect(usage).not_to be_nil, "Usage should be present in streaming response"
      expect(usage).to include('input_tokens')
      expect(usage).to include('output_tokens')

      # CRITICAL: Validate output_tokens_details for Ruby SDK compatibility in streaming
      expect(usage).to include('output_tokens_details')
      output_tokens_details = usage['output_tokens_details']
      expect(output_tokens_details).not_to be_nil,
        "output_tokens_details should not be nil in streaming response - Ruby SDK requires it to be an object"
      expect(output_tokens_details).to include('reasoning_tokens')
      expect(output_tokens_details['reasoning_tokens']).to be_a(Integer),
        "reasoning_tokens should be an Integer in streaming response"

      puts "✓ Streaming output_tokens_details validated: reasoning_tokens=#{output_tokens_details['reasoning_tokens']}"
      puts "=== Streaming Responses API test passed ==="
    end

    it 'should handle tool calls with anthropic model' do
      puts "\n=== Testing Responses API with tool calls using Anthropic ==="

      tools = [
        {
          type: "function",
          name: "get_weather",
          description: "Get current weather for a location",
          parameters: {
            type: "object",
            properties: {
              location: {
                type: "string",
                description: "City name"
              }
            },
            required: ["location"]
          }
        }
      ]

      response = client.responses.create(
        parameters: {
          model: "anthropic/claude-sonnet-4-5-20250929",
          input: "What's the weather in Paris?",
          tools: tools,
          max_output_tokens: 200
        }
      )

      puts "Tool call response: #{response.inspect}"

      # Validate response structure
      expect(response).to include('id')
      expect(response).to include('output')

      # Check for tool calls in output
      output = response['output']
      expect(output).to be_an(Array)
      expect(output.length).to be > 0

      # Validate usage with output_tokens_details
      usage = response['usage']
      expect(usage).not_to be_nil
      expect(usage).to include('output_tokens_details')
      output_tokens_details = usage['output_tokens_details']
      expect(output_tokens_details).not_to be_nil,
        "output_tokens_details should not be nil for tool call responses"

      puts "✓ Tool call response validated with output_tokens_details"
      puts "=== Tool calls test passed ==="
    end
  end

  describe 'OpenAI model comparison' do
    it 'should work the same way with OpenAI model' do
      puts "\n=== Testing Responses API with OpenAI model for comparison ==="

      response = client.responses.create(
        parameters: {
          model: "openai/gpt-4o-mini",
          input: "Say hello in one word",
          max_output_tokens: 50
        }
      )

      puts "OpenAI Response: #{response.inspect}"

      # Validate response structure matches Anthropic
      expect(response).to include('id')
      expect(response).to include('output')
      expect(response).to include('usage')

      usage = response['usage']
      expect(usage).to include('output_tokens_details')
      output_tokens_details = usage['output_tokens_details']
      expect(output_tokens_details).not_to be_nil,
        "OpenAI model should also have output_tokens_details"

      puts "✓ OpenAI model response validated"
      puts "=== OpenAI model test passed ==="
    end
  end
end
