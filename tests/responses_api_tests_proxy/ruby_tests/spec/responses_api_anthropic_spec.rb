require 'openai'
require 'rspec'

RSpec.describe 'Ruby SDK Responses API with Anthropic' do
  let(:client) do
    OpenAI::Client.new(
      access_token: "sk-1234",
      uri_base: "http://0.0.0.0:4000"
    )
  end

  it 'should make a non-streaming response call with anthropic' do
    response = client.responses.create(
      parameters: {
        model: "anthropic/claude-sonnet-4-5-20250929",
        input: "Say hi",
        max_output_tokens: 20
      }
    )

    puts "Response: #{response}"
    expect(response).to include('id')
    expect(response).to include('output')
    expect(response['status']).to eq('completed')
  end

  it 'should make a streaming response call with anthropic' do
    chunks_received = 0

    client.responses.create(
      parameters: {
        model: "anthropic/claude-sonnet-4-5-20250929",
        input: "Say hi",
        max_output_tokens: 20,
        stream: proc do |chunk, _bytesize|
          puts "Chunk: #{chunk}"
          chunks_received += 1
        end
      }
    )

    expect(chunks_received).to be > 0
  end
end
