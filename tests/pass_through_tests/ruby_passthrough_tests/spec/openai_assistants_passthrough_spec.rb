require 'openai'
require 'rspec'

RSpec.describe 'OpenAI Assistants Passthrough' do
  let(:client) do
    OpenAI::Client.new(
      access_token: "sk-1234",
      uri_base: "http://0.0.0.0:4000/openai"
    )
  end


  it 'performs basic assistant operations' do
    assistant = client.assistants.create(
      parameters: {
        name: "Math Tutor",
        instructions: "You are a personal math tutor. Write and run code to answer math questions.",
        tools: [{ type: "code_interpreter" }],
        model: "gpt-4o"
      }
    )
    expect(assistant).to include('id')
    expect(assistant['name']).to eq("Math Tutor")

    assistants_list = client.assistants.list
    expect(assistants_list['data']).to be_an(Array)
    expect(assistants_list['data']).to include(include('id' => assistant['id']))

    retrieved_assistant = client.assistants.retrieve(id: assistant['id'])
    expect(retrieved_assistant).to eq(assistant)

    deleted_assistant = client.assistants.delete(id: assistant['id'])
    expect(deleted_assistant['deleted']).to be true
    expect(deleted_assistant['id']).to eq(assistant['id'])
  end

  it 'performs streaming assistant operations' do
    puts "\n=== Starting Streaming Assistant Test ==="
    
    assistant = client.assistants.create(
      parameters: {
        name: "Math Tutor",
        instructions: "You are a personal math tutor. Write and run code to answer math questions.",
        tools: [{ type: "code_interpreter" }],
        model: "gpt-4o"
      }
    )
    puts "Created assistant: #{assistant['id']}"
    expect(assistant).to include('id')

    thread = client.threads.create
    puts "Created thread: #{thread['id']}"
    expect(thread).to include('id')

    message = client.messages.create(
      thread_id: thread['id'],
      parameters: {
        role: "user",
        content: "I need to solve the equation `3x + 11 = 14`. Can you help me?"
      }
    )
    puts "Created message: #{message['id']}"
    puts "User question: #{message['content']}"
    expect(message).to include('id')
    expect(message['role']).to eq('user')

    puts "\nStarting streaming response:"
    puts "------------------------"
    run = client.runs.create(
      thread_id: thread['id'],
      parameters: {
        assistant_id: assistant['id'],
        max_prompt_tokens: 256,
        max_completion_tokens: 16,
        stream: proc do |chunk, _bytesize|
          puts "Received chunk: #{chunk.inspect}"  # Debug: Print raw chunk
          if chunk["object"] == "thread.message.delta"
            content = chunk.dig("delta", "content")
            puts "Content: #{content.inspect}"  # Debug: Print content structure
            if content && content[0] && content[0]["text"]
              print content[0]["text"]["value"]
              $stdout.flush  # Ensure output is printed immediately
            end
          end
        end
      }
    )
    puts "\n------------------------"
    puts "Run completed: #{run['id']}"
    expect(run).not_to be_nil
  ensure
    client.assistants.delete(id: assistant['id']) if assistant && assistant['id']
    client.threads.delete(id: thread['id']) if thread && thread['id']
  end
end 