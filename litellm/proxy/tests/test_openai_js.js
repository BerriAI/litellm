const openai = require('openai');

// set DEBUG=true in env
process.env.DEBUG=false;
async function runOpenAI() {
  const client = new openai.OpenAI({
    apiKey: 'your_api_key_here',
    baseURL: 'http://0.0.0.0:8000'
  });

  try {
    const response = await client.chat.completions.create({
      model: 'azure-gpt-3.5',
      messages: [
        {
          role: 'user',
          content: 'this is a test request, write a short poem'.repeat(2000),
        },
      ],
    });

    console.log(response);
  } catch (error) {
    console.log("got this exception from server");
    console.error(error);
    console.log("done with exception from proxy");
  }
}

// Call the asynchronous function
runOpenAI();