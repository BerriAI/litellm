const openai = require('openai');

// set DEBUG=true in env
process.env.DEBUG=false;
async function runOpenAI() {
  const client = new openai.OpenAI({
    apiKey: 'sk-AbUQUhmKsghPMbOkTIntfQ',
    baseURL: 'http://0.0.0.0:8000'
  });

  try {
    const response = await client.chat.completions.create({
      model: 'azure-gpt-3.5',
      stream: true,
      max_tokens: 1000,
      messages: [
        {
          role: 'user',
          content: 'hi who are u ',
        },
      ],
    });

    console.log(response);
    for await (const chunk of response) {
      console.log(chunk);
      console.log(chunk.choices[0].delta.content);
    }
  } catch (error) {
    console.log("got this exception from server");
    console.error(error);
    console.log("done with exception from proxy");
  }
}

// Call the asynchronous function
runOpenAI();