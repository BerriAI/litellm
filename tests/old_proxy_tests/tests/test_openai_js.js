const openai = require('openai');

// set DEBUG=true in env
process.env.DEBUG=false;
async function runOpenAI() {
  const client = new openai.OpenAI({
    apiKey: 'sk-1234',
    baseURL: 'http://0.0.0.0:4000'
  });



  try {
    const response = await client.chat.completions.create({
      model: 'anthropic-claude-v2.1',
      stream: true,
      messages: [
        {
          role: 'user',
          content: 'write a 20 pg essay about YC '.repeat(6000),
        },
      ],
    });

    console.log(response);
    let original = '';
    for await (const chunk of response) {
      original += chunk.choices[0].delta.content;
      console.log(original);
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