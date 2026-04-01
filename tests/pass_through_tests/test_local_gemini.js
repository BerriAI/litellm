const { GoogleGenerativeAI, ModelParams, RequestOptions } = require("@google/generative-ai");

const modelParams = {
    model: 'gemini-2.5-flash-lite',
};
  
const requestOptions = {
    baseUrl: 'http://127.0.0.1:4000/gemini',
    customHeaders: {
        "tags": "gemini-js-sdk,gemini-2.5-flash-lite"
    }
};
  
const genAI = new GoogleGenerativeAI("sk-1234"); // litellm proxy API key
const model = genAI.getGenerativeModel(modelParams, requestOptions);

const testPrompt = "Explain how AI works";

async function main() {
  console.log("making request")
  try {
    const result = await model.generateContent(testPrompt);
    console.log(result.response.text());
  } catch (error) {
    console.error('Error details:', {
      name: error.name,
      message: error.message,
      cause: error.cause,
      // Check if there's a network error
      isNetworkError: error instanceof TypeError && error.message === 'fetch failed'
    });
    
    // Check if the server is running
    if (error instanceof TypeError && error.message === 'fetch failed') {
      console.error('Make sure your local server is running at http://localhost:4000');
    }
  }
}


async function main_streaming() {
    try {
        const streamingResult = await model.generateContentStream(testPrompt);
        for await (const item of streamingResult.stream) {
            console.log('stream chunk: ', JSON.stringify(item));
        }
        const aggregatedResponse = await streamingResult.response;
        console.log('aggregated response: ', JSON.stringify(aggregatedResponse));
    } catch (error) {
        console.error('Error details:', error);
    }
}

// main();
main_streaming();