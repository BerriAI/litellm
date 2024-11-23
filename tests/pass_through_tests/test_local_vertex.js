const { VertexAI, RequestOptions } = require('@google-cloud/vertexai');


// Import fetch if the SDK uses it
const originalFetch = global.fetch || require('node-fetch');

// Monkey-patch the fetch used internally
global.fetch = async function patchedFetch(url, options) {
    // Modify the URL to use HTTP instead of HTTPS
    if (url.startsWith('https://localhost:4000')) {
        url = url.replace('https://', 'http://');
    }
    console.log('Patched fetch sending request to:', url);
    return originalFetch(url, options);
};

const vertexAI = new VertexAI({
    project: 'adroit-crow-413218',
    location: 'us-central1',
    apiEndpoint: "localhost:4000/vertex-ai"
});


// Use customHeaders in RequestOptions
const requestOptions = {
    customHeaders: new Headers({
        "x-litellm-api-key": "sk-1234"
    })
};

const generativeModel = vertexAI.getGenerativeModel(
    { model: 'gemini-1.0-pro' },
    requestOptions
);

async function streamingResponse() {
    try {
        const request = {
            contents: [{role: 'user', parts: [{text: 'How are you doing today tell me your name?'}]}],
          };
        const streamingResult = await generativeModel.generateContentStream(request);
        for await (const item of streamingResult.stream) {
            console.log('stream chunk: ', JSON.stringify(item));
        }
        const aggregatedResponse = await streamingResult.response;
        console.log('aggregated response: ', JSON.stringify(aggregatedResponse));
    } catch (error) {
        console.error('Error:', error);
    }
}


async function nonStreamingResponse() {
    try {
        const request = {
            contents: [{role: 'user', parts: [{text: 'How are you doing today tell me your name?'}]}],
          };
        const response = await generativeModel.generateContent(request);
        console.log('non streaming response: ', JSON.stringify(response));
    } catch (error) {
        console.error('Error:', error);
    }
}



streamingResponse();
nonStreamingResponse();