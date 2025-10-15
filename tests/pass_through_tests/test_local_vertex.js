const { VertexAI, RequestOptions } = require('@google-cloud/vertexai');



const vertexAI = new VertexAI({
    project: 'pathrise-convert-1606954137718',
    location: 'us-central1',
    apiEndpoint: "127.0.0.1:4000/vertex-ai"
});

// Create customHeaders using Headers
const customHeaders = new Headers({
    "X-Litellm-Api-Key": "sk-1234",
    tags: "vertexjs,test-2"
});

// Use customHeaders in RequestOptions
const requestOptions = {
    customHeaders: customHeaders,
};

const generativeModel = vertexAI.getGenerativeModel(
    { model: 'gemini-2.5-flash-lite' },
    requestOptions
);

async function testModel() {
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

testModel();