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


describe('Vertex AI Tests', () => {
    test('should successfully generate content from Vertex AI', async () => {
        const vertexAI = new VertexAI({
            project: 'adroit-crow-413218',
            location: 'us-central1',
            apiEndpoint: "localhost:4000/vertex-ai"
        });

        const customHeaders = new Headers({
            "X-Litellm-Api-Key": "sk-1234"
        });

        const requestOptions = {
            customHeaders: customHeaders
        };

        const generativeModel = vertexAI.getGenerativeModel(
            { model: 'gemini-1.0-pro' },
            requestOptions
        );

        const request = {
            contents: [{role: 'user', parts: [{text: 'How are you doing today tell me your name?'}]}],
        };

        const streamingResult = await generativeModel.generateContentStream(request);
        
        // Add some assertions
        expect(streamingResult).toBeDefined();
        
        for await (const item of streamingResult.stream) {
            console.log('stream chunk:', JSON.stringify(item));
            expect(item).toBeDefined();
        }

        const aggregatedResponse = await streamingResult.response;
        console.log('aggregated response:', JSON.stringify(aggregatedResponse));
        expect(aggregatedResponse).toBeDefined();
    });


    test('should successfully generate non-streaming content from Vertex AI', async () => {
        const vertexAI = new VertexAI({project: 'adroit-crow-413218', location: 'us-central1', apiEndpoint: "localhost:4000/vertex-ai"});
        const customHeaders = new Headers({"X-Litellm-Api-Key": "sk-1234"});
        const requestOptions = {customHeaders: customHeaders};
        const generativeModel = vertexAI.getGenerativeModel({model: 'gemini-1.0-pro'}, requestOptions);
        const request = {contents: [{role: 'user', parts: [{text: 'What is 2+2?'}]}]};

        const result = await generativeModel.generateContent(request);
        expect(result).toBeDefined();
        expect(result.response).toBeDefined();
        console.log('non-streaming response:', JSON.stringify(result.response));
    });
});