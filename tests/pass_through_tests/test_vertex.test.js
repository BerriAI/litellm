const { VertexAI, RequestOptions } = require('@google-cloud/vertexai');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { writeFileSync } = require('fs');


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

function loadVertexAiCredentials() {
    console.log("loading vertex ai credentials");
    const filepath = path.dirname(__filename);
    const vertexKeyPath = path.join(filepath, "vertex_key.json");

    // Initialize default empty service account data
    let serviceAccountKeyData = {};

    // Try to read existing vertex_key.json
    try {
        const content = fs.readFileSync(vertexKeyPath, 'utf8');
        if (content && content.trim()) {
            serviceAccountKeyData = JSON.parse(content);
        }
    } catch (error) {
        // File doesn't exist or is invalid, continue with empty object
    }

    // Update with environment variables
    const privateKeyId = process.env.VERTEX_AI_PRIVATE_KEY_ID || "";
    const privateKey = (process.env.VERTEX_AI_PRIVATE_KEY || "").replace(/\\n/g, "\n");
    
    serviceAccountKeyData.private_key_id = privateKeyId;
    serviceAccountKeyData.private_key = privateKey;

    // Create temporary file
    const tempFilePath = path.join(os.tmpdir(), `vertex-credentials-${Date.now()}.json`);
    writeFileSync(tempFilePath, JSON.stringify(serviceAccountKeyData, null, 2));
    
    // Set environment variable
    process.env.GOOGLE_APPLICATION_CREDENTIALS = tempFilePath;
}

// Run credential loading before tests
beforeAll(() => {
    loadVertexAiCredentials();
});



describe('Vertex AI Tests', () => {
    test('should successfully generate content from Vertex AI', async () => {
        const vertexAI = new VertexAI({
            project: 'pathrise-convert-1606954137718',
            location: 'us-central1',
            apiEndpoint: "localhost:4000/vertex-ai"
        });

        const customHeaders = new Headers({
            "x-litellm-api-key": "sk-1234"
        });

        const requestOptions = {
            customHeaders: customHeaders
        };

        const generativeModel = vertexAI.getGenerativeModel(
            { model: 'gemini-2.5-flash-lite' },
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
        const vertexAI = new VertexAI({project: 'pathrise-convert-1606954137718', location: 'us-central1', apiEndpoint: "localhost:4000/vertex-ai"});
        const customHeaders = new Headers({"x-litellm-api-key": "sk-1234"});
        const requestOptions = {customHeaders: customHeaders};
        const generativeModel = vertexAI.getGenerativeModel({model: 'gemini-2.5-flash-lite'}, requestOptions);
        const request = {contents: [{role: 'user', parts: [{text: 'What is 2+2?'}]}]};

        const result = await generativeModel.generateContent(request);
        expect(result).toBeDefined();
        expect(result.response).toBeDefined();
        console.log('non-streaming response:', JSON.stringify(result.response));
    });
});