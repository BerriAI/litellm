const { VertexAI, RequestOptions } = require('@google-cloud/vertexai');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { writeFileSync } = require('fs');


// Import fetch if the SDK uses it
const originalFetch = global.fetch || require('node-fetch');

let lastCallId;

// Monkey-patch the fetch used internally
global.fetch = async function patchedFetch(url, options) {
    // Modify the URL to use HTTP instead of HTTPS
    if (url.startsWith('https://127.0.0.1:4000')) {
        url = url.replace('https://', 'http://');
    }
    console.log('Patched fetch sending request to:', url);
    
    const response = await originalFetch(url, options);
    
    // Store the call ID if it exists
    lastCallId = response.headers.get('x-litellm-call-id');
        
    return response;
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

// Configure Jest to retry flaky tests up to 3 times (useful for 429 rate limiting)
jest.retryTimes(3);

describe('Vertex AI Tests', () => {
    test('should successfully generate non-streaming content with tags', async () => {
        const vertexAI = new VertexAI({
            project: 'pathrise-convert-1606954137718',
            location: 'us-central1',
            apiEndpoint: "127.0.0.1:4000/vertex_ai"
        });

        const customHeaders = new Headers({
            "x-litellm-api-key": "sk-1234",
            "tags": "vertex-js-sdk,pass-through-endpoint"
        });

        const requestOptions = {
            customHeaders: customHeaders
        };

        const generativeModel = vertexAI.getGenerativeModel(
            { model: 'gemini-2.5-flash-lite' },
            requestOptions
        );

        const request = {
            contents: [{role: 'user', parts: [{text: 'Say "hello test" and nothing else'}]}]
        };

        const result = await generativeModel.generateContent(request);
        expect(result).toBeDefined();
        
        // Use the captured callId
        const callId = lastCallId;
        console.log("Captured Call ID:", callId);

        // Wait for spend to be logged
        await new Promise(resolve => setTimeout(resolve, 15000));

        // Check spend logs
        const spendResponse = await fetch(
            `http://127.0.0.1:4000/spend/logs?request_id=${callId}`,
            {
                headers: {
                    'Authorization': 'Bearer sk-1234'
                }
            }
        );
        
        const spendData = await spendResponse.json();
        console.log("spendData", spendData)
        expect(spendData).toBeDefined();
        expect(spendData[0].request_id).toBe(callId);
        expect(spendData[0].call_type).toBe('pass_through_endpoint');
        expect(spendData[0].request_tags).toEqual(['vertex-js-sdk', 'pass-through-endpoint']);
        expect(spendData[0].metadata).toHaveProperty('user_api_key');
        expect(spendData[0].model).toContain('gemini');
        expect(spendData[0].spend).toBeGreaterThan(0);
        expect(spendData[0].custom_llm_provider).toBe('vertex_ai');
    }, 25000);

    test('should successfully generate streaming content with tags', async () => {
        const vertexAI = new VertexAI({
            project: 'pathrise-convert-1606954137718',
            location: 'us-central1',
            apiEndpoint: "127.0.0.1:4000/vertex_ai"
        });

        const customHeaders = new Headers({
            "x-litellm-api-key": "sk-1234",
            "tags": "vertex-js-sdk,pass-through-endpoint"
        });

        const requestOptions = {
            customHeaders: customHeaders
        };

        const generativeModel = vertexAI.getGenerativeModel(
            { model: 'gemini-2.5-flash-lite' },
            requestOptions
        );

        const request = {
            contents: [{role: 'user', parts: [{text: 'Say "hello test" and nothing else'}]}]
        };

        const streamingResult = await generativeModel.generateContentStream(request);
        expect(streamingResult).toBeDefined();


        // Add some assertions
        expect(streamingResult).toBeDefined();
        
        for await (const item of streamingResult.stream) {
            console.log('stream chunk:', JSON.stringify(item));
            expect(item).toBeDefined();
        }

        const aggregatedResponse = await streamingResult.response;
        console.log('aggregated response:', JSON.stringify(aggregatedResponse));
        expect(aggregatedResponse).toBeDefined();

        // Use the captured callId
        const callId = lastCallId;
        console.log("Captured Call ID:", callId);

        // Wait for spend to be logged
        await new Promise(resolve => setTimeout(resolve, 15000));

        // Check spend logs
        const spendResponse = await fetch(
            `http://127.0.0.1:4000/spend/logs?request_id=${callId}`,
            {
                headers: {
                    'Authorization': 'Bearer sk-1234'
                }
            }
        );
        
        const spendData = await spendResponse.json();
        console.log("spendData", spendData)
        expect(spendData).toBeDefined();
        expect(spendData[0].request_id).toBe(callId);
        expect(spendData[0].call_type).toBe('pass_through_endpoint');
        expect(spendData[0].request_tags).toEqual(['vertex-js-sdk', 'pass-through-endpoint']);
        expect(spendData[0].metadata).toHaveProperty('user_api_key');
        expect(spendData[0].model).toContain('gemini');
        expect(spendData[0].spend).toBeGreaterThan(0);
        expect(spendData[0].custom_llm_provider).toBe('vertex_ai');
    }, 25000);
});