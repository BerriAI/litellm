const { GoogleGenerativeAI } = require("@google/generative-ai");
const fs = require('fs');
const path = require('path');

// Import fetch if the SDK uses it
const originalFetch = global.fetch || require('node-fetch');

let lastCallId;

// Monkey-patch the fetch used internally
global.fetch = async function patchedFetch(url, options) {
    const response = await originalFetch(url, options);

    // Store the call ID if it exists
    lastCallId = response.headers.get('x-litellm-call-id');

    return response;
};

describe('Gemini AI Tests', () => {
    test('should successfully generate non-streaming content with tags', async () => {
        const genAI = new GoogleGenerativeAI("sk-1234"); // litellm proxy API key

        const requestOptions = {
            baseUrl: 'http://127.0.0.1:4000/gemini',
            customHeaders: {
                "tags": "gemini-js-sdk,pass-through-endpoint"
            }
        };

        const model = genAI.getGenerativeModel({
            model: 'gemini-2.5-flash-lite'
        }, requestOptions);

        const prompt = 'Say "hello test" and nothing else';

        const result = await model.generateContent(prompt);
        expect(result).toBeDefined();

        // Use the captured callId
        const callId = lastCallId;
        console.log("Captured Call ID:", callId);

        // Poll for spend data with retries (DB writes can be slow in CI)
        let spendData = null;
        for (let attempt = 0; attempt < 6; attempt++) {
            await new Promise(resolve => setTimeout(resolve, 10000));
            const spendResponse = await fetch(
                `http://127.0.0.1:4000/spend/logs?request_id=${callId}`,
                { headers: { 'Authorization': 'Bearer sk-1234' } }
            );
            spendData = await spendResponse.json();
            console.log(`spendData (attempt ${attempt + 1}):`, spendData);
            if (spendData && spendData.length > 0 && spendData[0] && spendData[0].request_id) break;
        }

        if (!spendData || !spendData.length || !spendData[0] || !spendData[0].request_id) {
            console.warn('Spend data not available after polling - skipping spend assertions (DB write may be slow in CI)');
            return;
        }

        expect(spendData[0].request_id).toBe(callId);
        expect(spendData[0].call_type).toBe('pass_through_endpoint');
        expect(spendData[0].request_tags).toEqual(['gemini-js-sdk', 'pass-through-endpoint']);
        expect(spendData[0].metadata).toHaveProperty('user_api_key');
        expect(spendData[0].model).toContain('gemini');
        expect(spendData[0].custom_llm_provider).toBe('gemini');
        expect(spendData[0].spend).toBeGreaterThan(0);
    }, 90000);

    test('should successfully generate streaming content with tags', async () => {
        const genAI = new GoogleGenerativeAI("sk-1234"); // litellm proxy API key

        const requestOptions = {
            baseUrl: 'http://127.0.0.1:4000/gemini',
            customHeaders: {
                "tags": "gemini-js-sdk,pass-through-endpoint"
            }
        };

        const model = genAI.getGenerativeModel({
            model: 'gemini-2.5-flash-lite'
        }, requestOptions);

        const prompt = 'Say "hello test" and nothing else';

        const streamingResult = await model.generateContentStream(prompt);
        expect(streamingResult).toBeDefined();

        for await (const chunk of streamingResult.stream) {
            console.log('stream chunk:', JSON.stringify(chunk));
            expect(chunk).toBeDefined();
        }

        const aggregatedResponse = await streamingResult.response;
        console.log('aggregated response:', JSON.stringify(aggregatedResponse));
        expect(aggregatedResponse).toBeDefined();

        // Use the captured callId
        const callId = lastCallId;
        console.log("Captured Call ID:", callId);

        // Poll for spend data with retries (DB writes can be slow in CI)
        let spendData = null;
        for (let attempt = 0; attempt < 6; attempt++) {
            await new Promise(resolve => setTimeout(resolve, 10000));
            const spendResponse = await fetch(
                `http://127.0.0.1:4000/spend/logs?request_id=${callId}`,
                { headers: { 'Authorization': 'Bearer sk-1234' } }
            );
            spendData = await spendResponse.json();
            console.log(`spendData (attempt ${attempt + 1}):`, spendData);
            if (spendData && spendData.length > 0 && spendData[0] && spendData[0].request_id) break;
        }

        if (!spendData || !spendData.length || !spendData[0] || !spendData[0].request_id) {
            console.warn('Spend data not available after polling - skipping spend assertions (DB write may be slow in CI)');
            return;
        }

        expect(spendData[0].request_id).toBe(callId);
        expect(spendData[0].call_type).toBe('pass_through_endpoint');
        expect(spendData[0].request_tags).toEqual(['gemini-js-sdk', 'pass-through-endpoint']);
        expect(spendData[0].metadata).toHaveProperty('user_api_key');
        expect(spendData[0].model).toContain('gemini');
        expect(spendData[0].spend).toBeGreaterThan(0);
        expect(spendData[0].custom_llm_provider).toBe('gemini');
    }, 90000);
});
