
const {
    FunctionDeclarationSchemaType,
    HarmBlockThreshold,
    HarmCategory,
    VertexAI,
    RequestOptions
  } = require('@google-cloud/vertexai');
  
  const project = 'adroit-crow-413218';
  const location = 'us-central1';
  const textModel =  'gemini-1.0-pro';
  const visionModel = 'gemini-1.0-pro-vision';

  
  const vertexAI = new VertexAI({project: project, location: location, apiEndpoint: "localhost:4000/vertex-ai"});
  
  // Instantiate Gemini models
  const generativeModel = vertexAI.getGenerativeModel({
      model: textModel,
      // The following parameters are optional
      // They can also be passed to individual content generation requests
      safetySettings: [{category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE}],
      generationConfig: {maxOutputTokens: 256},
      systemInstruction: {
        role: 'system',
        parts: [{"text": `For example, you are a helpful customer service agent. tell me your name. in 5 pages`}]
      },
  })

async function streamGenerateContent() {
    const request = {
      contents: [{role: 'user', parts: [{text: 'How are you doing today?'}]}],
    };
    const streamingResult = await generativeModel.generateContentStream(request);
    for await (const item of streamingResult.stream) {
      console.log('stream chunk: ', JSON.stringify(item));
    }
    const aggregatedResponse = await streamingResult.response;
    console.log('aggregated response: ', JSON.stringify(aggregatedResponse));
  };
  
  streamGenerateContent();