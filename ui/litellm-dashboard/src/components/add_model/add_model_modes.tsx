// Define the available test modes
export const TEST_MODES = [
  { value: "chat", label: "Chat - /chat/completions" },
  { value: "completion", label: "Completion - /completions" },
  { value: "embedding", label: "Embedding - /embeddings" },
  { value: "audio_speech", label: "Audio Speech - /audio/speech" },
  { value: "audio_transcription", label: "Audio Transcription - /audio/transcriptions" },
  { value: "image_generation", label: "Image Generation - /images/generations" },
  { value: "rerank", label: "Rerank - /rerank" },
  { value: "realtime", label: "Realtime - /realtime" },
  { value: "batch", label: "Batch - /batch" },
  { value: "ocr", label: "OCR - /ocr" },
];

// Define the available auto router routing strategies
export const AUTO_ROUTER_MODES = [
  { value: "simple-shuffle", label: "Simple Shuffle - Random selection from available models" },
  { value: "least-busy", label: "Least Busy - Route to model with lowest current load" },
  { value: "latency-based", label: "Latency Based - Route to model with best response time" },
  { value: "cost-based", label: "Cost Based - Route to most cost-effective model" },
  { value: "usage-based", label: "Usage Based - Route based on historical usage patterns" },
  { value: "custom", label: "Custom - Use custom routing logic defined in config" },
];
