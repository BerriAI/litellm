// Define the available test modes
export const TEST_MODES = [
  { value: "chat", label: "Chat - /chat/completions" },
  { value: "completion", label: "Completion - /completions" },
  { value: "embedding", label: "Embedding - /embeddings" },
  { value: "audio_speech", label: "Audio Speech - /audio/speech" },
  { value: "audio_transcription", label: "Audio Transcription - /audio/transcriptions" },
  { value: "image_generation", label: "Image Generation - /images/generations" },
  { value: "rerank", label: "Rerank - /rerank" },
  { value: "realtime", label: "Realtime - /realtime"}
]; 