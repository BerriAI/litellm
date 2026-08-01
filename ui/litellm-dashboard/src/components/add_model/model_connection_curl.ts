export const TEST_MODE_ENDPOINT_PATHS: Record<string, string> = {
  chat: "/chat/completions",
  completion: "/completions",
  embedding: "/embeddings",
  audio_speech: "/audio/speech",
  audio_transcription: "/audio/transcriptions",
  image_generation: "/images/generations",
  video_generation: "/videos",
  rerank: "/rerank",
  realtime: "/realtime",
  batch: "/batch",
  ocr: "/ocr",
};

export const buildRequestUrl = (apiBase: string, testMode: string): string => {
  const path = TEST_MODE_ENDPOINT_PATHS[testMode];
  if (!path) return apiBase;
  const trimmed = apiBase.replace(/\/+$/, "");
  return trimmed.endsWith(path) ? trimmed : `${trimmed}${path}`;
};

export const buildTestConnectionCurl = ({
  apiBase,
  testMode,
  requestBody,
  requestHeaders,
}: {
  apiBase: string;
  testMode: string;
  requestBody: Record<string, unknown>;
  requestHeaders: Record<string, string>;
}): string => {
  const url = buildRequestUrl(apiBase, testMode);
  const body = JSON.stringify(requestBody, null, 2);
  const headerString = Object.entries(requestHeaders)
    .map(([key, value]) => `-H '${key}: ${value}'`)
    .join(" \\\n  ");
  const headerBlock = headerString ? `${headerString} \\\n  ` : "";
  return `curl -X POST \\\n  ${url} \\\n  ${headerBlock}-H 'Content-Type: application/json' \\\n  -d '${body}'`;
};
