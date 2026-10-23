export const DOCS_INDEX_URL = "https://docs.litellm.ai/llms.txt";

export async function fetchDocsIndex(signal: AbortSignal): Promise<string> {
  const response = await fetch(DOCS_INDEX_URL, { signal, redirect: "error" });
  if (!response.ok) throw new Error(`Docs index request failed with status ${response.status}`);
  return response.text();
}
