import openai from "openai";
import { getProxyBaseUrl } from "@/components/networking";
import NotificationManager from "@/components/molecules/notifications_manager";

export async function makeOpenAIAudioTranscriptionRequest(
  audioFile: File,
  updateUI: (transcription: string, model: string) => void,
  selectedModel: string,
  accessToken: string,
  tags?: string[],
  signal?: AbortSignal,
  language?: string,
  prompt?: string,
  responseFormat?: string,
  temperature?: number,
  customBaseUrl?: string,
) {
  // base url should be the current base_url
  const isLocal = process.env.NODE_ENV === "development";
  if (isLocal !== true) {
    console.log = function () {};
  }
  const proxyBaseUrl = customBaseUrl || getProxyBaseUrl();

  const client = new openai.OpenAI({
    apiKey: accessToken,
    baseURL: proxyBaseUrl,
    dangerouslyAllowBrowser: true,
    defaultHeaders: tags && tags.length > 0 ? { "x-litellm-tags": tags.join(",") } : undefined,
  });

  try {
    const response = await client.audio.transcriptions.create(
      {
        model: selectedModel,
        file: audioFile,
        ...(language ? { language: language } : {}),
        ...(prompt ? { prompt: prompt } : {}),
        ...(responseFormat ? { response_format: responseFormat as any } : {}),
        ...(temperature !== undefined ? { temperature: temperature } : {}),
      },
      { signal },
    );

    // The response is a transcription object with a text field
    if (response && response.text) {
      updateUI(response.text, selectedModel);
      NotificationManager.success(`Audio transcribed successfully`);
    } else {
      throw new Error("No transcription text in response");
    }
  } catch (error: any) {
    console.error("Error making audio transcription request:", error);

    if (signal?.aborted) {
    } else {
      let errorMessage = "Failed to transcribe audio";

      if (error?.error?.message) {
        errorMessage = error.error.message;
      } else if (error?.message) {
        errorMessage = error.message;
      }

      NotificationManager.fromBackend(`Audio transcription failed: ${errorMessage}`);
    }
    throw error; // Re-throw to allow the caller to handle the error
  }
}
