import openai from "openai";
import { getProxyBaseUrl } from "@/components/networking";
import NotificationManager from "@/components/molecules/notifications_manager";
import type { OpenAIVoice } from "../chat_ui/chatConstants";

export async function makeOpenAIAudioSpeechRequest(
  input: string,
  voice: OpenAIVoice,
  updateUI: (audioUrl: string, model: string) => void,
  selectedModel: string,
  accessToken: string,
  tags?: string[],
  signal?: AbortSignal,
  responseFormat?: string,
  speed?: number,
  customBaseUrl?: string,
) {
  // base url should be the current base_url
  const isLocal = process.env.NODE_ENV === "development";
  if (isLocal !== true) {
    console.log = function () {};
  }
  console.log("isLocal:", isLocal);
  const proxyBaseUrl = customBaseUrl || getProxyBaseUrl();
  const client = new openai.OpenAI({
    apiKey: accessToken,
    baseURL: proxyBaseUrl,
    dangerouslyAllowBrowser: true,
    defaultHeaders: tags && tags.length > 0 ? { "x-litellm-tags": tags.join(",") } : undefined,
  });

  try {
    const response = await client.audio.speech.create(
      {
        model: selectedModel,
        input: input,
        voice,
        ...(responseFormat ? { response_format: responseFormat as any } : {}),
        ...(speed ? { speed: speed } : {}),
      },
      { signal },
    );

    // Convert the response to a blob and create an object URL
    // The response from OpenAI SDK in the browser is a Response object with a blob() method
    const blob = await response.blob();
    const audioUrl = URL.createObjectURL(blob);

    updateUI(audioUrl, selectedModel);
  } catch (error) {
    if (signal?.aborted) {
      console.log("Audio speech request was cancelled");
    } else {
      NotificationManager.fromBackend(`Error occurred while generating speech. Please try again. Error: ${error}`);
    }
    throw error; // Re-throw to allow the caller to handle the error
  }
}
