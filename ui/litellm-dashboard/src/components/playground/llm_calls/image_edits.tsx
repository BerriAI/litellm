import openai from "openai";
import { getProxyBaseUrl } from "@/components/networking";
import NotificationManager from "@/components/molecules/notifications_manager";

export async function makeOpenAIImageEditsRequest(
  imageFiles: File | File[],
  prompt: string,
  updateUI: (imageUrl: string, model: string) => void,
  selectedModel: string,
  accessToken: string,
  tags?: string[],
  signal?: AbortSignal,
) {
  // base url should be the current base_url
  const isLocal = process.env.NODE_ENV === "development";
  if (isLocal !== true) {
    console.log = function () {};
  }
  console.log("isLocal:", isLocal);
  const proxyBaseUrl = getProxyBaseUrl();

  const client = new openai.OpenAI({
    apiKey: accessToken,
    baseURL: proxyBaseUrl,
    dangerouslyAllowBrowser: true,
    defaultHeaders: tags && tags.length > 0 ? { "x-litellm-tags": tags.join(",") } : undefined,
  });

  try {
    // handle single and multiple images
    const imagesToProcess = Array.isArray(imageFiles) ? imageFiles : [imageFiles];

    // For multiple images, we'll make separate API calls for each image
    // since OpenAI's edit endpoint processes one image at a time
    const results = [];

    for (let i = 0; i < imagesToProcess.length; i++) {
      const image = imagesToProcess[i];
      console.log(`Processing image ${i + 1} of ${imagesToProcess.length}`);

      const response = await client.images.edit(
        {
          model: selectedModel,
          image: image,
          prompt: prompt,
        },
        { signal },
      );

      console.log(`Response for image ${i + 1}:`, response.data);

      if (response.data && response.data[0]) {
        // Handle either URL or base64 data from response
        if (response.data[0].url) {
          // Use the URL directly
          updateUI(response.data[0].url, selectedModel);
          results.push(response.data[0].url);
        } else if (response.data[0].b64_json) {
          // Convert base64 to data URL format
          const base64Data = response.data[0].b64_json;
          const dataUrl = `data:image/png;base64,${base64Data}`;
          updateUI(dataUrl, selectedModel);
          results.push(dataUrl);
        }
      }
    }

    if (results.length > 1) {
      NotificationManager.success(`Successfully processed ${results.length} images`);
    }
  } catch (error: any) {
    console.error("Error making image edit request:", error);

    if (signal?.aborted) {
      console.log("Image edits request was cancelled");
    } else {
      let errorMessage = "Failed to edit image(s)";

      if (error?.error?.message) {
        errorMessage = error.error.message;
      } else if (error?.message) {
        errorMessage = error.message;
      }

      NotificationManager.fromBackend(`Image edit failed: ${errorMessage}`);
    }
    throw error; // Re-throw to allow the caller to handle the error
  }
}
