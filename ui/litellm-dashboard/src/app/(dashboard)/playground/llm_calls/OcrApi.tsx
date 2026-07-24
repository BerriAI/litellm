import NotificationManager from "@/components/molecules/notifications_manager";
import { getGlobalLitellmHeaderName, getProxyBaseUrl } from "@/components/networking";
import { createApiClient } from "@/lib/http/client";

interface OcrRequestParams {
  file: File;
  updateUI: (text: string, model: string) => void;
  selectedModel: string;
  accessToken: string;
  tags?: string[];
  signal?: AbortSignal;
  customBaseUrl?: string;
}

const formatOcrResponse = (response: unknown): string => {
  if (typeof response !== "object" || response === null || !("pages" in response)) {
    return JSON.stringify(response, null, 2);
  }

  const pages = (response as { pages?: unknown }).pages;
  if (!Array.isArray(pages)) {
    return JSON.stringify(response, null, 2);
  }

  const markdown = pages
    .map((page) =>
      typeof page === "object" && page !== null && "markdown" in page
        ? (page as { markdown?: unknown }).markdown
        : undefined,
    )
    .filter((pageMarkdown): pageMarkdown is string => typeof pageMarkdown === "string" && pageMarkdown.length > 0)
    .join("\n\n");

  return markdown || JSON.stringify(response, null, 2);
};

export async function makeOpenAIOcrRequest({
  file,
  updateUI,
  selectedModel,
  accessToken,
  tags,
  signal,
  customBaseUrl,
}: OcrRequestParams) {
  const client = createApiClient({
    getBaseUrl: () => customBaseUrl || getProxyBaseUrl(),
    getAuthHeaderName: getGlobalLitellmHeaderName,
    onError: (message) => NotificationManager.fromBackend(`OCR failed: ${message}`),
  });
  const formData = new FormData();
  formData.append("model", selectedModel);
  formData.append("file", file);

  const responseJson = await client.post<unknown>("/v1/ocr", {
    accessToken,
    rawBody: formData,
    headers: {
      ...(tags && tags.length > 0 ? { "x-litellm-tags": tags.join(",") } : {}),
    },
    signal,
  });

  updateUI(formatOcrResponse(responseJson), selectedModel);
  NotificationManager.success("OCR completed successfully");
}
