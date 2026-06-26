import { getGlobalLitellmHeaderName, getProxyBaseUrl } from "@/components/networking";
import { ApiError, createApiClient } from "@/lib/http/client";
import NotificationManager from "@/components/molecules/notifications_manager";

interface OCRPage {
  index: number;
  markdown: string;
}

interface OCRResponse {
  pages?: OCRPage[];
  model?: string;
}

interface MakeOCRRequestArgs {
  file: File;
  updateUI: (markdown: string, model: string) => void;
  selectedModel: string;
  accessToken: string;
  tags?: string[];
  signal?: AbortSignal;
  customBaseUrl?: string;
}

const formatOcrResponse = (data: OCRResponse): string => {
  const pages = data.pages ?? [];
  if (pages.length === 0) {
    return "No text was extracted from the document.";
  }
  return pages.map((page) => `## Page ${page.index + 1}\n\n${page.markdown ?? ""}`).join("\n\n---\n\n");
};

export async function makeOpenAIOCRRequest({
  file,
  updateUI,
  selectedModel,
  accessToken,
  tags,
  signal,
  customBaseUrl,
}: MakeOCRRequestArgs): Promise<void> {
  const client = createApiClient({
    getBaseUrl: () => customBaseUrl || getProxyBaseUrl(),
    getAuthHeaderName: getGlobalLitellmHeaderName,
  });

  const formData = new FormData();
  formData.append("model", selectedModel);
  formData.append("file", file);

  const extraHeaders = tags && tags.length > 0 ? { "x-litellm-tags": tags.join(",") } : undefined;

  try {
    const data = await client.post<OCRResponse>("/v1/ocr", {
      accessToken,
      rawBody: formData,
      signal,
      headers: extraHeaders,
    });
    updateUI(formatOcrResponse(data), data.model ?? selectedModel);
    NotificationManager.success("OCR completed successfully");
  } catch (error) {
    if (signal?.aborted) {
      throw error;
    }
    const message = error instanceof ApiError ? error.message : error instanceof Error ? error.message : String(error);
    NotificationManager.fromBackend(`OCR failed: ${message}`);
    throw error;
  }
}
