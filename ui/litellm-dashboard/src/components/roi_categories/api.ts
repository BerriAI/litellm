import { apiClient } from "@/components/networking";

export function roiRequest<T>(token: string, path: string, body?: object | FormData): Promise<T> {
  const url = `/roi/categories${path}`;
  if (!body) return apiClient.get<T>(url, { accessToken: token });
  if (body instanceof FormData) return apiClient.post<T>(url, { accessToken: token, rawBody: body });
  return apiClient.post<T>(url, { accessToken: token, body });
}
