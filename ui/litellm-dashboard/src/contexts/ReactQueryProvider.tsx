"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ApiError } from "@/lib/http/client";

const MAX_RETRIES = 3;

const isRetryable = (error: unknown): boolean => !(error instanceof ApiError) || error.status >= 500;

export const shouldRetry = (failureCount: number, error: unknown): boolean =>
  isRetryable(error) && failureCount < MAX_RETRIES;

const queryClient = new QueryClient({ defaultOptions: { queries: { retry: shouldRetry } } });

export default function ReactQueryProvider({ children }: { children: React.ReactNode }) {
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}
