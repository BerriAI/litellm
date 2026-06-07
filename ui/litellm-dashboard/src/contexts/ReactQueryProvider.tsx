"use client";

import { QueryClient, QueryClientProvider, QueryCache } from "@tanstack/react-query";
import { handleError } from "@/components/networking";
import { deriveErrorMessage } from "@/lib/http/client";

const queryClient = new QueryClient({
  queryCache: new QueryCache({
    onError: (error) => handleError(deriveErrorMessage(error)),
  }),
});

export default function ReactQueryProvider({ children }: { children: React.ReactNode }) {
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}
