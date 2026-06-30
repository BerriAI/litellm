import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import { $api } from "@/lib/http/api";

function UserCount() {
  const { data, isPending } = $api.useQuery("get", "/user/list");
  if (isPending) return <span>loading</span>;
  return <span>users: {(data as { users?: unknown[] })?.users?.length ?? 0}</span>;
}

function renderWithClient(ui: ReactNode) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>);
}

describe("$api (openapi-fetch + openapi-react-query)", () => {
  it("drives a schema-typed GET /user/list through the query client", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ users: [{ user_id: "u1" }, { user_id: "u2" }] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    renderWithClient(<UserCount />);

    await waitFor(() => expect(screen.getByText("users: 2")).toBeInTheDocument());
    const requested = fetchSpy.mock.calls[0]?.[0];
    const requestedUrl = requested instanceof Request ? requested.url : String(requested);
    expect(requestedUrl).toContain("/user/list");
  });

  it("surfaces a non-2xx response as an ApiError in the query error state", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "boom" }), {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }),
    );

    function Status() {
      const { error, isError } = $api.useQuery("get", "/user/list");
      return <span>{isError ? `error: ${(error as Error).message}` : "ok"}</span>;
    }

    renderWithClient(<Status />);
    await waitFor(() => expect(screen.getByText("error: boom")).toBeInTheDocument());
  });
});
