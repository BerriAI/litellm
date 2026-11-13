import { QueryClient } from "@tanstack/react-query";
import { vi } from "vitest";

export const jsonResponse = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

export interface BudgetRequest {
  path: string;
  method: string;
  body: unknown;
}

export function budgetTestNetwork(handler: (request: BudgetRequest) => Response | Promise<Response>) {
  const claims = btoa(JSON.stringify({ key: "budget-test-token", exp: Math.floor(Date.now() / 1000) + 3600 }));
  document.cookie = `token=e30.${claims}.test; path=/`;
  const requests: BudgetRequest[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn<typeof fetch>(async (input, init) => {
      const url = input instanceof Request ? input.url : String(input);
      const path = new URL(url, "http://localhost").pathname;
      if (path.endsWith("litellm-runtime-config.json")) return jsonResponse({});
      if (path.endsWith("litellm-ui-config"))
        return jsonResponse({ server_root_path: "", proxy_base_url: "http://localhost" });
      if (path === "/management/v1/users/me") return jsonResponse({ user_id: "admin", user_role: "proxy_admin" });
      const method = init?.method ?? (input instanceof Request ? input.method : "GET");
      const requestBody = input instanceof Request ? await input.text() : "";
      const raw = typeof init?.body === "string" ? init.body : requestBody;
      const body: unknown = raw ? JSON.parse(raw) : undefined;
      const request = { path, method, body };
      requests.push(request);
      return handler(request);
    }),
  );
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 }, mutations: { retry: false } },
  });
  return { requests, client };
}
