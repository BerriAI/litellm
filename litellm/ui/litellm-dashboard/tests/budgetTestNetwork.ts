import { QueryClient } from "@tanstack/react-query";
import { vi } from "vitest";

export function budgetTestNetwork(path: "/budget/new" | "/budget/update") {
  const claims = btoa(JSON.stringify({ key: "budget-test-token", exp: Math.floor(Date.now() / 1000) + 3600 }));
  document.cookie = `token=e30.${claims}.test; path=/`;
  const write = vi.fn<(body: unknown) => void>();
  const json = (body: unknown) =>
    new Response(JSON.stringify(body), {
      headers: { "Content-Type": "application/json" },
    });
  vi.stubGlobal(
    "fetch",
    vi.fn<typeof fetch>(async (input, init) => {
      const url = new URL(input instanceof Request ? input.url : String(input), "http://localhost");
      if (url.pathname.endsWith("litellm-runtime-config.json")) return json({});
      if (url.pathname.endsWith("litellm-ui-config"))
        return json({ server_root_path: "", proxy_base_url: "http://localhost" });
      if (url.pathname === "/management/v1/users/me") return json({ user_id: "admin", user_role: "proxy_admin" });
      const method = init?.method ?? (input instanceof Request ? input.method : "GET");
      if (url.pathname !== path || method !== "POST") throw new Error(`Unexpected request: ${method} ${url.pathname}`);
      const requestBody = input instanceof Request ? await input.text() : "";
      const raw = typeof init?.body === "string" ? init.body : requestBody;
      const body: unknown = JSON.parse(raw);
      write(body);
      return json({});
    }),
  );
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 }, mutations: { retry: false } },
  });
  return { client, write };
}
