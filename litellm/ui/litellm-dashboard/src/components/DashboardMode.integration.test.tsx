import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { useControlPlane } from "@/contexts/ControlPlaneContext";
import { ControlPlanePage } from "./ControlPlanePage";
import BulkCreateUsersButton from "./bulk_create_users_button";
import { ErrorNotice, ResultsTable } from "./ControlPlaneShared";
import MCPServerSelector from "./mcp_server_management/MCPServerSelector";
import MCPToolPermissions from "./mcp_server_management/MCPToolPermissions";
import {
  getGuardrailsList,
  keyListCall,
  modelAvailableCall,
  modelPatchUpdateCall,
  organizationDeleteCall,
  userListCall,
} from "./networking";
import { fetchClient } from "@/lib/http/api";
import { proxyFetch } from "@/lib/http/controlPlaneProxy";
import { afterEach, describe, expect, it, vi } from "vitest";
import DashboardMode from "./DashboardMode";
import { controlPlaneSessionKey } from "@/lib/http/controlPlane";

const configPath = "/litellm/.well-known/litellm-ui-config";
const prefix = "/équipe–ops+qa/v1";
const plane = (id: string, name: string) => ({
  id,
  name,
  source: "config",
  credentials: ["service_token"],
  url: `http://${id}.invalid`,
});

function mockApi({
  scopes,
  bindings,
  rootPath = "",
  disabled = false,
  conflictingRoleSave = () => Response.json({ detail: "Role is managed by configuration." }, { status: 409 }),
}: {
  scopes: string[];
  bindings: string[] | null;
  rootPath?: string;
  disabled?: boolean;
  conflictingRoleSave?: () => Response | Promise<Response>;
}) {
  const responses = new Map<string, unknown>([
    [
      configPath,
      {
        mode: "control_plane_api",
        control_plane_api_prefix: prefix,
        server_root_path: rootPath,
        admin_ui_disabled: disabled,
      },
    ],
    [`${prefix}/auth/login`, { access_token: "control-session", token_type: "bearer", expires_in: 900 }],
    [
      `${prefix}/auth/me`,
      { subject: "operator", email: null, roles: ["test"], scopes, dataplanes: bindings, token_kind: "ui" },
    ],
    [`${prefix}/dataplanes`, { items: [plane("east", "East"), plane("west", "West")] }],
    [`${prefix}/dataplanes/east/health`, { status: "ok" }],
    [`${prefix}/dataplanes/west/health`, { status: "unreachable" }],
    [`${prefix}/dataplanes/east/proxy/key/list`, { keys: [{ key_alias: "East only", token: "east-key-id" }] }],
    [`${prefix}/dataplanes/west/proxy/key/list`, { keys: [{ key_alias: "West only", token: "west-key-id" }] }],
    [
      `${prefix}/dataplanes/east/proxy/organization/list`,
      [{ organization_id: "org-1", organization_alias: "East organization" }],
    ],
    [
      `${prefix}/dataplanes/east/proxy/v1/model/info`,
      {
        data: [
          {
            model_name: "East model",
            model_info: { id: "model-1", db_model: true },
            litellm_params: { model: "openai/existing", headers: { Authorization: "REDACTED" }, rpm: 20 },
          },
        ],
      },
    ],
    [`${prefix}/dataplanes/east/proxy/model/model-1/update`, {}],
    [
      `${prefix}/dataplanes/east/proxy/guardrails/list`,
      { guardrails: [{ guardrail_id: "config-1", guardrail_name: "Configured guardrail" }] },
    ],
    [
      `${prefix}/dataplanes/east/proxy/v2/guardrails/list`,
      {
        guardrails: [
          { guardrail_id: "config-1", guardrail_name: "Configured guardrail", guardrail_definition_location: "config" },
          { guardrail_id: "db-1", guardrail_name: "Saved guardrail", guardrail_definition_location: "db" },
        ],
      },
    ],
    [`${prefix}/roles`, { items: [] }],
    [
      `${prefix}/fleet/models`,
      {
        partial: true,
        unreachable: [{ id: "west", reason: "connection refused" }],
        items: [{ model_name: "east-model", dataplane_id: "east", provider: "openai", mode: "chat" }],
      },
    ],
  ]);
  const fetcher = vi.fn(async (input: string, options?: RequestInit) => {
    if (input !== configPath && !input.startsWith(`${rootPath}/`)) throw new Error(`Unexpected request ${input}`);
    const pathname = input === configPath ? input : input.split("?")[0].slice(rootPath.length);
    if (pathname === `${prefix}/dataplanes/east/proxy/user/list`) {
      const page = Number(new URL(input, "http://localhost").searchParams.get("page") ?? 1);
      const users = {
        users: [{ user_id: `east-user-${page}` }],
        page,
        page_size: 25,
        total_pages: 2,
      };
      return Response.json(users);
    }
    if (pathname === `${prefix}/dataplanes/east/proxy/model/info`) {
      return Response.json({
        data: [
          { model_name: "east-model", model_info: { access_groups: ["east-group"] } },
          { model_name: "east-model" },
          { model_name: "openai/*" },
        ],
      });
    }
    if (pathname === `${prefix}/dataplanes/east/proxy/organization/update`)
      return options?.method === "PATCH" ? Response.json({}) : new Response(null, { status: 405 });
    if (pathname === `${prefix}/dataplanes/east/proxy/organization/delete`)
      return options?.method === "DELETE" ? Response.json([]) : new Response(null, { status: 405 });
    if (pathname === `${prefix}/dataplanes/east/proxy/key/generate`) {
      return Response.json({ detail: "Dataplane is unavailable." }, { status: 503 });
    }
    if (options?.method === "PUT" && pathname === `${prefix}/roles/conflict`) {
      return conflictingRoleSave();
    }
    if (pathname === `${prefix}/audit`) {
      return Response.json({ detail: "Scope no longer granted." }, { status: 403 });
    }
    if (!responses.has(pathname)) throw new Error(`Unexpected request ${pathname}`);
    return Response.json(responses.get(pathname));
  });
  vi.stubGlobal("fetch", fetcher);
  return fetcher;
}

function FixtureMode({ initialSection = "" }: { initialSection?: string }) {
  const [section, setSection] = useState(initialSection);
  return (
    <DashboardMode>
      <WorkspaceFixture section={section} setSection={setSection} />
    </DashboardMode>
  );
}

function WorkspaceFixture({ section, setSection }: { section: string; setSection: (value: string) => void }) {
  const session = useControlPlane();
  const [mcpSelection, setMcpSelection] = useState<{
    servers: string[];
    accessGroups: string[];
    toolsets: string[];
  }>({ servers: ["server-1"], accessGroups: [], toolsets: [] });
  const [toolPermissions, setToolPermissions] = useState<Record<string, string[]>>({ "server-1": ["read"] });
  const keys = useQuery({
    queryKey: ["keys", session?.selected],
    queryFn: () => keyListCall(session!.token, null, null, null, null, null, 1, 25),
    enabled: Boolean(session?.selected && section === "keys"),
  });
  if (!session) return <p>Legacy dashboard</p>;
  return (
    <>
      <h1>Global Control Plane</h1>
      <label>
        Dataplane
        <select value={session.selected ?? ""} onChange={(event) => session.select(event.target.value)}>
          {session.options.map(({ id, label }) => (
            <option key={id} value={id}>
              {label}
            </option>
          ))}
        </select>
      </label>
      <button onClick={() => setSection("keys")}>Virtual Keys</button>
      <button onClick={() => setSection("fleet-models")}>Fleet Models</button>
      <button onClick={() => setSection("dataplanes")}>Dataplanes</button>
      <button onClick={() => setSection("roles")}>Roles</button>
      <button onClick={() => setSection("audit")}>Audit Log</button>
      <button onClick={session.logout}>Sign out</button>
      {section === "bulk-users" && (
        <BulkCreateUsersButton accessToken={session.token} teams={[]} possibleUIRoles={null} />
      )}
      {section === "mcp-grants" && (
        <fieldset>
          <legend>MCP grants</legend>
          <MCPServerSelector accessToken={session.token} value={mcpSelection} onChange={setMcpSelection} />
          <MCPToolPermissions
            accessToken={session.token}
            selectedServers={mcpSelection.servers}
            toolPermissions={toolPermissions}
            onChange={setToolPermissions}
          />
        </fieldset>
      )}
      {section === "keys" ? (
        <>
          <ErrorNotice error={keys.error} />
          <ResultsTable
            columns={["Alias"]}
            rows={keys.data?.keys.map((key: { key_alias: string }) => [key.key_alias]) ?? []}
          />
        </>
      ) : (
        <ControlPlanePage section={section} />
      )}
    </>
  );
}

async function submitLogin() {
  fireEvent.change(await screen.findByLabelText("Username"), { target: { value: "operator" } });
  fireEvent.change(screen.getByLabelText("Password"), { target: { value: "test-password" } });
  fireEvent.click(screen.getByRole("button", { name: "Sign in" }));
  await screen.findByRole("heading", { name: "Global Control Plane" });
}

async function signIn(initialSection = "") {
  render(
    <NuqsTestingAdapter hasMemory>
      <QueryClientProvider client={new QueryClient()}>
        <FixtureMode initialSection={initialSection} />
      </QueryClientProvider>
    </NuqsTestingAdapter>,
  );
  await submitLogin();
}

function overrideResponse(
  fetcher: ReturnType<typeof mockApi>,
  path: string,
  respond: () => Response | Promise<Response>,
) {
  const originalFetch = fetcher.getMockImplementation();
  if (!originalFetch) throw new Error("Expected an API fixture before overriding its response.");
  fetcher.mockImplementation(async (input, options) =>
    input.split("?")[0] === `${prefix}${path}` ? respond() : originalFetch(input, options),
  );
}

afterEach(() => {
  sessionStorage.clear();
  localStorage.clear();
  vi.unstubAllGlobals();
});

describe("control plane dashboard mode", () => {
  it.each([
    { switchDataplanes: false, expectedWrites: ["east", "east"] },
    { switchDataplanes: true, expectedWrites: ["east"] },
  ])(
    "keeps bulk user creation on its starting dataplane (switch=$switchDataplanes)",
    async ({ switchDataplanes, expectedWrites }) => {
      const fetcher = mockApi({ scopes: ["dataplanes:read", "proxy:write"], bindings: null });
      const firstWrite = Promise.withResolvers<Response>();
      const writes: string[] = [];
      overrideResponse(fetcher, "/dataplanes/east/proxy/user/new", () => {
        writes.push("east");
        return writes.length === 1 ? firstWrite.promise : Response.json({ user_id: "second" });
      });
      overrideResponse(fetcher, "/dataplanes/west/proxy/user/new", () => {
        writes.push("west");
        return Response.json({ user_id: "second" });
      });
      await signIn("bulk-users");
      fireEvent.click(screen.getByRole("button", { name: /Bulk Invite Users/ }));
      const csv = new File(
        ["user_email,user_role\nfirst@example.com,internal_user\nsecond@example.com,internal_user"],
        "users.csv",
        { type: "text/csv" },
      );
      fireEvent.change(screen.getByLabelText(/Drag and drop your CSV file here/), { target: { files: [csv] } });
      const [create] = await screen.findAllByRole("button", { name: "Create 2 Users" });
      fireEvent.click(create);
      await waitFor(() => expect(writes).toEqual(["east"]));

      if (switchDataplanes) {
        fireEvent.click(screen.getByRole("button", { name: "Close" }));
        fireEvent.change(await screen.findByRole("combobox", { name: "Dataplane" }), { target: { value: "west" } });
        await waitFor(() => expect(screen.getByRole("combobox", { name: "Dataplane" })).toHaveValue("west"));
      }

      await act(async () => firstWrite.resolve(Response.json({ user_id: "first" })));
      expect(writes).toEqual(expectedWrites);
    },
  );

  it("returns to sign-in when the control plane rejects the session during a workspace request", async () => {
    const fetcher = mockApi({ scopes: ["dataplanes:read", "proxy:read"], bindings: null });
    await signIn();
    await screen.findByRole("option", { name: "East (ok)" });
    overrideResponse(fetcher, "/dataplanes/east/proxy/key/list", () =>
      Response.json({ detail: "Session token expired." }, { status: 401 }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Virtual Keys" }));
    expect(await screen.findByRole("heading", { name: "Control Plane Login" })).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("Your control plane session expired. Sign in again.");
    await waitFor(() => expect(sessionStorage.getItem(controlPlaneSessionKey(prefix))).toBeNull());
    await submitLogin();
  });

  it("keeps the control-plane session when a dataplane rejects its credentials", async () => {
    const fetcher = mockApi({ scopes: ["dataplanes:read", "proxy:read"], bindings: null });
    await signIn();
    await screen.findByRole("option", { name: "East (ok)" });
    overrideResponse(fetcher, "/dataplanes/east/proxy/key/list", () =>
      Response.json(
        { detail: "Dataplane rejected the credential." },
        {
          status: 401,
          headers: { "X-LiteLLM-Dataplane": "east" },
        },
      ),
    );
    fireEvent.click(screen.getByRole("button", { name: "Virtual Keys" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Dataplane rejected the credential.");
    expect(screen.getByRole("heading", { name: "Global Control Plane" })).toBeInTheDocument();
    expect(sessionStorage.getItem(controlPlaneSessionKey(prefix))).toBe("control-session");
  });

  it("lists saved and configured guardrails on the selected dataplane", async () => {
    const fetcher = mockApi({ scopes: ["dataplanes:read", "proxy:read"], bindings: null });
    await signIn();
    await screen.findByRole("option", { name: "East (ok)" });
    const result = await getGuardrailsList("control-session");
    expect(result.guardrails.map((guardrail: { guardrail_name: string }) => guardrail.guardrail_name)).toEqual([
      "Configured guardrail",
      "Saved guardrail",
    ]);
    expect(fetcher).toHaveBeenCalledWith(
      `${prefix}/dataplanes/east/proxy/v2/guardrails/list`,
      expect.objectContaining({ method: "GET" }),
    );
  });

  it.each([
    { field: "model_name", value: "Renamed model", body: { model_name: "Renamed model" } },
    { field: "Provider model", value: "openai/replacement", body: { litellm_params: { model: "openai/replacement" } } },
  ])("edits a stored model's $field without resending listed parameters", async ({ body }) => {
    const fetcher = mockApi({ scopes: ["dataplanes:read", "proxy:read", "proxy:write"], bindings: null });
    await signIn();
    await screen.findByRole("option", { name: "East (ok)" });
    await modelPatchUpdateCall("control-session", body, "model-1");
    expect(fetcher).toHaveBeenCalledWith(
      `${prefix}/dataplanes/east/proxy/model/model-1/update`,
      expect.objectContaining({ method: "PATCH" }),
    );
    const request = fetcher.mock.calls.find(([url]) => url.endsWith("/model/model-1/update"));
    expect(JSON.parse(new TextDecoder().decode(request?.[1]?.body as ArrayBuffer))).toEqual(body);
  });

  it("sends control plane authentication under the discovered server root", async () => {
    const fetcher = mockApi({ scopes: [], bindings: null, rootPath: "/gateway" });
    const otherPlaneKey = controlPlaneSessionKey(`/other${prefix}`);
    sessionStorage.setItem(otherPlaneKey, "other-session");
    await signIn();
    expect(fetcher).toHaveBeenCalledWith(`/gateway${prefix}/auth/login`, expect.objectContaining({ method: "POST" }));
    expect(fetcher).toHaveBeenCalledWith(
      `/gateway${prefix}/auth/me`,
      expect.objectContaining({
        method: "GET",
        headers: { Accept: "application/json", Authorization: "Bearer control-session" },
      }),
    );
    expect(sessionStorage.getItem(otherPlaneKey)).toBe("other-session");
  });

  it("keeps both dashboards and authentication disabled when discovery disables the UI", async () => {
    const fetcher = mockApi({ scopes: [], bindings: null, disabled: true });
    render(
      <QueryClientProvider client={new QueryClient()}>
        <FixtureMode />
      </QueryClientProvider>,
    );
    expect(await screen.findByText("The Admin UI is disabled.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Sign in" })).not.toBeInTheDocument();
    expect(screen.queryByText("Legacy dashboard")).not.toBeInTheDocument();
    expect(fetcher.mock.calls.map(([url]) => url)).toEqual([configPath]);
  });

  it.each([
    {
      action: "Edit organizations",
      confirm: "Save",
      route: "update",
      method: "PATCH",
      body: { organization_alias: "Renamed", organization_id: "org-1" },
    },
    {
      action: "Delete organizations",
      confirm: "Confirm delete",
      route: "delete",
      method: "DELETE",
      body: { organization_ids: ["org-1"] },
    },
  ])("uses the dataplane HTTP method for $action", async ({ route, method, body }) => {
    const fetcher = mockApi({ scopes: ["dataplanes:read", "proxy:read", "proxy:write"], bindings: null });
    await signIn();
    await screen.findByRole("option", { name: "East (ok)" });
    if (route === "update") {
      await fetchClient.PATCH("/organization/update", {
        body: { organization_id: "org-1", organization_alias: "Renamed" },
      });
    } else {
      await organizationDeleteCall("control-session", "org-1");
    }
    expect(fetcher).toHaveBeenCalledWith(
      `${prefix}/dataplanes/east/proxy/organization/${route}`,
      expect.objectContaining({ method }),
    );
    const request = fetcher.mock.calls.find(([url]) => url.endsWith(`/organization/${route}`));
    expect(JSON.parse(new TextDecoder().decode(request?.[1]?.body as ArrayBuffer))).toEqual(body);
  });

  it("loads later users with the dataplane's page and page_size parameters", async () => {
    const fetcher = mockApi({ scopes: ["dataplanes:read", "proxy:read"], bindings: null });
    await signIn();
    await screen.findByRole("option", { name: "East (ok)" });
    const first = await userListCall("control-session", null, 1, 25);
    const next = await userListCall("control-session", null, 2, 25);
    expect(first.users[0]?.user_id).toBe("east-user-1");
    expect(next.users[0]?.user_id).toBe("east-user-2");
    expect(fetcher).toHaveBeenCalledWith(
      expect.stringContaining("/user/list?page=2&page_size=25"),
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("loads model choices through the selected dataplane's model catalog", async () => {
    const fetcher = mockApi({ scopes: ["dataplanes:read", "proxy:read"], bindings: null });
    await signIn();
    await screen.findByRole("option", { name: "East (ok)" });
    const models = await modelAvailableCall("control-session", "", "", false, "east-team");
    expect(models.data.map(({ id }: { id: string }) => id)).toEqual(["east-model", "east-group"]);
    expect(fetcher).toHaveBeenCalledWith(
      expect.stringContaining("/proxy/model/info?teamId=east-team"),
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("authenticates once and switches dataplanes without contacting either directly", async () => {
    const fetcher = mockApi({ scopes: ["dataplanes:read", "proxy:read", "fleet:read"], bindings: null });
    await signIn();
    expect(fetcher).toHaveBeenCalledWith(
      `${prefix}/auth/login`,
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ username: "operator", password: "test-password" }),
        credentials: "omit",
      }),
    );
    expect(fetcher).toHaveBeenCalledWith(
      `${prefix}/auth/me`,
      expect.objectContaining({
        headers: { Accept: "application/json", Authorization: "Bearer control-session" },
      }),
    );
    expect(sessionStorage.getItem(controlPlaneSessionKey(prefix))).toBe("control-session");
    expect(localStorage.length).toBe(0);
    expect(await screen.findByRole("option", { name: "East (ok)" })).toBeInTheDocument();
    expect(await screen.findByRole("option", { name: "West (unreachable)" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Virtual Keys" }));
    expect(await screen.findByRole("cell", { name: "East only" })).toBeInTheDocument();
    fireEvent.change(screen.getByRole("combobox", { name: "Dataplane" }), { target: { value: "west" } });
    expect(await screen.findByRole("cell", { name: "West only" })).toBeInTheDocument();
    expect(screen.queryByRole("cell", { name: "East only" })).not.toBeInTheDocument();
    expect(fetcher).toHaveBeenCalledWith(
      expect.stringContaining(`${prefix}/dataplanes/west/proxy/key/list?`),
      expect.objectContaining({
        headers: expect.any(Headers),
        credentials: "omit",
        redirect: "error",
      }),
    );
    const west = fetcher.mock.calls.find(([url]) => url.includes("/west/proxy/key/list"));
    expect(new Headers(west?.[1]?.headers).get("Authorization")).toBe("Bearer control-session");
    expect(fetcher.mock.calls.filter(([url]) => url === `${prefix}/auth/login`)).toHaveLength(1);
    expect(fetcher.mock.calls.map(([url]) => url).every((url) => url === configPath || url.startsWith(prefix))).toBe(
      true,
    );

    fireEvent.click(screen.getByRole("button", { name: "Fleet Models" }));
    expect(await screen.findByText("Partial results. Totals do not cover every dataplane.")).toBeInTheDocument();
    expect(screen.getByText("Unreachable: west (connection refused)")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Sign out" }));
    expect(await screen.findByRole("heading", { name: "Control Plane Login" })).toBeInTheDocument();
    expect(sessionStorage.getItem(controlPlaneSessionKey(prefix))).toBeNull();
  });

  it("enforces the binding, leaves scoped admin actions visible, and propagates upstream failures", async () => {
    const fetcher = mockApi({
      scopes: ["dataplanes:read", "dataplanes:write", "proxy:read", "proxy:write"],
      bindings: ["east"],
    });
    await signIn();
    expect(await screen.findByRole("option", { name: "East (ok)" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: /West/ })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Dataplanes" }));
    expect(screen.getByRole("button", { name: "Create dataplane" })).toHaveAttribute("aria-disabled", "true");
    expect(await screen.findByRole("button", { name: "Edit" })).toHaveAttribute("aria-disabled", "true");
    await expect(proxyFetch("/key/generate", { method: "POST" })).resolves.toMatchObject({ status: 503 });
    expect(fetcher).toHaveBeenCalledWith(
      `${prefix}/dataplanes/east/proxy/key/generate`,
      expect.objectContaining({ method: "POST" }),
    );
    expect(fetcher.mock.calls.some(([url]) => url.includes("/west/"))).toBe(false);
  });

  it("updates the picker after creating and deleting a runtime dataplane", async () => {
    const fetcher = mockApi({ scopes: ["dataplanes:read", "dataplanes:write", "proxy:read"], bindings: null });
    const north = { ...plane("north", "North"), source: "runtime" };
    let items = [plane("east", "East"), plane("west", "West")];
    overrideResponse(fetcher, "/dataplanes", () => Response.json({ items }));
    overrideResponse(fetcher, "/dataplanes/north/health", () => Response.json({ status: "ok" }));
    const apiFetch = fetcher.getMockImplementation()!;
    fetcher.mockImplementation(async (input, options) => {
      if (input === `${prefix}/dataplanes/north` && options?.method === "PUT") {
        items = [...items, north];
        return Response.json(north);
      }
      if (input === `${prefix}/dataplanes/north` && options?.method === "DELETE") {
        items = items.filter(({ id }) => id !== "north");
        return new Response(null, { status: 204 });
      }
      return apiFetch(input, options);
    });

    await signIn("dataplanes");
    expect(await screen.findByRole("option", { name: "East (ok)" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Create dataplane" }));
    fireEvent.change(screen.getByLabelText("Dataplane ID"), { target: { value: "north" } });
    fireEvent.change(screen.getByLabelText("URL"), { target: { value: north.url } });
    fireEvent.change(screen.getByLabelText("Service token value"), { target: { value: "north-test-token" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    expect(await screen.findByRole("row", { name: /north North/ })).toBeInTheDocument();
    expect(await screen.findByRole("option", { name: "North (ok)" })).toBeInTheDocument();
    fireEvent.change(screen.getByRole("combobox", { name: "Dataplane" }), { target: { value: "north" } });
    expect(await screen.findByRole("combobox", { name: "Dataplane" })).toHaveValue("north");

    const selectedRow = await screen.findByRole("row", { name: /north North/ });
    fireEvent.click(within(selectedRow).getByRole("button", { name: "Delete" }));
    fireEvent.click(screen.getByRole("button", { name: "Confirm delete" }));

    await waitFor(() => expect(screen.queryByRole("option", { name: /North/ })).not.toBeInTheDocument());
    expect(await screen.findByRole("combobox", { name: "Dataplane" })).toHaveValue("east");
  });

  it("disables MCP grant changes without requesting unsupported tool discovery", async () => {
    const fetcher = mockApi({ scopes: ["dataplanes:read", "proxy:read", "proxy:write"], bindings: null });
    const proxy = "/dataplanes/east/proxy";
    overrideResponse(fetcher, `${proxy}/v1/mcp/server`, () =>
      Response.json([{ server_id: "server-1", server_name: "Test MCP Server" }]),
    );
    overrideResponse(fetcher, `${proxy}/v1/mcp/access_groups`, () => Response.json({ access_groups: [] }));
    overrideResponse(fetcher, `${proxy}/v1/mcp/toolset`, () => Response.json([]));
    overrideResponse(fetcher, `${proxy}/mcp-rest/tools/list`, () => Response.json({}, { status: 403 }));
    await signIn("mcp-grants");

    expect(await screen.findByText("Test MCP Server (server-1)")).toBeInTheDocument();
    const selector = within(screen.getByRole("group", { name: "MCP grants" })).getByRole("combobox");
    expect(selector).toBeDisabled();
    expect(selector).toHaveAccessibleDescription("This action isn't available through the control plane yet.");
    expect(fetcher.mock.calls.some(([url]) => url.includes("/mcp-rest/tools/list"))).toBe(false);
  });

  it("searches deployed fleet models and refreshes the fleet inventory", async () => {
    const fetcher = mockApi({ scopes: ["dataplanes:read", "proxy:read", "fleet:read"], bindings: null });
    await signIn();
    fireEvent.click(screen.getByRole("button", { name: "Fleet Models" }));
    expect(await screen.findByRole("cell", { name: "east-model" })).toBeInTheDocument();
    expect(screen.getByText("View deployed models across the dataplanes you can access.")).toBeInTheDocument();
    const search = screen.getByPlaceholderText("Search");
    fireEvent.change(search, { target: { value: "not-a-model" } });
    expect(screen.queryByRole("cell", { name: "east-model" })).not.toBeInTheDocument();
    fireEvent.change(search, { target: { value: "openai" } });
    expect(screen.getByRole("cell", { name: "east-model" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Refresh" }));
    await waitFor(() => expect(fetcher.mock.calls.filter(([url]) => url.endsWith("/fleet/models"))).toHaveLength(2));
  });

  it("leaves legacy worker mode on the existing dashboard", async () => {
    const fetcher = vi.fn(async () => Response.json({ is_control_plane: true, workers: [{ id: "west" }] }));
    vi.stubGlobal("fetch", fetcher);
    render(<DashboardMode>Legacy dashboard</DashboardMode>);
    expect(await screen.findByText("Legacy dashboard")).toBeInTheDocument();
    expect(fetcher.mock.calls.filter(([url]) => typeof url === "string" && url.includes("/auth/"))).toHaveLength(0);
  });

  it("locks a role editor during save and preserves its inputs when the API rejects it", async () => {
    const save = Promise.withResolvers<Response>();
    mockApi({ scopes: ["roles:write", "audit:read"], bindings: null, conflictingRoleSave: () => save.promise });
    await signIn();
    fireEvent.click(screen.getByRole("button", { name: "Roles" }));
    expect(await screen.findByText("No rows match your search or filters.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Create role" }));
    const name = screen.getByLabelText("Role name");
    fireEvent.change(name, { target: { value: "conflict" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(name).toBeDisabled();
    save.resolve(Response.json({ detail: "Role is managed by configuration." }, { status: 409 }));
    expect(await screen.findByRole("alert")).toHaveTextContent("409: Role is managed by configuration.");
    expect(name).toBeEnabled();
    expect(name).toHaveValue("conflict");

    fireEvent.click(screen.getByRole("button", { name: "Audit Log" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("403: Scope no longer granted.");
  });
});
