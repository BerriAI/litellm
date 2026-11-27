import { act, renderHook, waitFor } from "@testing-library/react";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { MCPServer } from "@/components/mcp_tools/types";
import { TOOLS_OAUTH_UI_STATE_KEY } from "@/hooks/mcpOAuthUtils";
import { setSecureItem } from "@/utils/secureStorage";
import {
  type McpPageAccess,
  resolveVisibleTab,
  useMcpServerTab,
  useMcpServersUrlState,
  visiblePageTabs,
  visibleServerTabs,
} from "./useMcpServersUrlState";

const EDIT_OAUTH_UI_STATE_KEY = "litellm-mcp-oauth-edit-state";

vi.mock("./mcp_server_edit", () => ({ EDIT_OAUTH_UI_STATE_KEY: "litellm-mcp-oauth-edit-state" }));

const ADMIN: McpPageAccess = { isAdmin: true, canViewLiveConnections: true };
const ROLE_LOADING: McpPageAccess = { isAdmin: false, canViewLiveConnections: false };

const server = (server_id: string): MCPServer => ({
  server_id,
  created_at: "",
  updated_at: "",
  created_by: "user",
  updated_by: "user",
});

const SERVERS = [server("alpha"), server("bravo")];

const wrapperFor = (searchParams: string, onUrlUpdate: OnUrlUpdateFunction) =>
  function UrlWrapper({ children }: { children: ReactNode }) {
    return (
      <NuqsTestingAdapter
        searchParams={searchParams}
        onUrlUpdate={onUrlUpdate}
        hasMemory
        resetUrlUpdateQueueOnMount={false}
      >
        {children}
      </NuqsTestingAdapter>
    );
  };

const renderPageState = (
  searchParams: string,
  initial: { access: McpPageAccess; servers?: readonly MCPServer[]; loaded?: boolean } = { access: ADMIN },
) => {
  const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
  const hook = renderHook(({ access, servers, loaded }) => useMcpServersUrlState(access, servers, loaded), {
    wrapper: wrapperFor(searchParams, onUrlUpdate),
    initialProps: { access: initial.access, servers: initial.servers ?? SERVERS, loaded: initial.loaded ?? true },
  });
  return { ...hook, onUrlUpdate };
};

const flushUrl = () => act(() => new Promise((resolve) => setTimeout(resolve, 100)));

const everyWriteKeeps = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>, key: string) =>
  onUrlUpdate.mock.calls.every(([update]) => update.searchParams.has(key));

describe("resolveVisibleTab", () => {
  it("returns the requested tab when the role can see it and the fallback when it cannot", () => {
    expect(resolveVisibleTab("tool-search", visiblePageTabs(ADMIN), "servers")).toBe("tool-search");
    expect(resolveVisibleTab("tool-search", visiblePageTabs(ROLE_LOADING), "servers")).toBe("servers");
  });
});

describe("visiblePageTabs", () => {
  it.each([
    [
      ADMIN,
      [
        "servers",
        "toolsets",
        "connect",
        "semantic-filter",
        "tool-search",
        "network-settings",
        "submitted",
        "connections",
      ],
    ],
    [
      { isAdmin: true, canViewLiveConnections: false },
      ["servers", "toolsets", "connect", "semantic-filter", "tool-search", "network-settings", "submitted"],
    ],
    [ROLE_LOADING, ["servers", "toolsets", "connect"]],
  ])("gives %o the tabs %o", (access, tabs) => {
    expect(visiblePageTabs(access)).toEqual(tabs);
  });
});

describe("visibleServerTabs", () => {
  it.each([
    [{ isProxyAdmin: true, canViewUserCredentials: true }, ["overview", "tools", "settings", "user-credentials"]],
    [{ isProxyAdmin: true, canViewUserCredentials: false }, ["overview", "tools", "settings"]],
    [{ isProxyAdmin: false, canViewUserCredentials: false }, ["overview", "tools"]],
  ])("gives %o the server tabs %o", (access, tabs) => {
    expect(visibleServerTabs(access)).toEqual(tabs);
  });
});

describe("useMcpServersUrlState page tab", () => {
  beforeEach(() => window.sessionStorage.clear());

  it("keeps an admin tab link while the role loads and opens it once the admin role arrives", async () => {
    const { result, rerender, onUrlUpdate } = renderPageState("?tab=semantic-filter", { access: ROLE_LOADING });

    expect(result.current.tab).toBe("servers");
    await flushUrl();
    expect(everyWriteKeeps(onUrlUpdate, "tab")).toBe(true);

    rerender({ access: ADMIN, servers: SERVERS, loaded: true });

    expect(result.current.tab).toBe("semantic-filter");
  });

  it.each(["bogus", "SEMANTIC-FILTER"])("drops the unknown tab %s from the URL", async (value) => {
    const { result, onUrlUpdate } = renderPageState(`?tab=${value}`);

    expect(result.current.tab).toBe("servers");
    await flushUrl();
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.has("tab")).toBe(false);
  });

  it("uses the first tab when the URL repeats the key", () => {
    const { result } = renderPageState("?tab=toolsets&tab=connect");

    expect(result.current.tab).toBe("toolsets");
  });

  it("writes nothing to a bare URL", async () => {
    const { result, onUrlUpdate } = renderPageState("");

    await flushUrl();
    expect(onUrlUpdate).not.toHaveBeenCalled();

    act(() => result.current.setTab("toolsets"));
    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalledTimes(1));
    expect(onUrlUpdate.mock.calls[0][0].searchParams.toString()).toBe("tab=toolsets");
  });
});

describe("useMcpServersUrlState server detail", () => {
  beforeEach(() => window.sessionStorage.clear());

  it("pushes history when a server opens and when it closes", async () => {
    const { result, onUrlUpdate } = renderPageState("?team=t1");

    act(() => result.current.openServer("bravo"));
    await flushUrl();
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("server")).toBe("bravo");
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].options.history).toBe("push");
    expect(result.current.selectedServer?.server_id).toBe("bravo");

    act(() => result.current.closeServer());
    await flushUrl();
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.has("server")).toBe(false);
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("team")).toBe("t1");
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].options.history).toBe("push");
  });

  it("captures fill_env_vars once and drops only that key", async () => {
    const { result, onUrlUpdate } = renderPageState("?fill_env_vars=alpha&team=t1");

    expect(result.current.envVarsDeepLinkId).toBe("alpha");
    await flushUrl();
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.has("fill_env_vars")).toBe(false);
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("team")).toBe("t1");
    expect(result.current.envVarsDeepLinkId).toBe("alpha");
  });
});

describe("useMcpServersUrlState OAuth restore", () => {
  beforeEach(() => window.sessionStorage.clear());

  it("returns to the tools tab of the server stored by the tools OAuth flow", async () => {
    setSecureItem(TOOLS_OAUTH_UI_STATE_KEY, JSON.stringify({ serverId: "alpha" }));

    const { result, onUrlUpdate } = renderPageState("");

    await flushUrl();
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("server")).toBe("alpha");
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("server_tab")).toBe("tools");
    expect(result.current.editServer).toBe(false);
    expect(window.sessionStorage.getItem(TOOLS_OAUTH_UI_STATE_KEY)).toBeNull();
  });

  it("returns to the editable settings of the server stored by the edit OAuth flow", async () => {
    setSecureItem(EDIT_OAUTH_UI_STATE_KEY, JSON.stringify({ serverId: "bravo" }));

    const { result, onUrlUpdate } = renderPageState("");

    await flushUrl();
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("server")).toBe("bravo");
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("server_tab")).toBe("settings");
    expect(result.current.editServer).toBe(true);
  });

  it("drops the stored edit state once the servers load without that server", async () => {
    setSecureItem(EDIT_OAUTH_UI_STATE_KEY, JSON.stringify({ serverId: "deleted" }));

    const { result, rerender } = renderPageState("", { access: ADMIN, servers: [], loaded: false });

    await flushUrl();
    expect(window.sessionStorage.getItem(EDIT_OAUTH_UI_STATE_KEY)).not.toBeNull();

    rerender({ access: ADMIN, servers: SERVERS, loaded: true });

    expect(window.sessionStorage.getItem(EDIT_OAUTH_UI_STATE_KEY)).toBeNull();
    expect(result.current.selectedServer).toBeNull();
  });
});

describe("useMcpServerTab", () => {
  const renderServerTab = (searchParams: string, access = { isProxyAdmin: true, canViewUserCredentials: true }) => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    const hook = renderHook((props) => useMcpServerTab(props), {
      wrapper: wrapperFor(searchParams, onUrlUpdate),
      initialProps: access,
    });
    return { ...hook, onUrlUpdate };
  };

  it("keeps a settings link for a role that cannot see it and opens it once the role can", async () => {
    const { result, rerender, onUrlUpdate } = renderServerTab("?server=alpha&server_tab=settings", {
      isProxyAdmin: false,
      canViewUserCredentials: false,
    });

    expect(result.current[0]).toBe("overview");
    await flushUrl();
    expect(everyWriteKeeps(onUrlUpdate, "server_tab")).toBe(true);

    rerender({ isProxyAdmin: true, canViewUserCredentials: false });

    expect(result.current[0]).toBe("settings");
  });

  it("opens user credentials only for a role that can view them", () => {
    expect(renderServerTab("?server_tab=user-credentials").result.current[0]).toBe("user-credentials");
    expect(
      renderServerTab("?server_tab=user-credentials", { isProxyAdmin: true, canViewUserCredentials: false }).result
        .current[0],
    ).toBe("overview");
  });
});
