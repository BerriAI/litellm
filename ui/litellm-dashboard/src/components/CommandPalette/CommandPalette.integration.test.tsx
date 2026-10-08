import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import type { KeyResponse } from "@/components/key_team_helpers/key_list";
import { writeStorage } from "@/lib/storage";
import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";
import { keyDetailHref } from "@/utils/entityLinks";
import { uiHref } from "@/utils/uiHref";
import { renderWithProviders, testQueryClient } from "../../../tests/test-utils";
import { COMMAND_PALETTE_HINT_KEY } from "./CommandPaletteHint";
import { CommandPaletteProvider } from "./CommandPaletteProvider";
import { CommandPaletteTrigger } from "./CommandPaletteTrigger";

const mocks = vi.hoisted(() => ({
  pathname: "/api-keys",
  push: vi.fn<(href: string) => void>(),
  useKeys: vi.fn(),
  teamAlias: "platform-team",
  isPlaceholderData: false,
}));

vi.mock("next/navigation", () => ({
  usePathname: () => mocks.pathname,
  useRouter: () => ({ push: mocks.push }),
}));

vi.mock("@/app/(dashboard)/hooks/keys/useKeys", () => ({
  useKeys: mocks.useKeys,
}));
vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useTeams: () => ({
    data: [{ team_id: "team-1", team_alias: mocks.teamAlias, members_with_roles: [] }],
  }),
}));
vi.mock("@/app/(dashboard)/hooks/useIsOrgAdmin", () => ({
  default: () => false,
}));
vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({
  useUISettings: () => ({ data: { values: {} } }),
}));

const HIGH_VOLUME_KEY = keyFixture("key-prod-0", "high-volume-prod", "sk-...4zoA", "team-1");
const PROD_KEY = keyFixture("key-prod-1", "prod-backend", "sk-...AHeA", "team-1");
const STAGING_KEY = keyFixture("key-staging-1", "staging-agent", "sk-...stgB", "unknown-team");
const KEY_FIXTURES = [HIGH_VOLUME_KEY, PROD_KEY, STAGING_KEY];
const UI_SETTINGS_RESPONSE = {
  server_root_path: "",
  proxy_base_url: null,
  admin_ui_disabled: false,
  auto_redirect_to_sso: false,
  sso_configured: false,
  is_control_plane: false,
  workers: [],
};
const PROD_KEY_LIST_OPTIONS = { search: "prod", sortBy: "created_at", sortOrder: "desc", expand: "user" };
const cachedKeyResults = new Map<string, KeyResponse[]>();

function keyFixture(token: string, alias: string, keyName: string, teamId: string | null = null): KeyResponse {
  return {
    token,
    token_id: token,
    key_name: keyName,
    key_alias: alias,
    spend: 2.5,
    team_id: teamId,
    team_alias: "",
  } as KeyResponse;
}

function sessionCookie() {
  const encode = (value: object) =>
    btoa(JSON.stringify(value)).replaceAll("=", "").replaceAll("+", "-").replaceAll("/", "_");
  const claims = {
    key: "sk-session-test",
    user_id: "test-admin",
    user_role: "proxy_admin",
    premium_user: true,
    auth_header_name: "X-Gateway-Session",
    exp: Date.now() / 1000 + 3600,
  };
  document.cookie = `token=${encode({ alg: "none" })}.${encode(claims)}.test; Path=/`;
}

function renderPalette() {
  return renderWithProviders(
    <CommandPaletteProvider>
      <CommandPaletteTrigger />
    </CommandPaletteProvider>,
  );
}

beforeEach(() => {
  testQueryClient.clear();
  window.localStorage.clear();
  cachedKeyResults.clear();
  vi.clearAllMocks();
  mocks.pathname = "/api-keys";
  mocks.teamAlias = "platform-team";
  mocks.isPlaceholderData = false;
  sessionCookie();
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) =>
      String(input).includes("/v2/team/list")
        ? Response.json({
            teams: [{ team_id: "team-1", team_alias: "platform-team" }],
            total_pages: 1,
          })
        : Response.json(UI_SETTINGS_RESPONSE),
    ),
  );
  vi.mocked(useKeys).mockImplementation((_page, _pageSize, options) => {
    const query = options.search?.toLowerCase() ?? "";
    const keys =
      cachedKeyResults.get(query) ??
      KEY_FIXTURES.filter(
        (key) => !query || key.key_alias.toLowerCase().includes(query) || key.key_name.toLowerCase().includes(query),
      );
    cachedKeyResults.set(query, keys);
    return {
      data: { keys, total_count: keys.length, current_page: 1, total_pages: 1 },
      isFetching: false,
      isError: false,
      isPlaceholderData: mocks.isPlaceholderData,
    } as ReturnType<typeof useKeys>;
  });
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
  testQueryClient.clear();
  vi.unstubAllGlobals();
  document.cookie = "token=; Max-Age=0; Path=/";
});

describe("CommandPalette integration", () => {
  it("shows the discovery hint on the keys route", async () => {
    renderPalette();

    expect(await screen.findByText("to search keys")).toBeVisible();
  });

  it("opens the palette from the hint and persists that it was seen", async () => {
    const user = userEvent.setup();
    renderPalette();

    await user.click(screen.getByRole("button", { name: /to search keys/ }));

    expect(await screen.findByRole("combobox", { name: "Search" })).toBeVisible();
    expect(screen.queryByText("to search keys")).not.toBeInTheDocument();
    expect(window.localStorage.getItem(COMMAND_PALETTE_HINT_KEY.name)).toBe("true");
  });

  it("hides and persists the discovery hint when Ctrl+K opens the palette", async () => {
    renderPalette();

    fireEvent.keyDown(document, { key: "k", ctrlKey: true });

    expect(await screen.findByRole("combobox", { name: "Search" })).toBeVisible();
    expect(screen.queryByText("to search keys")).not.toBeInTheDocument();
    expect(window.localStorage.getItem(COMMAND_PALETTE_HINT_KEY.name)).toBe("true");
  });

  it("dismisses the discovery hint without opening the palette", async () => {
    const user = userEvent.setup();
    renderPalette();

    await user.click(screen.getByRole("button", { name: "Dismiss search hint" }));

    expect(screen.queryByRole("dialog", { name: "Command palette" })).not.toBeInTheDocument();
    expect(screen.queryByText("to search keys")).not.toBeInTheDocument();
    expect(window.localStorage.getItem(COMMAND_PALETTE_HINT_KEY.name)).toBe("true");
  });

  it("hides the discovery hint in memory when storage cannot persist its dismissal", async () => {
    const user = userEvent.setup();
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("Storage unavailable");
    });
    renderPalette();

    await user.click(await screen.findByRole("button", { name: "Dismiss search hint" }));

    expect(screen.queryByText("to search keys")).not.toBeInTheDocument();
    expect(window.localStorage.getItem(COMMAND_PALETTE_HINT_KEY.name)).toBeNull();
  });

  it("keeps the discovery hint hidden after closing the palette when storage is unavailable", async () => {
    const user = userEvent.setup();
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("Storage unavailable");
    });
    renderPalette();

    fireEvent.keyDown(document, { key: "k", ctrlKey: true });
    const input = await screen.findByRole("combobox", { name: "Search" });
    expect(screen.queryByText("to search keys")).not.toBeInTheDocument();

    await user.keyboard("{Escape}");

    expect(screen.queryByText("to search keys")).not.toBeInTheDocument();
    expect(window.localStorage.getItem(COMMAND_PALETTE_HINT_KEY.name)).toBeNull();
  });

  it("does not show the discovery hint when it has already been seen", () => {
    writeStorage(COMMAND_PALETTE_HINT_KEY, true);
    renderPalette();

    expect(screen.queryByText("to search keys")).not.toBeInTheDocument();
  });

  it("shows global-search copy outside the keys routes", async () => {
    mocks.pathname = "/logs";
    renderPalette();

    expect(await screen.findByText("to search or jump to a page")).toBeVisible();
  });

  it("opens and focuses with Ctrl+K, rejects extra modifiers, and toggles closed", async () => {
    renderPalette();
    fireEvent.keyDown(document, { key: "k", ctrlKey: true, shiftKey: true });
    fireEvent.keyDown(document, { key: "k", ctrlKey: true, altKey: true });
    expect(screen.queryByRole("dialog", { name: "Command palette" })).not.toBeInTheDocument();

    fireEvent.keyDown(document, { key: "k", ctrlKey: true });
    const input = await screen.findByRole("combobox", { name: "Search" });
    expect(input).toHaveFocus();
    fireEvent.change(input, { target: { value: "stale" } });

    fireEvent.keyDown(document, { key: "k", ctrlKey: true });
    expect(screen.queryByRole("dialog", { name: "Command palette" })).not.toBeInTheDocument();

    fireEvent.keyDown(document, { key: "k", ctrlKey: true });
    const reopenedInput = await screen.findByRole("combobox", { name: "Search" });
    expect(reopenedInput).toHaveValue("");
  });

  it("searches keys, shows key details, and opens the selected key", async () => {
    const user = userEvent.setup();
    renderPalette();

    fireEvent.keyDown(document, { key: "k", ctrlKey: true });
    const input = await screen.findByRole("combobox", { name: "Search" });
    expect(input).toHaveAttribute("placeholder", "Search keys by alias or ID…");
    expect(screen.getAllByText("Virtual Keys")).toHaveLength(2);

    await user.type(input, "prod");
    await waitFor(() => {
      expect(vi.mocked(useKeys)).toHaveBeenLastCalledWith(1, 8, expect.objectContaining(PROD_KEY_LIST_OPTIONS), {
        enabled: true,
      });
    });
    const firstKeyOption = await screen.findByRole("option", { name: /high-volume-prod/ });
    const keyOption = await screen.findByRole("option", { name: /prod-backend/ });
    expect(firstKeyOption).toHaveAttribute("aria-selected", "true");
    expect(keyOption).toHaveTextContent("platform-team");
    expect(keyOption).toHaveTextContent("$2.50");
    expect(screen.queryByRole("option", { name: /staging-agent/ })).not.toBeInTheDocument();

    fireEvent.keyDown(input, { key: "ArrowDown" });
    expect(keyOption).toHaveAttribute("aria-selected", "true");
    fireEvent.keyDown(input, { key: "Enter" });
    expect(mocks.push).toHaveBeenCalledWith(keyDetailHref(PROD_KEY.token));
    expect(screen.queryByRole("dialog", { name: "Command palette" })).not.toBeInTheDocument();
  });

  it("opens the first matching key when Enter is pressed without navigating", async () => {
    const user = userEvent.setup();
    renderPalette();

    fireEvent.keyDown(document, { key: "k", ctrlKey: true });
    const input = await screen.findByRole("combobox", { name: "Search" });
    await user.type(input, "prod");
    const firstKeyOption = await screen.findByRole("option", { name: /high-volume-prod/ });
    expect(firstKeyOption).toHaveAttribute("aria-selected", "true");

    fireEvent.keyDown(input, { key: "Enter" });
    expect(mocks.push).toHaveBeenCalledWith(keyDetailHref(HIGH_VOLUME_KEY.token));
  });

  it("does not activate stale key results while the new query is debouncing", () => {
    vi.useFakeTimers();
    renderPalette();

    fireEvent.keyDown(document, { key: "k", ctrlKey: true });
    const input = screen.getByRole("combobox", { name: "Search" });
    fireEvent.change(input, { target: { value: "prod" } });
    vi.advanceTimersByTime(DEBOUNCE_WAIT_MS / 2);

    expect(screen.getByText("Searching…")).toBeVisible();
    expect(screen.queryByRole("option", { name: /high-volume-prod/ })).not.toBeInTheDocument();
    expect(screen.getByRole("option", { name: /Filter the keys table/ })).toBeVisible();

    fireEvent.keyDown(input, { key: "Enter" });

    expect(mocks.push).toHaveBeenCalledWith(`${uiHref("api-keys")}?key_search=prod`);
    expect(mocks.push).not.toHaveBeenCalledWith(keyDetailHref(HIGH_VOLUME_KEY.token));
    expect(mocks.push).not.toHaveBeenCalledWith(keyDetailHref(PROD_KEY.token));
  });

  it("does not show placeholder key rows after the query debounce completes", () => {
    vi.useFakeTimers();
    mocks.isPlaceholderData = true;
    renderPalette();

    fireEvent.keyDown(document, { key: "k", ctrlKey: true });
    const input = screen.getByRole("combobox", { name: "Search" });
    fireEvent.change(input, { target: { value: "prod" } });
    act(() => {
      vi.advanceTimersByTime(DEBOUNCE_WAIT_MS);
    });

    expect(screen.getByText("Searching…")).toBeVisible();
    expect(screen.queryByRole("option", { name: /high-volume-prod/ })).not.toBeInTheDocument();
  });

  it("keeps the same key selected when its subtitle changes", async () => {
    const { rerender } = renderPalette();

    fireEvent.keyDown(document, { key: "k", ctrlKey: true });
    const input = await screen.findByRole("combobox", { name: "Search" });
    const secondKeyOption = await screen.findByRole("option", { name: /prod-backend/ });

    fireEvent.keyDown(input, { key: "ArrowDown" });
    expect(secondKeyOption).toHaveAttribute("aria-selected", "true");

    mocks.teamAlias = "renamed-platform-team";
    rerender(
      <CommandPaletteProvider>
        <CommandPaletteTrigger />
      </CommandPaletteProvider>,
    );

    const updatedSecondKeyOption = await screen.findByRole("option", { name: /prod-backend/ });
    expect(updatedSecondKeyOption).toHaveTextContent("renamed-platform-team");
    expect(updatedSecondKeyOption).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("option", { name: /high-volume-prod/ })).toHaveAttribute("aria-selected", "false");
  });

  it("omits an unknown team ID from the key subtitle", async () => {
    const user = userEvent.setup();
    renderPalette();

    fireEvent.keyDown(document, { key: "k", ctrlKey: true });
    const input = await screen.findByRole("combobox", { name: "Search" });
    await user.type(input, "staging");
    const stagingOption = await screen.findByRole("option", { name: /staging-agent/ });
    expect(stagingOption).not.toHaveTextContent("unknown-team");
  });

  it("keeps only non-empty groups and the filter action when no keys match", async () => {
    const user = userEvent.setup();
    renderPalette();

    fireEvent.keyDown(document, { key: "k", ctrlKey: true });
    const input = await screen.findByRole("combobox", { name: "Search" });
    await user.type(input, "no-match");

    expect(await screen.findByText("No keys match “no-match”")).toBeVisible();
    expect(screen.getByRole("group", { name: "Keys" })).toHaveTextContent("No keys match “no-match”");
    expect(screen.getByRole("option", { name: /Filter the keys table/ })).toBeVisible();
    expect(screen.queryByRole("group", { name: "Pages" })).not.toBeInTheDocument();
  });

  it("shows one global no-results message when the query has no matches", async () => {
    const user = userEvent.setup();
    mocks.pathname = "/teams";
    renderPalette();

    fireEvent.keyDown(document, { key: "k", ctrlKey: true });
    const input = await screen.findByRole("combobox", { name: "Search" });
    await user.type(input, "no-match");

    expect(await screen.findByText("No results for “no-match”")).toBeVisible();
    expect(screen.queryByRole("group")).not.toBeInTheDocument();
    expect(screen.queryByRole("option", { name: /Search virtual keys/ })).not.toBeInTheDocument();
  });

  it("opens the virtual-key table with the current search applied", async () => {
    const user = userEvent.setup();
    renderPalette();

    fireEvent.keyDown(document, { key: "k", ctrlKey: true });
    const input = await screen.findByRole("combobox", { name: "Search" });
    await user.type(input, "prod");
    await waitFor(() => {
      expect(vi.mocked(useKeys)).toHaveBeenLastCalledWith(1, 8, expect.objectContaining({ search: "prod" }), {
        enabled: true,
      });
    });

    fireEvent.keyDown(input, { key: "ArrowDown" });
    fireEvent.keyDown(input, { key: "ArrowDown" });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(mocks.push).toHaveBeenCalledWith(`${uiHref("api-keys")}?key_search=prod`);
  });

  it("opens a matching global page from the teams route", async () => {
    const user = userEvent.setup();
    mocks.pathname = "/teams";
    renderPalette();

    fireEvent.keyDown(document, { key: "k", ctrlKey: true });
    const input = await screen.findByRole("combobox", { name: "Search" });
    expect(input).toHaveAttribute("placeholder", "Search pages and actions…");
    await user.type(input, "logs");

    const logsOption = await screen.findByRole("option", { name: /^Logs/ });
    expect(logsOption).toBeVisible();
    fireEvent.keyDown(input, { key: "Enter" });
    expect(mocks.push).toHaveBeenCalledWith(uiHref("logs"));
  });

  it("clears the global query when switching to key search", async () => {
    const user = userEvent.setup();
    mocks.pathname = "/teams";
    renderPalette();

    fireEvent.keyDown(document, { key: "k", ctrlKey: true });
    const input = await screen.findByRole("combobox", { name: "Search" });
    await user.type(input, "keys");

    expect(await screen.findByRole("option", { name: /Search virtual keys/ })).toBeVisible();
    fireEvent.keyDown(input, { key: "Enter" });

    const keysInput = await screen.findByRole("combobox", { name: "Search" });
    expect(keysInput).toHaveValue("");
    expect(await screen.findByText("Recent keys")).toBeVisible();
  });

  it("switches from an empty keys scope to global search on Backspace", async () => {
    renderPalette();
    fireEvent.keyDown(document, { key: "k", ctrlKey: true });
    const input = await screen.findByRole("combobox", { name: "Search" });
    expect(input).toHaveAttribute("placeholder", "Search keys by alias or ID…");

    fireEvent.keyDown(input, { key: "Backspace" });
    expect(input).toHaveAttribute("placeholder", "Search pages and actions…");
  });
});
