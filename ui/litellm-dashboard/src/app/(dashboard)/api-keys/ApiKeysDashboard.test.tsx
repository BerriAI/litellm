import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { useEffect } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { CreateKeyPrefillData } from "@/components/organisms/create_key_button";

import { render as renderBare, renderWithProviders as render, screen, waitFor } from "../../../../tests/test-utils";

interface CreateKeyMockProps {
  autoOpenCreate?: boolean;
  prefillData?: CreateKeyPrefillData;
  onAutoOpened?: () => void;
}

const { teamListCall, authorizedSession, createKeyProps } = vi.hoisted(() => ({
  teamListCall: vi.fn(() => new Promise(() => {})),
  authorizedSession: vi.fn(),
  createKeyProps: vi.fn<(props: CreateKeyMockProps) => void>(),
}));

const session = (overrides: { userRole?: string; isViewOnly?: boolean } = {}) => ({
  isLoading: false,
  isAuthorized: true,
  token: "jwt",
  accessToken: "sk-access",
  userId: "u-123",
  userEmail: "admin@example.com",
  userRole: "Admin",
  isViewOnly: false,
  premiumUser: false,
  disabledPersonalKeyCreation: false,
  showSSOBanner: false,
  ...overrides,
});

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => authorizedSession(),
}));

vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  teamListCall,
}));

vi.mock("@/components/VirtualKeysPage/VirtualKeysTable", () => ({
  VirtualKeysTable: ({ headerActions }: { headerActions?: React.ReactNode }) => (
    <div>
      {headerActions}
      <table aria-label="Virtual Keys" />
    </div>
  ),
}));

vi.mock("@/components/organisms/create_key_button", () => ({
  default: function CreateKeyMock(props: CreateKeyMockProps) {
    createKeyProps(props);
    const { autoOpenCreate, onAutoOpened } = props;
    useEffect(() => {
      if (autoOpenCreate) {
        onAutoOpened?.();
      }
    }, [autoOpenCreate, onAutoOpened]);
    return <button type="button">Create Key</button>;
  },
}));

import ApiKeysDashboard from "./ApiKeysDashboard";

const renderKeepingMountWrites = (searchParams: string, onUrlUpdate: OnUrlUpdateFunction) =>
  renderBare(
    <NuqsTestingAdapter
      searchParams={searchParams}
      onUrlUpdate={onUrlUpdate}
      hasMemory
      resetUrlUpdateQueueOnMount={false}
    >
      <ApiKeysDashboard />
    </NuqsTestingAdapter>,
  );

const firstCreateKeyProps = () => createKeyProps.mock.calls[0][0];
const lastCreateKeyProps = () => createKeyProps.mock.lastCall?.[0];

describe("ApiKeysDashboard", () => {
  beforeEach(() => {
    teamListCall.mockClear();
    createKeyProps.mockClear();
    authorizedSession.mockReturnValue(session());
    sessionStorage.clear();
  });

  it("scopes the team list to the signed-in user for non-admin roles", () => {
    authorizedSession.mockReturnValue(session({ userRole: "Internal User" }));
    render(<ApiKeysDashboard />);

    expect(teamListCall).toHaveBeenCalledWith("sk-access", 1, 100, { userID: "u-123" });
  });

  it("renders the keys table with a Create Key action for roles that can write", () => {
    render(<ApiKeysDashboard />);

    expect(screen.getByRole("table", { name: "Virtual Keys" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Create Key" })).toBeInTheDocument();
  });

  it("hides Create Key for view-only roles", () => {
    authorizedSession.mockReturnValue(session({ isViewOnly: true }));
    render(<ApiKeysDashboard />);

    expect(screen.getByRole("table", { name: "Virtual Keys" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Create Key" })).not.toBeInTheDocument();
  });

  it("leaves other pages' session state intact when the tab reloads", () => {
    sessionStorage.setItem("chatHistory", '[{"role":"user","content":"hi"}]');
    sessionStorage.setItem("selectedModel", "gpt-5.5");
    render(<ApiKeysDashboard />);

    window.dispatchEvent(new Event("beforeunload"));

    expect(sessionStorage.getItem("chatHistory")).toBe('[{"role":"user","content":"hi"}]');
    expect(sessionStorage.getItem("selectedModel")).toBe("gpt-5.5");
  });

  describe("create key deep link", () => {
    it("passes the sanitized prefill from the URL to Create Key", () => {
      const longAlias = `  ${"a".repeat(300)}  `;
      render(<ApiKeysDashboard />, {
        searchParams: {
          create: "true",
          owned_by: "service_account",
          team_id: "  team-1 ",
          key_alias: longAlias,
          models: " gpt-4o , ,claude-sonnet-4-5,",
          key_type: "llm_api",
        },
      });

      const expectedPrefill: CreateKeyPrefillData = {
        owned_by: "service_account",
        team_id: "team-1",
        key_alias: "a".repeat(256),
        models: ["gpt-4o", "claude-sonnet-4-5"],
        key_type: "llm_api",
      };
      const { autoOpenCreate, prefillData } = firstCreateKeyProps();
      expect(autoOpenCreate).toBe(true);
      expect(prefillData).toEqual(expectedPrefill);
    });

    it("drops owned_by and key_type values outside the allowed set", () => {
      render(<ApiKeysDashboard />, {
        searchParams: "?create=true&owned_by=someone_else&key_type=root&key_alias=my-key",
      });

      const expectedPrefill: CreateKeyPrefillData = {
        owned_by: undefined,
        team_id: undefined,
        key_alias: "my-key",
        models: undefined,
        key_type: undefined,
      };
      expect(firstCreateKeyProps().prefillData).toEqual(expectedPrefill);
    });

    it("caps models at 100 entries", () => {
      const models = Array.from({ length: 150 }, (_, index) => `model-${index}`).join(",");
      render(<ApiKeysDashboard />, { searchParams: { create: "true", models } });

      expect(firstCreateKeyProps().prefillData?.models).toHaveLength(100);
      expect(firstCreateKeyProps().prefillData?.models?.at(-1)).toBe("model-99");
    });

    it("opens without prefill when create=true has no prefill params", () => {
      render(<ApiKeysDashboard />, { searchParams: "?create=true" });

      expect(firstCreateKeyProps().autoOpenCreate).toBe(true);
      expect(firstCreateKeyProps().prefillData).toBeUndefined();
    });

    it("ignores prefill params unless create=true", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderKeepingMountWrites("?create=false&key_alias=my-key", onUrlUpdate);
      await new Promise((resolve) => setTimeout(resolve, 50));

      expect(lastCreateKeyProps()?.autoOpenCreate).toBe(false);
      expect(lastCreateKeyProps()?.prefillData).toBeUndefined();
      expect(onUrlUpdate).not.toHaveBeenCalled();
    });

    it("clears every create key param in one write once the modal opens and keeps the rest", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderKeepingMountWrites(
        "?create=true&owned_by=you&team_id=team-1&key_alias=my-key&models=gpt-4o&key_type=default" +
          "&filter_team=team-9&key_search=abc",
        onUrlUpdate,
      );

      await waitFor(() => expect(onUrlUpdate).toHaveBeenCalledTimes(1));
      const [{ searchParams, options }] = onUrlUpdate.mock.calls[0];
      expect([...searchParams.keys()].sort()).toEqual(["filter_team", "key_search"]);
      expect(searchParams.get("filter_team")).toBe("team-9");
      expect(searchParams.get("key_search")).toBe("abc");
      expect(options.history).toBe("replace");
      await waitFor(() => expect(lastCreateKeyProps()?.autoOpenCreate).toBe(false));
      expect(lastCreateKeyProps()?.prefillData).toBeUndefined();
    });
  });
});
