import { render, screen, fireEvent, act } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import UISettings from "./UISettings";
import { toast } from "@/lib/toast";

const mockUseAuthorized = vi.hoisted(() => vi.fn());
const mockUseUISettings = vi.hoisted(() => vi.fn());
const mockUseUpdateUISettings = vi.hoisted(() => vi.fn());

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  __esModule: true,
  default: mockUseAuthorized,
}));

vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({
  useUISettings: mockUseUISettings,
}));

vi.mock("@/app/(dashboard)/hooks/uiSettings/useUpdateUISettings", () => ({
  useUpdateUISettings: mockUseUpdateUISettings,
}));

const buildSettingsResponse = (overrides?: Partial<Record<string, unknown>>) => ({
  data: {
    field_schema: {
      description: "UI settings description",
      properties: {
        disable_model_add_for_internal_users: {
          description: "Disable model add for internal users",
        },
        disable_team_admin_delete_team_user: {
          description: "Disable team admin delete team user",
        },
        require_auth_for_public_ai_hub: {
          description: "Require authentication for public AI Hub",
        },
      },
    },
    values: {
      disable_model_add_for_internal_users: false,
      disable_team_admin_delete_team_user: false,
      require_auth_for_public_ai_hub: false,
    },
  },
  isLoading: false,
  isError: false,
  error: null,
  ...overrides,
});

describe("UISettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseAuthorized.mockReturnValue({ accessToken: "test-token" });
    mockUseUISettings.mockReturnValue(buildSettingsResponse());
    mockUseUpdateUISettings.mockReturnValue({
      mutate: vi.fn(),
      isPending: false,
      error: null,
    });
  });

  it("should render", () => {
    render(<UISettings />);

    expect(screen.getByText("UI Settings")).toBeInTheDocument();
    expect(screen.getByRole("switch", { name: "Disable model add for internal users" })).toBeInTheDocument();
    expect(screen.getByRole("switch", { name: "Disable team admin delete team user" })).toBeInTheDocument();
    expect(screen.getByRole("switch", { name: "Require authentication for public AI Hub" })).toBeInTheDocument();
  });

  it("should toggle setting and call update", () => {
    const mutateMock = vi.fn((_settings, options) => {
      options?.onSuccess?.();
    });

    mockUseUpdateUISettings.mockReturnValue({
      mutate: mutateMock,
      isPending: false,
      error: null,
    });

    render(<UISettings />);

    const toggle = screen.getByRole("switch", { name: "Disable model add for internal users" });

    act(() => {
      fireEvent.click(toggle);
    });

    expect(mutateMock).toHaveBeenCalledWith(
      { disable_model_add_for_internal_users: true },
      expect.objectContaining({
        onSuccess: expect.any(Function),
        onError: expect.any(Function),
      }),
    );
    expect(toast.success).toHaveBeenCalledWith("UI settings updated successfully");
  });

  it("should toggle disable team admin delete team user setting and call update", () => {
    const mutateMock = vi.fn((_settings, options) => {
      options?.onSuccess?.();
    });

    mockUseUpdateUISettings.mockReturnValue({
      mutate: mutateMock,
      isPending: false,
      error: null,
    });

    render(<UISettings />);

    const toggle = screen.getByRole("switch", { name: "Disable team admin delete team user" });

    act(() => {
      fireEvent.click(toggle);
    });

    expect(mutateMock).toHaveBeenCalledWith(
      { disable_team_admin_delete_team_user: true },
      expect.objectContaining({
        onSuccess: expect.any(Function),
        onError: expect.any(Function),
      }),
    );
    expect(toast.success).toHaveBeenCalledWith("UI settings updated successfully");
  });

  it("disconnects Moyai by patching moyai_url to null", () => {
    const mutateMock = vi.fn((_settings, options) => {
      options?.onSuccess?.();
    });

    mockUseUpdateUISettings.mockReturnValue({
      mutate: mutateMock,
      isPending: false,
      error: null,
    });
    mockUseUISettings.mockReturnValue(
      buildSettingsResponse({
        data: {
          ...buildSettingsResponse().data,
          values: { ...buildSettingsResponse().data.values, moyai_url: "https://moyai.example.com" },
        },
      }),
    );

    render(<UISettings />);

    expect(screen.getByText("Connected to https://moyai.example.com")).toBeInTheDocument();
    act(() => {
      fireEvent.click(screen.getByRole("button", { name: "Disconnect" }));
    });

    expect(mutateMock).toHaveBeenCalledWith(
      { moyai_url: null },
      expect.objectContaining({ onSuccess: expect.any(Function) }),
    );
  });

  it("shows a link to the Moyai page when not connected", () => {
    render(<UISettings />);

    expect(screen.getByText("Not connected")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Connect from the Moyai page" })).toHaveAttribute("href", "/ui/moyai");
  });

  it("should toggle require auth for public AI Hub setting and call update", () => {
    const mutateMock = vi.fn((_settings, options) => {
      options?.onSuccess?.();
    });

    mockUseUpdateUISettings.mockReturnValue({
      mutate: mutateMock,
      isPending: false,
      error: null,
    });

    render(<UISettings />);

    const toggle = screen.getByRole("switch", { name: "Require authentication for public AI Hub" });

    act(() => {
      fireEvent.click(toggle);
    });

    expect(mutateMock).toHaveBeenCalledWith(
      { require_auth_for_public_ai_hub: true },
      expect.objectContaining({
        onSuccess: expect.any(Function),
        onError: expect.any(Function),
      }),
    );
    expect(toast.success).toHaveBeenCalledWith("UI settings updated successfully");
  });

  describe("settings the proxy config file sets", () => {
    const switchSettings = [
      ["disable_model_add_for_internal_users", "Disable model add for internal users"],
      ["disable_team_admin_delete_team_user", "Disable team admin delete team user"],
      ["require_auth_for_public_ai_hub", "Require authentication for public AI Hub"],
      ["forward_client_headers_to_llm_api", "Forward client headers to LLM API"],
      ["forward_llm_provider_auth_headers", "Forward LLM provider auth headers"],
      ["enable_projects_ui", "Enable Projects UI"],
      ["enable_chat_ui", "Enable Chat page"],
      ["disable_agents_for_internal_users", "Disable agents for internal users"],
      ["allow_agents_for_team_admins", "Allow agents for team admins"],
      ["disable_vector_stores_for_internal_users", "Disable vector stores for internal users"],
      ["allow_vector_stores_for_team_admins", "Allow vector stores for team admins"],
      ["scope_user_search_to_org", "Scope user search to organization"],
      ["disable_custom_api_keys", "Disable custom Virtual key values"],
    ] as const;

    const mockSources = (source: Record<string, string>, values: Record<string, unknown> = {}) => {
      const base = buildSettingsResponse().data;
      mockUseUISettings.mockReturnValue(
        buildSettingsResponse({
          data: {
            ...base,
            field_schema: {
              ...base.field_schema,
              properties: {
                ...base.field_schema.properties,
                enable_projects_ui: { description: "Enable Projects UI" },
              },
            },
            values: {
              ...base.values,
              disable_agents_for_internal_users: true,
              disable_vector_stores_for_internal_users: true,
              ...values,
            },
            source,
          },
        }),
      );
    };

    it.each(switchSettings)("greys out %s and points to the config file, leaving the rest editable", (key, name) => {
      const mutateMock = vi.fn();
      mockUseUpdateUISettings.mockReturnValue({ mutate: mutateMock, isPending: false, error: null });
      mockSources(Object.fromEntries(switchSettings.map(([other]) => [other, other === key ? "config" : "db"])));

      render(<UISettings />);

      expect(screen.getByRole("switch", { name })).toHaveAttribute("aria-disabled", "true");
      expect(screen.getByText(`general_settings.${key}`)).toBeInTheDocument();
      expect(screen.getAllByText(/proxy config file/)).toHaveLength(1);
      switchSettings
        .filter(([other]) => other !== key)
        .forEach(([, otherName]) =>
          expect(screen.getByRole("switch", { name: otherName })).not.toHaveAttribute("aria-disabled", "true"),
        );

      fireEvent.click(screen.getByRole("switch", { name }));

      expect(mutateMock).not.toHaveBeenCalled();
    });

    it("keeps every switch editable when the values come from the database or defaults", () => {
      mockSources(Object.fromEntries(switchSettings.map(([key], index) => [key, index % 2 === 0 ? "db" : "default"])));

      render(<UISettings />);

      switchSettings.forEach(([, name]) =>
        expect(screen.getByRole("switch", { name })).not.toHaveAttribute("aria-disabled", "true"),
      );
      expect(screen.queryByText(/proxy config file/)).not.toBeInTheDocument();
    });

    it("blocks disconnecting Moyai when the config file sets moyai_url", () => {
      const mutateMock = vi.fn();
      mockUseUpdateUISettings.mockReturnValue({ mutate: mutateMock, isPending: false, error: null });
      mockSources({ moyai_url: "config" }, { moyai_url: "https://moyai.example.com" });

      render(<UISettings />);
      fireEvent.click(screen.getByRole("button", { name: "Disconnect" }));

      expect(screen.getByRole("button", { name: "Disconnect" })).toBeDisabled();
      expect(screen.getByText("general_settings.moyai_url")).toBeInTheDocument();
      expect(mutateMock).not.toHaveBeenCalled();
    });

    it("locks internal user page visibility when the config file sets it", () => {
      mockSources({ enabled_ui_pages_internal_users: "config" }, { enabled_ui_pages_internal_users: ["usage"] });

      render(<UISettings />);

      expect(screen.getByText("general_settings.enabled_ui_pages_internal_users")).toBeInTheDocument();
    });
  });
});
