import { render, screen, fireEvent, act } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import UISettings from "./UISettings";
import NotificationManager from "@/components/molecules/notifications_manager";

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

vi.mock("@/components/molecules/notifications_manager", () => ({
  default: {
    success: vi.fn(),
    fromBackend: vi.fn(),
  },
}));

const buildSettingsResponse = (overrides?: Partial<Record<string, unknown>>) => ({
  data: {
    field_schema: {
      description: "UI settings description",
      properties: {
        disable_model_add_for_internal_users: {
          description: "Disable model add for internal users",
        },
        allow_model_add_for_team_admins: {
          description: "Allow model add for team admins",
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
      allow_model_add_for_team_admins: false,
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
    expect(screen.getByRole("switch", { name: "Allow model add for team admins" })).toBeInTheDocument();
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
    expect(NotificationManager.success).toHaveBeenCalledWith("UI settings updated successfully");
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
    expect(NotificationManager.success).toHaveBeenCalledWith("UI settings updated successfully");
  });

  it("should toggle allow model add for team admins setting and call update", () => {
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
          field_schema: buildSettingsResponse().data.field_schema,
          values: {
            ...buildSettingsResponse().data.values,
            disable_model_add_for_internal_users: true,
          },
        },
      }),
    );

    render(<UISettings />);

    const toggle = screen.getByRole("switch", { name: "Allow model add for team admins" });

    act(() => {
      fireEvent.click(toggle);
    });

    expect(mutateMock).toHaveBeenCalledWith(
      { allow_model_add_for_team_admins: true },
      expect.objectContaining({
        onSuccess: expect.any(Function),
        onError: expect.any(Function),
      }),
    );
    expect(NotificationManager.success).toHaveBeenCalledWith("UI settings updated successfully");
  });

  // The exemption only matters once the kill switch is on, so the control stays disabled until then.
  it("should disable the team-admin exemption toggle until the kill switch is on", () => {
    render(<UISettings />);

    const toggle = screen.getByRole("switch", { name: "Allow model add for team admins" });

    expect(toggle).toBeDisabled();
  });

  it("should enable the team-admin exemption toggle once the kill switch is on", () => {
    mockUseUISettings.mockReturnValue(
      buildSettingsResponse({
        data: {
          field_schema: buildSettingsResponse().data.field_schema,
          values: {
            ...buildSettingsResponse().data.values,
            disable_model_add_for_internal_users: true,
          },
        },
      }),
    );

    render(<UISettings />);

    const toggle = screen.getByRole("switch", { name: "Allow model add for team admins" });

    expect(toggle).not.toBeDisabled();
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
    expect(NotificationManager.success).toHaveBeenCalledWith("UI settings updated successfully");
  });
});
