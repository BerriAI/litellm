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

  it("should save user banner settings", () => {
    const mutateMock = vi.fn((_settings, options) => {
      options?.onSuccess?.();
    });

    mockUseUISettings.mockReturnValue(
      buildSettingsResponse({
        data: {
          field_schema: {
            description: "UI settings description",
            properties: {
              user_banner_enabled: {
                description: "Show a banner to dashboard users",
              },
              user_banner_message: {
                description: "Message shown in the dashboard banner",
              },
              user_banner_type: {
                description: "Visual style for the dashboard banner",
              },
            },
          },
          values: {
            user_banner_enabled: false,
            user_banner_message: "Initial message",
            user_banner_type: "info",
          },
        },
      }),
    );
    mockUseUpdateUISettings.mockReturnValue({
      mutate: mutateMock,
      isPending: false,
      error: null,
    });

    render(<UISettings />);

    act(() => {
      fireEvent.change(screen.getByLabelText("User banner message"), {
        target: { value: "Updated dashboard announcement" },
      });
    });
    act(() => {
      fireEvent.click(screen.getByRole("switch", { name: "Publish user banner" }));
    });
    act(() => {
      fireEvent.click(screen.getByRole("button", { name: "Save user banner" }));
    });

    expect(mutateMock).toHaveBeenCalledWith(
      {
        user_banner_enabled: true,
        user_banner_message: "Updated dashboard announcement",
        user_banner_type: "info",
      },
      expect.objectContaining({
        onSuccess: expect.any(Function),
        onError: expect.any(Function),
      }),
    );
    expect(NotificationManager.success).toHaveBeenCalledWith("User banner settings updated successfully");
  });
});
