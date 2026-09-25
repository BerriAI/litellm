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

const buildSettingsResponse = (overrides?: Partial<Record<string, unknown>>, source: Record<string, string> = {}) => ({
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
    source,
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

  describe("config.yaml owned settings", () => {
    it("freezes a switch whose source is config and does not call update", () => {
      const mutateMock = vi.fn();
      mockUseUpdateUISettings.mockReturnValue({ mutate: mutateMock, isPending: false, error: null });
      mockUseUISettings.mockReturnValue(
        buildSettingsResponse(undefined, {
          disable_model_add_for_internal_users: "config",
          disable_team_admin_delete_team_user: "db",
        }),
      );

      render(<UISettings />);

      const frozen = screen.getByRole("switch", { name: "Disable model add for internal users" });
      expect(frozen).toHaveAttribute("data-disabled");
      expect(frozen).toHaveAttribute("aria-disabled", "true");

      act(() => {
        fireEvent.click(frozen);
      });

      expect(mutateMock).not.toHaveBeenCalled();
    });

    it.each(["env", "default", "db", "unset"])("keeps a switch editable when its source is %s", (source) => {
      const mutateMock = vi.fn();
      mockUseUpdateUISettings.mockReturnValue({ mutate: mutateMock, isPending: false, error: null });
      mockUseUISettings.mockReturnValue(
        buildSettingsResponse(undefined, { disable_model_add_for_internal_users: source }),
      );

      render(<UISettings />);

      const toggle = screen.getByRole("switch", { name: "Disable model add for internal users" });
      expect(toggle).not.toHaveAttribute("data-disabled");
      expect(toggle).not.toHaveAttribute("aria-disabled", "true");

      act(() => {
        fireEvent.click(toggle);
      });

      expect(mutateMock).toHaveBeenCalledWith({ disable_model_add_for_internal_users: true }, expect.anything());
    });

    it("freezes only the config owned switch and leaves siblings editable", () => {
      mockUseUISettings.mockReturnValue(buildSettingsResponse(undefined, { require_auth_for_public_ai_hub: "config" }));

      render(<UISettings />);

      expect(screen.getByRole("switch", { name: "Require authentication for public AI Hub" })).toHaveAttribute(
        "data-disabled",
      );
      expect(screen.getByRole("switch", { name: "Disable model add for internal users" })).not.toHaveAttribute(
        "data-disabled",
      );
      expect(screen.getByRole("switch", { name: "Disable team admin delete team user" })).not.toHaveAttribute(
        "data-disabled",
      );
    });
  });
});
