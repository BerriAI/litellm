import { act, fireEvent, getByRole, render, waitFor, within } from "@testing-library/react";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import {
  alertingSettingsCall,
  deleteCallback,
  getCallbackConfigsCall,
  getCallbacksCall,
  setCallbacksCall,
} from "./networking";
import Settings from "./settings";

vi.mock("./networking", () => ({
  getCallbacksCall: vi.fn(),
  getCallbackConfigsCall: vi.fn(),
  setCallbacksCall: vi.fn(),
  serviceHealthCheck: vi.fn(),
  deleteCallback: vi.fn(),
  alertingSettingsCall: vi.fn().mockResolvedValue([]),
}));

vi.mock("./molecules/notifications_manager", () => ({
  __esModule: true,
  default: {
    success: vi.fn(),
    fromBackend: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
    clear: vi.fn(),
  },
}));

vi.mock("./alerting/alerting_settings", () => ({
  __esModule: true,
  default: () => <div>Mock Alerting Settings</div>,
}));

vi.mock("./email_settings", () => ({
  __esModule: true,
  default: () => <div>Mock Email Settings</div>,
}));

vi.mock("./CloudZeroCostTracking/CloudZeroCostTracking", () => ({
  __esModule: true,
  default: () => <div>Mock CloudZero Cost Tracking</div>,
}));

// Polyfill ResizeObserver for components relying on it in tests
if (typeof window !== "undefined" && !window.ResizeObserver) {
  window.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}

beforeAll(() => {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  });
});

describe("Settings", () => {
  const defaultProps = {
    accessToken: "token",
    userRole: "admin",
    userID: "user-123",
    premiumUser: false,
  };
  const mockGetCallbacksCall = vi.mocked(getCallbacksCall);
  const mockGetCallbackConfigsCall = vi.mocked(getCallbackConfigsCall);
  const mockAlertingSettingsCall = vi.mocked(alertingSettingsCall);
  const mockSetCallbacksCall = vi.mocked(setCallbacksCall);
  const mockDeleteCallback = vi.mocked(deleteCallback);

  beforeEach(() => {
    vi.clearAllMocks();
    mockGetCallbacksCall.mockResolvedValue({
      callbacks: [],
      available_callbacks: [],
      alerts: [],
    });
    mockGetCallbackConfigsCall.mockResolvedValue([]);
    mockAlertingSettingsCall.mockResolvedValue([]);
  });

  it("should render the logging callbacks tab when access token is provided", async () => {
    const { getByText } = render(<Settings {...defaultProps} />);

    await waitFor(() => {
      expect(getByText("Active Logging Callbacks")).toBeInTheDocument();
    });
  });

  it("should display additional settings tabs", async () => {
    const { getByText } = render(<Settings {...defaultProps} />);

    await waitFor(() => {
      expect(getByText("CloudZero Cost Tracking")).toBeInTheDocument();
      expect(getByText("Alerting Types")).toBeInTheDocument();
      expect(getByText("Alerting Settings")).toBeInTheDocument();
      expect(getByText("Email Alerts")).toBeInTheDocument();
    });
  });

  it("should load callback configs from the backend when access token is provided", async () => {
    render(<Settings {...defaultProps} />);

    await waitFor(() => {
      expect(mockGetCallbackConfigsCall).toHaveBeenCalledWith(defaultProps.accessToken);
    });
  });

  it("should display edit modal with fields when edit is clicked", async () => {
    const mockCallback = {
      name: "langfuse",
      variables: {
        LANGFUSE_PUBLIC_KEY: "test-public-key",
        LANGFUSE_SECRET_KEY: "test-secret-key",
        LANGFUSE_HOST: "https://test.langfuse.com",
        SLACK_WEBHOOK_URL: null,
        OPENMETER_API_KEY: null,
      },
    };

    const mockCallbackConfig = {
      id: "langfuse",
      displayName: "Langfuse",
      dynamic_params: {
        LANGFUSE_PUBLIC_KEY: {
          type: "text",
          ui_name: "Public Key",
          required: true,
        },
        LANGFUSE_SECRET_KEY: {
          type: "password",
          ui_name: "Secret Key",
          required: true,
        },
        LANGFUSE_HOST: {
          type: "text",
          ui_name: "Host",
          required: false,
        },
      },
    };

    mockGetCallbacksCall.mockResolvedValue({
      callbacks: [mockCallback],
      available_callbacks: {
        langfuse: {
          litellm_callback_name: "langfuse",
          litellm_callback_params: ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"],
          ui_callback_name: "Langfuse",
        },
      },
      alerts: [],
    });

    mockGetCallbackConfigsCall.mockResolvedValue([mockCallbackConfig]);

    const { getByText, container } = render(<Settings {...defaultProps} />);

    await waitFor(() => {
      expect(getByText("Active Logging Callbacks")).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(getByText("Langfuse")).toBeInTheDocument();
    });

    const actionsCell = container.querySelector('[class*="flex justify-end gap-2"]');
    expect(actionsCell).toBeTruthy();

    const icons = actionsCell?.querySelectorAll("svg");
    expect(icons?.length).toBeGreaterThanOrEqual(2);

    const editIconParent = icons?.[1]?.closest('[class*="cursor-pointer"]');
    expect(editIconParent).toBeTruthy();

    act(() => {
      fireEvent.click(editIconParent!);
    });

    await waitFor(() => {
      expect(getByText("Edit Callback Settings")).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(getByText("Public Key")).toBeInTheDocument();
      expect(getByText("Secret Key")).toBeInTheDocument();
      expect(getByText("Host")).toBeInTheDocument();
    });
  });

  it("should display CloudZero Cost Tracking tab", async () => {
    const { getByText } = render(<Settings {...defaultProps} />);

    await waitFor(() => {
      expect(getByText("Active Logging Callbacks")).toBeInTheDocument();
    });

    expect(getByText("CloudZero Cost Tracking")).toBeInTheDocument();
  });

  it("should display websearch_interception callback when returned by getCallbacksCall with params", async () => {
    mockGetCallbacksCall.mockResolvedValue({
      callbacks: [
        {
          name: "websearch_interception",
          type: "success_and_failure",
          variables: {},
          params: { enabled_providers: ["bedrock", "azure"] },
        },
      ],
      available_callbacks: {
        websearch_interception: {
          litellm_callback_name: "websearch_interception",
          litellm_callback_params: ["enabled_providers", "search_tool_name"],
          ui_callback_name: "WebSearch Interception",
        },
      },
      alerts: [],
    });
    mockGetCallbackConfigsCall.mockResolvedValue([]);

    const { getByText } = render(<Settings {...defaultProps} />);

    await waitFor(() => {
      expect(getByText("Active Logging Callbacks")).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(getByText("WebSearch Interception")).toBeInTheDocument();
      expect(getByText("Success & Failure")).toBeInTheDocument();
    });
  });

  describe("Logging callbacks", () => {
    const langfuseAvailableCallbacks = {
      langfuse: {
        litellm_callback_name: "langfuse",
        litellm_callback_params: [] as string[],
        ui_callback_name: "Langfuse",
      },
    };

    const getCallbacksResponseWithLangfuse = (callback: { name: string; type: "success"; variables: Record<string, unknown> }) => ({
      callbacks: [callback],
      available_callbacks: { ...langfuseAvailableCallbacks, langfuse: { ...langfuseAvailableCallbacks.langfuse, litellm_callback_params: Object.keys(callback.variables).filter((k) => callback.variables[k] != null) } },
      alerts: [],
    });

    it("should call setCallbacksCall with expected payload when editing and saving a callback", async () => {
      const mockCallback = {
        name: "langfuse",
        type: "success" as const,
        variables: {
          LANGFUSE_PUBLIC_KEY: "pk-test",
          LANGFUSE_SECRET_KEY: "sk-test",
          LANGFUSE_HOST: "https://langfuse.example.com",
          SLACK_WEBHOOK_URL: null,
          OPENMETER_API_KEY: null,
        },
      };
      const mockCallbackConfig = {
        id: "langfuse",
        displayName: "Langfuse",
        dynamic_params: {
          LANGFUSE_PUBLIC_KEY: { type: "text", ui_name: "Public Key", required: true },
          LANGFUSE_SECRET_KEY: { type: "password", ui_name: "Secret Key", required: true },
          LANGFUSE_HOST: { type: "text", ui_name: "Host", required: false },
        },
      };

      mockGetCallbacksCall.mockResolvedValue(getCallbacksResponseWithLangfuse(mockCallback));
      mockGetCallbackConfigsCall.mockResolvedValue([mockCallbackConfig]);
      mockSetCallbacksCall.mockResolvedValue(undefined);

      const { getByText, getByTestId } = render(<Settings {...defaultProps} />);

      await waitFor(() => expect(getByText("Langfuse")).toBeInTheDocument());

      act(() => fireEvent.click(getByTestId("logging-callback-edit-langfuse")));
      await waitFor(() => expect(getByText("Edit Callback Settings")).toBeInTheDocument());

      const dialog = getByRole(document.body, "dialog", { name: /Edit Callback Settings/i });
      act(() => fireEvent.click(within(dialog).getByRole("button", { name: /Save Changes/i })));

      await waitFor(() =>
        expect(mockSetCallbacksCall).toHaveBeenCalledWith(
          defaultProps.accessToken,
          expect.objectContaining({
            litellm_settings: expect.objectContaining({ success_callback: ["langfuse"] }),
          })
        )
      );
    });

    it("should call deleteCallback with callback name when delete is confirmed", async () => {
      const mockCallback = { name: "langfuse", type: "success" as const, variables: {} };
      mockGetCallbacksCall.mockResolvedValue(getCallbacksResponseWithLangfuse(mockCallback));
      mockGetCallbackConfigsCall.mockResolvedValue([]);
      mockDeleteCallback.mockResolvedValue(undefined);

      const { getByText, getByTestId } = render(<Settings {...defaultProps} />);

      await waitFor(() => expect(getByText("Langfuse")).toBeInTheDocument());

      act(() => fireEvent.click(getByTestId("logging-callback-delete-langfuse")));
      await waitFor(() => expect(getByRole(document.body, "dialog", { name: "Delete Callback" })).toBeInTheDocument());

      const deleteDialog = getByRole(document.body, "dialog", { name: "Delete Callback" });
      act(() => fireEvent.click(within(deleteDialog).getByRole("button", { name: /^Delete$/ })));

      await waitFor(() => expect(mockDeleteCallback).toHaveBeenCalledWith(defaultProps.accessToken, "langfuse"));
    });

    it("should handle getCallbacksCall rejection without crashing", async () => {
      const NotificationsManager = (await import("./molecules/notifications_manager")).default;
      mockGetCallbacksCall.mockRejectedValue(new Error("Network error"));

      const { container, getByText } = render(<Settings {...defaultProps} />);

      await waitFor(() => expect(getByText("Active Logging Callbacks")).toBeInTheDocument());
      await waitFor(() => expect(NotificationsManager.fromBackend).toHaveBeenCalled());

      expect(container).toBeInTheDocument();
    });

    it("should handle setCallbacksCall rejection when saving callback without crashing", async () => {
      const NotificationsManager = (await import("./molecules/notifications_manager")).default;
      const mockCallback = { name: "langfuse", type: "success" as const, variables: {} };
      mockGetCallbacksCall.mockResolvedValue(getCallbacksResponseWithLangfuse(mockCallback));
      mockGetCallbackConfigsCall.mockResolvedValue([]);
      mockSetCallbacksCall.mockRejectedValue(new Error("Save failed"));

      const { getByText, getByTestId } = render(<Settings {...defaultProps} />);

      await waitFor(() => expect(getByText("Langfuse")).toBeInTheDocument());

      act(() => fireEvent.click(getByTestId("logging-callback-edit-langfuse")));
      await waitFor(() => expect(getByText("Edit Callback Settings")).toBeInTheDocument());

      const editDialog = getByRole(document.body, "dialog", { name: /Edit Callback Settings/i });
      act(() => fireEvent.click(within(editDialog).getByRole("button", { name: /Save Changes/i })));

      await waitFor(() => expect(NotificationsManager.fromBackend).toHaveBeenCalled());
    });
  });
});
