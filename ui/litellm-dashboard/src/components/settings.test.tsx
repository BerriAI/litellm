import { act, fireEvent, render, waitFor } from "@testing-library/react";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { alertingSettingsCall, getCallbackConfigsCall, getCallbacksCall } from "./networking";
import Settings from "./settings";

// Settings (and its CloudZero cost-tracking child) renders react-query hooks, so
// every render must sit under a QueryClientProvider. Retries off so a failed
// query surfaces immediately instead of hanging the test.
const renderSettings = (props: Record<string, unknown>) =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <Settings {...(props as any)} />
    </QueryClientProvider>,
  );

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

// Settings now pulls logging-destination credentials via the useCredentials
// react-query hook; the test renders <Settings> without a QueryClientProvider,
// so stub the hook to a stable empty result instead of standing up a client.
vi.mock("@/app/(dashboard)/hooks/credentials/useCredentials", () => ({
  useCredentials: () => ({ data: { credentials: [] }, refetch: vi.fn() }),
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
    const { getByText } = renderSettings(defaultProps);

    await waitFor(() => {
      expect(getByText("Active Logging Callbacks")).toBeInTheDocument();
    });
  });

  it("should display additional settings tabs", async () => {
    const { getByText } = renderSettings(defaultProps);

    await waitFor(() => {
      expect(getByText("CloudZero Cost Tracking")).toBeInTheDocument();
      expect(getByText("Alerting Types")).toBeInTheDocument();
      expect(getByText("Alerting Settings")).toBeInTheDocument();
      expect(getByText("Email Alerts")).toBeInTheDocument();
    });
  });

  it("should load callback configs from the backend when access token is provided", async () => {
    renderSettings(defaultProps);

    await waitFor(() => {
      expect(mockGetCallbackConfigsCall).toHaveBeenCalledWith(defaultProps.accessToken);
    });
  });

  it("should display edit modal with fields when edit is clicked", async () => {
    // Use a config callback that is NOT a logging-destination backend (langfuse,
    // arize, etc. are filtered from the table via NON_CALLBACK_LOGGING_IDS and
    // edited through the destination flow instead). Datadog is a plain config
    // callback, so it still renders a row with the legacy Test/Edit/Delete actions.
    const mockCallback = {
      name: "datadog",
      variables: {
        DD_API_KEY: "test-api-key",
        DD_SITE: "us5.datadoghq.com",
      },
    };

    const mockCallbackConfig = {
      id: "datadog",
      displayName: "Datadog",
      dynamic_params: {
        DD_API_KEY: {
          type: "password",
          ui_name: "API Key",
          required: true,
        },
        DD_SITE: {
          type: "text",
          ui_name: "Site",
          required: true,
        },
      },
    };

    mockGetCallbacksCall.mockResolvedValue({
      callbacks: [mockCallback],
      available_callbacks: {
        datadog: {
          litellm_callback_name: "datadog",
          litellm_callback_params: ["DD_API_KEY", "DD_SITE"],
          ui_callback_name: "Datadog",
        },
      },
      alerts: [],
    });

    mockGetCallbackConfigsCall.mockResolvedValue([mockCallbackConfig]);

    const { getByText, container } = renderSettings(defaultProps);

    await waitFor(() => {
      expect(getByText("Active Logging Callbacks")).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(getByText("Datadog")).toBeInTheDocument();
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
      expect(getByText("API Key")).toBeInTheDocument();
      expect(getByText("Site")).toBeInTheDocument();
    });
  });

  it("should display CloudZero Cost Tracking tab", async () => {
    const { getByText } = renderSettings(defaultProps);

    await waitFor(() => {
      expect(getByText("Active Logging Callbacks")).toBeInTheDocument();
    });

    expect(getByText("CloudZero Cost Tracking")).toBeInTheDocument();
  });
});
