import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { alertingSettingsCall, getCallbackConfigsCall, getCallbacksCall } from "./networking";
import Settings from "./settings";

type SettingsTestProps = {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
  premiumUser: boolean;
};

// Settings (and its CloudZero cost-tracking child) renders react-query hooks, so
// every render must sit under a QueryClientProvider. Retries off so a failed
// query surfaces immediately instead of hanging the test.
const renderSettings = (props: SettingsTestProps) =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <Settings {...props} />
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

    const user = userEvent.setup();
    const { getByText } = renderSettings(defaultProps);

    await waitFor(() => {
      expect(getByText("Active Logging Callbacks")).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(getByText("Datadog")).toBeInTheDocument();
    });

    await user.click(screen.getByTestId("callback-actions-datadog-success"));
    await user.click(await screen.findByTestId("callback-action-edit"));

    await waitFor(() => {
      expect(getByText("Edit Callback Settings")).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(getByText("API Key")).toBeInTheDocument();
      expect(getByText("Site")).toBeInTheDocument();
    });
  });

  it("should hold the callbacks table in loading state until the fetch settles", async () => {
    let resolveCallbacks: (value: {
      callbacks: never[];
      available_callbacks: never[];
      alerts: never[];
    }) => void = () => {};
    mockGetCallbacksCall.mockReturnValue(
      new Promise((resolve) => {
        resolveCallbacks = resolve;
      }),
    );

    renderSettings(defaultProps);

    expect(screen.getAllByTestId("skeleton-row").length).toBeGreaterThan(0);

    await act(async () => {
      resolveCallbacks({ callbacks: [], available_callbacks: [], alerts: [] });
    });

    await waitFor(() => {
      expect(screen.queryByTestId("skeleton-row")).not.toBeInTheDocument();
    });
    expect(screen.getByText("No callbacks configured")).toBeInTheDocument();
  });

  it("should resolve loading without fetching when the user id is missing", async () => {
    renderSettings({ ...defaultProps, userID: null });

    await waitFor(() => {
      expect(screen.queryByTestId("skeleton-row")).not.toBeInTheDocument();
    });
    expect(mockGetCallbacksCall).not.toHaveBeenCalled();
    expect(screen.getByText("No callbacks configured")).toBeInTheDocument();
  });

  it("should display CloudZero Cost Tracking tab", async () => {
    const { getByText } = renderSettings(defaultProps);

    await waitFor(() => {
      expect(getByText("Active Logging Callbacks")).toBeInTheDocument();
    });

    expect(getByText("CloudZero Cost Tracking")).toBeInTheDocument();
  });
});
