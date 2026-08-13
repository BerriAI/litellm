import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Form } from "antd";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { alertingSettingsCall, getCallbackConfigsCall, getCallbacksCall } from "./networking";
import Settings, { backendCallbackLogoSrc, CallbackSelector } from "./settings";

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

    const user = userEvent.setup();
    const { getByText } = render(<Settings {...defaultProps} />);

    await waitFor(() => {
      expect(getByText("Active Logging Callbacks")).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(getByText("Langfuse")).toBeInTheDocument();
    });

    await user.click(screen.getByTestId("callback-actions-langfuse-success"));
    await user.click(await screen.findByTestId("callback-action-edit"));

    await waitFor(() => {
      expect(getByText("Edit Callback Settings")).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(getByText("Public Key")).toBeInTheDocument();
      expect(getByText("Secret Key")).toBeInTheDocument();
      expect(getByText("Host")).toBeInTheDocument();
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

    render(<Settings {...defaultProps} />);

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
    render(<Settings {...defaultProps} userID={null as unknown as string} />);

    await waitFor(() => {
      expect(screen.queryByTestId("skeleton-row")).not.toBeInTheDocument();
    });
    expect(mockGetCallbacksCall).not.toHaveBeenCalled();
    expect(screen.getByText("No callbacks configured")).toBeInTheDocument();
  });

  it("should display CloudZero Cost Tracking tab", async () => {
    const { getByText } = render(<Settings {...defaultProps} />);

    await waitFor(() => {
      expect(getByText("Active Logging Callbacks")).toBeInTheDocument();
    });

    expect(getByText("CloudZero Cost Tracking")).toBeInTheDocument();
  });
});

describe("backendCallbackLogoSrc", () => {
  it("prefixes bare filenames with the assets logo folder", () => {
    expect(backendCallbackLogoSrc("datadog.png")).toBe("/ui/assets/logos/datadog.png");
  });

  it("passes through urls, data uris, and paths untouched", () => {
    expect(backendCallbackLogoSrc("https://logos.example.com/x.png")).toBe("https://logos.example.com/x.png");
    expect(backendCallbackLogoSrc("data:image/png;base64,abc")).toBe("data:image/png;base64,abc");
    expect(backendCallbackLogoSrc("/custom/path.png")).toBe("/custom/path.png");
  });

  it("returns undefined when the backend provides no logo", () => {
    expect(backendCallbackLogoSrc(undefined)).toBeUndefined();
    expect(backendCallbackLogoSrc(null)).toBeUndefined();
    expect(backendCallbackLogoSrc("")).toBeUndefined();
  });
});

describe("CallbackSelector logos", () => {
  it("resolves backend logos per entry: bare filename, external url, and missing logo", async () => {
    const callbackConfigs = [
      { id: "langfuse", displayName: "Langfuse", logo: "langfuse.png" },
      { id: "hosted", displayName: "Hosted", logo: "https://logos.example.com/hosted.png" },
      { id: "nologo", displayName: "NoLogo" },
    ];

    render(
      <Form>
        <CallbackSelector callbackConfigs={callbackConfigs} selectedCallback={null} onCallbackChange={vi.fn()} />
      </Form>,
    );

    fireEvent.mouseDown(screen.getByRole("combobox"));

    expect(await screen.findByAltText("Langfuse logo")).toHaveAttribute("src", "/ui/assets/logos/langfuse.png");
    expect(screen.getByAltText("Hosted logo")).toHaveAttribute("src", "https://logos.example.com/hosted.png");
    expect(screen.queryByAltText("NoLogo logo")).toBeNull();
    expect(screen.getByText("N")).toBeInTheDocument();
  });
});
