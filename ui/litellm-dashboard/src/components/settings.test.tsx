import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { FormProvider, useForm } from "react-hook-form";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { alertingSettingsCall, getCallbackConfigsCall, getCallbacksCall, setCallbacksCall } from "./networking";
import Settings, { backendCallbackLogoSrc, CallbackSelector } from "./settings";

vi.mock("./networking", () => ({
  getCallbacksCall: vi.fn(),
  getCallbackConfigsCall: vi.fn(),
  setCallbacksCall: vi.fn(),
  serviceHealthCheck: vi.fn(),
  deleteCallback: vi.fn(),
  alertingSettingsCall: vi.fn().mockResolvedValue([]),
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

  const openLangfuseEditModal = async () => {
    mockGetCallbacksCall.mockResolvedValue({
      callbacks: [
        {
          name: "langfuse",
          variables: {
            LANGFUSE_PUBLIC_KEY: "test-public-key",
            LANGFUSE_SECRET_KEY: "test-secret-key",
            LANGFUSE_HOST: "https://test.langfuse.com",
            SLACK_WEBHOOK_URL: null,
            OPENMETER_API_KEY: null,
          },
        },
      ],
      available_callbacks: {
        langfuse: {
          litellm_callback_name: "langfuse",
          litellm_callback_params: ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"],
          ui_callback_name: "Langfuse",
        },
      },
      alerts: [],
    });

    mockGetCallbackConfigsCall.mockResolvedValue([
      {
        id: "langfuse",
        displayName: "Langfuse",
        dynamic_params: {
          LANGFUSE_PUBLIC_KEY: { type: "text", ui_name: "Public Key", required: true },
          LANGFUSE_SECRET_KEY: { type: "password", ui_name: "Secret Key", required: true },
          LANGFUSE_HOST: { type: "text", ui_name: "Host", required: false },
        },
      },
    ]);

    const user = userEvent.setup();
    render(<Settings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Active Logging Callbacks")).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText("Langfuse")).toBeInTheDocument();
    });

    await user.click(screen.getByTestId("callback-actions-langfuse-success"));
    await user.click(await screen.findByTestId("callback-action-edit"));

    await waitFor(() => {
      expect(screen.getByText("Edit Callback Settings")).toBeInTheDocument();
    });

    return user;
  };

  it("should display edit modal with fields when edit is clicked", async () => {
    await openLangfuseEditModal();

    await waitFor(() => {
      expect(screen.getByText("Public Key")).toBeInTheDocument();
      expect(screen.getByText("Secret Key")).toBeInTheDocument();
      expect(screen.getByText("Host")).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByLabelText("Public Key")).toHaveValue("test-public-key");
    });
    expect(screen.getByLabelText("Secret Key")).toHaveValue("test-secret-key");
    expect(screen.getByLabelText("Host")).toHaveValue("https://test.langfuse.com");

    const danglingLabels = [...document.querySelectorAll("label[for]")].filter(
      (label) => document.getElementById(label.getAttribute("for") as string) === null,
    );
    expect(danglingLabels).toEqual([]);
  });

  it("should post the edited callback variables when the edit modal is saved", async () => {
    const user = await openLangfuseEditModal();

    await waitFor(() => {
      expect(screen.getByLabelText("Host")).toHaveValue("https://test.langfuse.com");
    });

    await user.clear(screen.getByLabelText("Host"));
    fireEvent.change(screen.getByLabelText("Host"), { target: { value: "https://edited.langfuse.com" } });
    await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Save Changes" }));

    await waitFor(() => {
      expect(vi.mocked(setCallbacksCall)).toHaveBeenCalledWith("token", {
        environment_variables: {
          callback: "langfuse",
          LANGFUSE_PUBLIC_KEY: "test-public-key",
          LANGFUSE_SECRET_KEY: "test-secret-key",
          LANGFUSE_HOST: "https://edited.langfuse.com",
        },
        litellm_settings: { success_callback: ["langfuse"] },
      });
    });
  });

  it("should block the edit submit when a required field is emptied", async () => {
    const user = await openLangfuseEditModal();

    await waitFor(() => {
      expect(screen.getByLabelText("Public Key")).toHaveValue("test-public-key");
    });

    await user.clear(screen.getByLabelText("Public Key"));
    await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Save Changes" }));

    expect(await screen.findByText("Please enter the public key")).toBeInTheDocument();
    expect(vi.mocked(setCallbacksCall)).not.toHaveBeenCalled();
  });

  it("should send the typed webhook url for an alert type when the alerting tab is saved", async () => {
    const user = userEvent.setup();
    render(<Settings {...defaultProps} />);

    await user.click(await screen.findByRole("tab", { name: "Alerting Types" }));

    const webhookInput = document.querySelector('input[name="llm_exceptions"]') as HTMLInputElement;
    expect(webhookInput).not.toBeNull();
    fireEvent.change(webhookInput, { target: { value: "https://hooks.example.com/llm-exceptions" } });

    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await waitFor(() => {
      expect(vi.mocked(setCallbacksCall)).toHaveBeenCalledWith("token", {
        general_settings: expect.objectContaining({
          alert_to_webhook_url: expect.objectContaining({
            llm_exceptions: "https://hooks.example.com/llm-exceptions",
          }),
        }),
      });
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

const CallbackSelectorHarness = ({
  callbackConfigs,
}: {
  callbackConfigs: { id: string; displayName: string; logo?: string }[];
}) => {
  const form = useForm<Record<string, string>>();
  return (
    <FormProvider {...form}>
      <CallbackSelector callbackConfigs={callbackConfigs} selectedCallback={null} onCallbackChange={vi.fn()} />
    </FormProvider>
  );
};

describe("CallbackSelector logos", () => {
  it("resolves backend logos per entry: bare filename, external url, and missing logo", async () => {
    const callbackConfigs = [
      { id: "langfuse", displayName: "Langfuse", logo: "langfuse.png" },
      { id: "hosted", displayName: "Hosted", logo: "https://logos.example.com/hosted.png" },
      { id: "nologo", displayName: "NoLogo" },
    ];

    render(<CallbackSelectorHarness callbackConfigs={callbackConfigs} />);

    await userEvent.click(screen.getByRole("combobox"));

    expect(await screen.findByAltText("Langfuse logo")).toHaveAttribute("src", "/ui/assets/logos/langfuse.png");
    expect(screen.getByAltText("Hosted logo")).toHaveAttribute("src", "https://logos.example.com/hosted.png");
    expect(screen.queryByAltText("NoLogo logo")).not.toBeInTheDocument();
    expect(screen.getByText("N")).toBeInTheDocument();
  });
});
