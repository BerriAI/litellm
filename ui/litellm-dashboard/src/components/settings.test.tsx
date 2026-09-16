import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import type { ReactNode } from "react";
import { FormProvider, useForm } from "react-hook-form";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../tests/test-utils";
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
    renderWithProviders(<Settings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Active Logging Callbacks")).toBeInTheDocument();
    });
  });

  it("should display additional settings tabs", async () => {
    renderWithProviders(<Settings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("CloudZero Cost Tracking")).toBeInTheDocument();
      expect(screen.getByText("Alerting Types")).toBeInTheDocument();
      expect(screen.getByText("Alerting Settings")).toBeInTheDocument();
      expect(screen.getByText("Email Alerts")).toBeInTheDocument();
    });
  });

  it("should load callback configs from the backend when access token is provided", async () => {
    renderWithProviders(<Settings {...defaultProps} />);

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
    renderWithProviders(<Settings {...defaultProps} />);

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

  const mockOtelCallback = (variables: Record<string, string | null>) => {
    mockGetCallbacksCall.mockResolvedValue({
      callbacks: [{ name: "otel", variables }],
      available_callbacks: {
        otel: {
          litellm_callback_name: "otel",
          litellm_callback_params: ["OTEL_EXPORTER", "OTEL_EXPORTER_OTLP_PROTOCOL", "OTEL_ENDPOINT", "OTEL_HEADERS"],
          ui_callback_name: "OpenTelemetry",
        },
      },
      alerts: [],
    });
    mockGetCallbackConfigsCall.mockResolvedValue([
      {
        id: "otel",
        displayName: "Open Telemetry",
        dynamic_params: {
          otel_endpoint: { type: "text", ui_name: "Endpoint URL", required: true },
          otel_exporter_otlp_protocol: {
            type: "select",
            ui_name: "Export Protocol",
            options: ["http/protobuf", "http/json"],
            required: false,
          },
        },
      },
    ]);
  };

  const openOtelEditModal = async () => {
    const user = userEvent.setup();
    renderWithProviders(<Settings {...defaultProps} />);
    await user.click(await screen.findByTestId("callback-actions-otel-success"));
    await user.click(await screen.findByTestId("callback-action-edit"));
    return user;
  };

  it("should post the chosen export protocol when a select dynamic param is saved", async () => {
    mockOtelCallback({ OTEL_ENDPOINT: "http://collector:4318" });
    const user = await openOtelEditModal();

    expect(await screen.findByLabelText("Endpoint URL")).toHaveValue("http://collector:4318");
    await user.click(screen.getByLabelText("Export Protocol"));
    await user.click(await screen.findByRole("option", { name: "http/json" }));
    expect(screen.getByLabelText("Export Protocol")).toHaveTextContent("http/json");

    await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Save Changes" }));

    await waitFor(() => {
      expect(vi.mocked(setCallbacksCall)).toHaveBeenCalledWith(
        "token",
        expect.objectContaining({
          environment_variables: expect.objectContaining({
            callback: "otel",
            otel_endpoint: "http://collector:4318",
            otel_exporter_otlp_protocol: "http/json",
          }),
        }),
      );
    });
  });

  it("should show the saved export protocol in the edit modal and keep it on an unchanged save", async () => {
    mockOtelCallback({ OTEL_ENDPOINT: "http://collector:4318", OTEL_EXPORTER_OTLP_PROTOCOL: "http/json" });
    const user = await openOtelEditModal();

    expect(await screen.findByLabelText("Endpoint URL")).toHaveValue("http://collector:4318");
    expect(screen.getByLabelText("Export Protocol")).toHaveTextContent("http/json");

    await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Save Changes" }));

    await waitFor(() => {
      expect(vi.mocked(setCallbacksCall)).toHaveBeenCalledWith("token", {
        environment_variables: {
          callback: "otel",
          otel_endpoint: "http://collector:4318",
          otel_exporter_otlp_protocol: "http/json",
        },
        litellm_settings: { success_callback: ["otel"] },
      });
    });
  });

  const mockS3Callback = (variables: Record<string, string | null>, callbackName = "s3") => {
    mockGetCallbacksCall.mockResolvedValue({
      callbacks: [{ name: callbackName, variables }],
      available_callbacks: {
        s3: {
          litellm_callback_name: "s3",
          litellm_callback_params: [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_REGION_NAME",
            "S3_LOG_PROMPTS_ONLY",
          ],
          ui_callback_name: "s3 Bucket (AWS)",
        },
      },
      alerts: [],
    });
    mockGetCallbackConfigsCall.mockResolvedValue([
      {
        id: "s3",
        displayName: "S3",
        dynamic_params: {
          s3_bucket_name: { type: "text", ui_name: "S3 Bucket Name", required: false },
          s3_log_prompts_only: { type: "boolean", ui_name: "Log Prompts Only", required: false },
        },
      },
    ]);
  };

  const openS3EditModal = async (callbackName = "s3") => {
    const user = userEvent.setup();
    renderWithProviders(<Settings {...defaultProps} />);
    await user.click(await screen.findByTestId(`callback-actions-${callbackName}-success`));
    await user.click(await screen.findByTestId("callback-action-edit"));
    return user;
  };

  it("should render a saved boolean dynamic param as a checked switch and post false when toggled off", async () => {
    mockS3Callback({ S3_LOG_PROMPTS_ONLY: "true" });
    const user = await openS3EditModal();

    const promptsOnlySwitch = await screen.findByRole("switch", { name: "Log Prompts Only" });
    expect(promptsOnlySwitch).toBeChecked();

    await user.click(promptsOnlySwitch);
    expect(promptsOnlySwitch).not.toBeChecked();
    await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Save Changes" }));

    await waitFor(() => {
      expect(vi.mocked(setCallbacksCall)).toHaveBeenCalledWith(
        "token",
        expect.objectContaining({
          environment_variables: expect.objectContaining({ callback: "s3", s3_log_prompts_only: "false" }),
        }),
      );
    });
  });

  it("should render an unset boolean dynamic param as an unchecked switch and post true when toggled on", async () => {
    mockS3Callback({ S3_LOG_PROMPTS_ONLY: null });
    const user = await openS3EditModal();

    const promptsOnlySwitch = await screen.findByRole("switch", { name: "Log Prompts Only" });
    expect(promptsOnlySwitch).not.toBeChecked();

    await user.click(promptsOnlySwitch);
    await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Save Changes" }));

    await waitFor(() => {
      expect(vi.mocked(setCallbacksCall)).toHaveBeenCalledWith(
        "token",
        expect.objectContaining({
          environment_variables: expect.objectContaining({ callback: "s3", s3_log_prompts_only: "true" }),
        }),
      );
    });
  });

  it.each(["True", "1"])("should render a boolean dynamic param stored as %s as a checked switch", async (stored) => {
    mockS3Callback({ S3_LOG_PROMPTS_ONLY: stored });
    await openS3EditModal();

    expect(await screen.findByRole("switch", { name: "Log Prompts Only" })).toBeChecked();
  });

  it("should resolve the s3_v2 callback to the s3 dynamic params and post under the s3_v2 name", async () => {
    mockS3Callback({ S3_LOG_PROMPTS_ONLY: null }, "s3_v2");
    const user = await openS3EditModal("s3_v2");

    const promptsOnlySwitch = await screen.findByRole("switch", { name: "Log Prompts Only" });
    expect(promptsOnlySwitch).not.toBeChecked();
    expect(within(screen.getByRole("dialog")).getByRole("combobox", { name: "Callback" })).toHaveValue("S3");

    await user.click(promptsOnlySwitch);
    await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Save Changes" }));

    await waitFor(() => {
      expect(vi.mocked(setCallbacksCall)).toHaveBeenCalledWith(
        "token",
        expect.objectContaining({
          environment_variables: expect.objectContaining({ callback: "s3_v2", s3_log_prompts_only: "true" }),
          litellm_settings: { success_callback: ["s3_v2"] },
        }),
      );
    });
  });

  it("should send the typed webhook url for an alert type when the alerting tab is saved", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Settings {...defaultProps} />);

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

    renderWithProviders(<Settings {...defaultProps} />);

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
    renderWithProviders(<Settings {...defaultProps} userID={null as unknown as string} />);

    await waitFor(() => {
      expect(screen.queryByTestId("skeleton-row")).not.toBeInTheDocument();
    });
    expect(mockGetCallbacksCall).not.toHaveBeenCalled();
    expect(screen.getByText("No callbacks configured")).toBeInTheDocument();
  });

  describe("URL state", () => {
    const langfuseVariables = (host: string) => ({
      LANGFUSE_PUBLIC_KEY: "test-public-key",
      LANGFUSE_SECRET_KEY: "test-secret-key",
      LANGFUSE_HOST: host,
      SLACK_WEBHOOK_URL: null,
      OPENMETER_API_KEY: null,
    });

    const mockLangfuseRows = () => {
      mockGetCallbacksCall.mockResolvedValue({
        callbacks: [
          { name: "langfuse", type: "success", variables: langfuseVariables("https://success.langfuse.com") },
          { name: "langfuse", type: "failure", variables: langfuseVariables("https://failure.langfuse.com") },
          { name: "datadog", type: "success", variables: langfuseVariables(""), read_only: true },
        ],
        available_callbacks: {},
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
    };

    const renderRetainingMountWrites = (searchParams: string, onUrlUpdate: OnUrlUpdateFunction) =>
      render(<Settings {...defaultProps} />, {
        wrapper: ({ children }: { children: ReactNode }) => (
          <NuqsTestingAdapter
            searchParams={searchParams}
            onUrlUpdate={onUrlUpdate}
            hasMemory
            resetUrlUpdateQueueOnMount={false}
          >
            {children}
          </NuqsTestingAdapter>
        ),
      });

    const lastUrl = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) => onUrlUpdate.mock.calls.at(-1)?.[0];

    it("opens the tab named in the URL", async () => {
      renderWithProviders(<Settings {...defaultProps} />, { searchParams: "?tab=ms-teams-alerts" });
      expect(await screen.findByRole("tab", { name: "MS Teams Alerts" })).toHaveAttribute("aria-selected", "true");
      expect(screen.getByRole("tab", { name: "Logging Callbacks" })).toHaveAttribute("aria-selected", "false");
    });

    it("writes the clicked tab to the URL and drops it for Logging Callbacks", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<Settings {...defaultProps} />, { onUrlUpdate });

      await user.click(await screen.findByRole("tab", { name: "Alerting Settings" }));
      expect(lastUrl(onUrlUpdate)?.searchParams.get("tab")).toBe("alerting-settings");
      expect(screen.getByRole("tab", { name: "Alerting Settings" })).toHaveAttribute("aria-selected", "true");

      await user.click(screen.getByRole("tab", { name: "Logging Callbacks" }));
      expect(lastUrl(onUrlUpdate)?.searchParams.has("tab")).toBe(false);
    });

    it("selects every tab that is clicked", async () => {
      const user = userEvent.setup();
      renderWithProviders(<Settings {...defaultProps} />);

      const tabs = await screen.findAllByRole("tab");
      expect(tabs).toHaveLength(6);
      for (const tab of tabs) {
        await user.click(tab);
        expect(tab).toHaveAttribute("aria-selected", "true");
      }
    });

    it("falls back to Logging Callbacks for an unknown tab in the URL", async () => {
      renderWithProviders(<Settings {...defaultProps} />, { searchParams: "?tab=billing" });
      expect(await screen.findByRole("tab", { name: "Logging Callbacks" })).toHaveAttribute("aria-selected", "true");
    });

    it("opens the edit dialog for the callback and mode in the URL", async () => {
      mockLangfuseRows();
      renderWithProviders(<Settings {...defaultProps} />, { searchParams: "?callback=langfuse&callback_mode=failure" });

      const dialog = await screen.findByRole("dialog", { name: "Edit Callback Settings" });
      await waitFor(() => expect(within(dialog).getByLabelText("Host")).toHaveValue("https://failure.langfuse.com"));
    });

    it("treats a callback link without a mode as the success registration", async () => {
      mockLangfuseRows();
      renderWithProviders(<Settings {...defaultProps} />, { searchParams: "?callback=langfuse" });

      const dialog = await screen.findByRole("dialog", { name: "Edit Callback Settings" });
      await waitFor(() => expect(within(dialog).getByLabelText("Host")).toHaveValue("https://success.langfuse.com"));
    });

    it("pushes the edited callback and its mode into the URL", async () => {
      mockLangfuseRows();
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<Settings {...defaultProps} />, { onUrlUpdate });

      await user.click(await screen.findByTestId("callback-actions-langfuse-failure"));
      await user.click(await screen.findByTestId("callback-action-edit"));

      expect(lastUrl(onUrlUpdate)?.searchParams.get("callback")).toBe("langfuse");
      expect(lastUrl(onUrlUpdate)?.searchParams.get("callback_mode")).toBe("failure");
      expect(lastUrl(onUrlUpdate)?.options.history).toBe("push");
      const dialog = await screen.findByRole("dialog", { name: "Edit Callback Settings" });
      await waitFor(() => expect(within(dialog).getByLabelText("Host")).toHaveValue("https://failure.langfuse.com"));
    });

    it("clears the callback from the URL when the edit dialog is cancelled", async () => {
      mockLangfuseRows();
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<Settings {...defaultProps} />, {
        searchParams: "?callback=langfuse&callback_mode=failure",
        onUrlUpdate,
      });

      const dialog = await screen.findByRole("dialog", { name: "Edit Callback Settings" });
      await user.click(within(dialog).getByRole("button", { name: "Cancel" }));

      expect(lastUrl(onUrlUpdate)?.searchParams.has("callback")).toBe(false);
      expect(lastUrl(onUrlUpdate)?.searchParams.has("callback_mode")).toBe(false);
      await waitFor(() =>
        expect(screen.queryByRole("dialog", { name: "Edit Callback Settings" })).not.toBeInTheDocument(),
      );
    });

    it("clears the callback from the URL after a successful save", async () => {
      mockLangfuseRows();
      vi.mocked(setCallbacksCall).mockResolvedValue({});
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<Settings {...defaultProps} />, { searchParams: "?callback=langfuse", onUrlUpdate });

      const dialog = await screen.findByRole("dialog", { name: "Edit Callback Settings" });
      await waitFor(() => expect(within(dialog).getByLabelText("Host")).toHaveValue("https://success.langfuse.com"));
      await user.click(within(dialog).getByRole("button", { name: "Save Changes" }));

      await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.has("callback")).toBe(false));
      expect(vi.mocked(setCallbacksCall)).toHaveBeenCalledTimes(1);
    });

    it.each([
      ["an unknown callback", "?callback=missing&tab=email-alerts"],
      [
        "a mode the callback is not registered for",
        "?callback=langfuse&callback_mode=success_and_failure&tab=email-alerts",
      ],
      ["a read-only callback", "?callback=datadog&tab=email-alerts"],
    ])("drops a link to %s once callbacks load", async (_label, searchParams) => {
      mockLangfuseRows();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderRetainingMountWrites(searchParams, onUrlUpdate);

      await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
      expect(lastUrl(onUrlUpdate)?.searchParams.has("callback")).toBe(false);
      expect(lastUrl(onUrlUpdate)?.searchParams.has("callback_mode")).toBe(false);
      expect(lastUrl(onUrlUpdate)?.searchParams.get("tab")).toBe("email-alerts");
      expect(lastUrl(onUrlUpdate)?.options.history).toBe("replace");
      expect(screen.queryByRole("dialog", { name: "Edit Callback Settings" })).not.toBeInTheDocument();
    });

    it("keeps a callback link while callbacks are still loading", async () => {
      mockGetCallbacksCall.mockReturnValue(new Promise(() => {}));
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderRetainingMountWrites("?callback=langfuse", onUrlUpdate);

      await new Promise((resolve) => setTimeout(resolve, 100));
      expect(onUrlUpdate).not.toHaveBeenCalled();
    });
  });

  it("should display CloudZero Cost Tracking tab", async () => {
    renderWithProviders(<Settings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Active Logging Callbacks")).toBeInTheDocument();
    });

    expect(screen.getByText("CloudZero Cost Tracking")).toBeInTheDocument();
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
