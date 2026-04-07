import { act, fireEvent, render, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { alertingSettingsCall } from "./networking";
import Settings from "./settings";

vi.mock("./networking", () => ({
  getCallbacksCall: vi.fn(),
  getCallbackConfigsCall: vi.fn(),
  setCallbacksCall: vi.fn(),
  serviceHealthCheck: vi.fn(),
  deleteCallback: vi.fn(),
  alertingSettingsCall: vi.fn().mockResolvedValue([]),
}));

const mockCallbacksData = {
  callbacks: [] as any[],
  alerts: [] as any[],
  availableCallbacks: {} as Record<string, any>,
};

const mockCallbackConfigs: any[] = [];

vi.mock("@/app/(dashboard)/hooks/callbacks/useCallbacks", () => ({
  useCallbacks: () => ({
    data: mockCallbacksData,
    isLoading: false,
    isError: false,
  }),
  useCallbackConfigs: () => ({
    data: mockCallbackConfigs,
    isLoading: false,
    isError: false,
  }),
  callbackKeys: { all: ["callbacks"], lists: () => ["callbacks", "list"], list: () => ["callbacks", "list", {}] },
}));

const mockUpdateMutate = vi.fn();
const mockDeleteMutate = vi.fn();

vi.mock("@/app/(dashboard)/hooks/callbacks/useUpdateCallback", () => ({
  useUpdateCallback: () => ({
    mutate: mockUpdateMutate,
    isPending: false,
  }),
}));

vi.mock("@/app/(dashboard)/hooks/callbacks/useDeleteCallback", () => ({
  useDeleteCallback: () => ({
    mutate: mockDeleteMutate,
    isPending: false,
  }),
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

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe("Settings", () => {
  const defaultProps = {
    accessToken: "token",
    userRole: "admin",
    userID: "user-123",
    premiumUser: false,
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockCallbacksData.callbacks = [];
    mockCallbacksData.alerts = [];
    mockCallbacksData.availableCallbacks = {};
    mockCallbackConfigs.length = 0;
  });

  it("should render the logging callbacks tab when access token is provided", async () => {
    const { getByText } = render(<Settings {...defaultProps} />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(getByText("Active Logging Callbacks")).toBeInTheDocument();
    });
  });

  it("should display additional settings tabs", async () => {
    const { getByText } = render(<Settings {...defaultProps} />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(getByText("CloudZero Cost Tracking")).toBeInTheDocument();
      expect(getByText("Alerting Types")).toBeInTheDocument();
      expect(getByText("Alerting Settings")).toBeInTheDocument();
      expect(getByText("Email Alerts")).toBeInTheDocument();
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

    mockCallbacksData.callbacks = [mockCallback];
    mockCallbacksData.availableCallbacks = {
      langfuse: {
        litellm_callback_name: "langfuse",
        litellm_callback_params: ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"],
        ui_callback_name: "Langfuse",
      },
    };
    mockCallbackConfigs.push(mockCallbackConfig);

    const { getByText, container } = render(<Settings {...defaultProps} />, { wrapper: createWrapper() });

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
    const { getByText } = render(<Settings {...defaultProps} />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(getByText("Active Logging Callbacks")).toBeInTheDocument();
    });

    expect(getByText("CloudZero Cost Tracking")).toBeInTheDocument();
  });
});
