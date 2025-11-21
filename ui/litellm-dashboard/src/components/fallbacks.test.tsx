import { render, waitFor } from "@testing-library/react";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import Fallbacks from "./fallbacks";
import { getCallbacksCall, setCallbacksCall } from "./networking";

vi.mock("./networking", () => ({
  getCallbacksCall: vi.fn(),
  setCallbacksCall: vi.fn(),
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

vi.mock("./add_fallbacks", () => ({
  __esModule: true,
  default: () => <div>Mock Add Fallbacks</div>,
}));

vi.mock("openai", () => ({
  default: {
    OpenAI: vi.fn().mockImplementation(() => ({
      chat: {
        completions: {
          create: vi.fn(),
        },
      },
    })),
  },
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

describe("Fallbacks", () => {
  const defaultProps = {
    accessToken: "token",
    userRole: "admin",
    userID: "user-123",
    modelData: { data: [] },
  };
  const mockGetCallbacksCall = vi.mocked(getCallbacksCall);
  const mockSetCallbacksCall = vi.mocked(setCallbacksCall);

  beforeEach(() => {
    vi.clearAllMocks();
    mockGetCallbacksCall.mockResolvedValue({
      router_settings: {
        fallbacks: [],
      },
    });
  });

  it("should render an empty table with headers when access token is provided", async () => {
    const { getByText } = render(<Fallbacks {...defaultProps} />);

    await waitFor(() => {
      expect(getByText("Model Name")).toBeInTheDocument();
      expect(getByText("Fallbacks")).toBeInTheDocument();
      expect(getByText("Actions")).toBeInTheDocument();
    });
  });

  it("should render fallback data when callback data is returned from network call", async () => {
    const mockFallbackData = {
      router_settings: {
        fallbacks: [{ "xai/grok-2": ["xai/grok-4", "gpt-4"] }, { "gpt-3.5-turbo": ["gpt-4"] }],
      },
    };

    mockGetCallbacksCall.mockResolvedValue(mockFallbackData);

    const { getByText } = render(<Fallbacks {...defaultProps} />);

    await waitFor(() => {
      expect(getByText("xai/grok-2")).toBeInTheDocument();
      expect(getByText("xai/grok-4, gpt-4")).toBeInTheDocument();
      expect(getByText("gpt-3.5-turbo")).toBeInTheDocument();
      expect(getByText("gpt-4")).toBeInTheDocument();
    });

    expect(mockGetCallbacksCall).toHaveBeenCalledWith(
      defaultProps.accessToken,
      defaultProps.userID,
      defaultProps.userRole,
    );
  });
});
