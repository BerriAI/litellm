import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import Fallbacks from "./fallbacks";
import { getCallbacksCall, setCallbacksCall } from "./networking";

vi.mock("./networking", () => ({
  getCallbacksCall: vi.fn(),
  setCallbacksCall: vi.fn(),
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
          create: vi.fn().mockResolvedValue({
            model: "test-model",
          }),
        },
      },
    })),
  },
}));

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
    mockSetCallbacksCall.mockResolvedValue({});
  });

  it("should render", async () => {
    render(<Fallbacks {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByRole("columnheader", { name: "Model Name" })).toBeInTheDocument();
    });
  });

  it("should render fallback data when callback data is returned from network call", async () => {
    const mockFallbackData = {
      router_settings: {
        fallbacks: [{ "xai/grok-2": ["xai/grok-4", "gpt-4"] }, { "gpt-3.5-turbo": ["gpt-4"] }],
      },
    };

    mockGetCallbacksCall.mockResolvedValue(mockFallbackData);

    render(<Fallbacks {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("xai/grok-2")).toBeInTheDocument();
      expect(screen.getByText("xai/grok-4, gpt-4")).toBeInTheDocument();
      expect(screen.getByText("gpt-3.5-turbo")).toBeInTheDocument();
      expect(screen.getByText("gpt-4")).toBeInTheDocument();
    });

    expect(mockGetCallbacksCall).toHaveBeenCalledWith(
      defaultProps.accessToken,
      defaultProps.userID,
      defaultProps.userRole,
    );
  });

  it("should render AddFallbacks component", async () => {
    render(<Fallbacks {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Mock Add Fallbacks")).toBeInTheDocument();
    });
  });

  it("should not render when access token is not provided", () => {
    const { container } = render(<Fallbacks {...defaultProps} accessToken={null} />);
    expect(container.firstChild).toBeNull();
  });
});
