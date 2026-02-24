import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../../tests/test-utils";
import UsageAIChatModal from "./UsageAIChatModal";

beforeAll(() => {
  if (typeof window !== "undefined" && !window.ResizeObserver) {
    window.ResizeObserver = class ResizeObserver {
      observe() {}
      unobserve() {}
      disconnect() {}
    } as any;
  }
});

vi.mock("../../networking", () => ({
  modelHubCall: vi.fn().mockResolvedValue({
    data: [
      { model_group: "gpt-4" },
      { model_group: "claude-3-opus" },
    ],
  }),
  getProxyBaseUrl: vi.fn().mockReturnValue("http://localhost:4000"),
}));

vi.mock("openai", () => {
  return {
    default: {
      OpenAI: vi.fn().mockImplementation(() => ({
        chat: {
          completions: {
            create: vi.fn(),
          },
        },
      })),
    },
  };
});

const mockUserSpendData = {
  results: [
    {
      date: "2025-01-01",
      metrics: {
        spend: 100.5,
        api_requests: 1000,
        successful_requests: 950,
        failed_requests: 50,
        total_tokens: 50000,
        prompt_tokens: 30000,
        completion_tokens: 20000,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      breakdown: {
        models: {
          "gpt-4": {
            metrics: {
              spend: 80.0,
              api_requests: 800,
              successful_requests: 780,
              failed_requests: 20,
              total_tokens: 40000,
              prompt_tokens: 24000,
              completion_tokens: 16000,
              cache_read_input_tokens: 0,
              cache_creation_input_tokens: 0,
            },
            metadata: {},
            api_key_breakdown: {},
          },
        },
        model_groups: {},
        mcp_servers: {},
        providers: {
          openai: {
            metrics: {
              spend: 100.5,
              api_requests: 1000,
              successful_requests: 950,
              failed_requests: 50,
              total_tokens: 50000,
              prompt_tokens: 30000,
              completion_tokens: 20000,
              cache_read_input_tokens: 0,
              cache_creation_input_tokens: 0,
            },
            metadata: {},
            api_key_breakdown: {},
          },
        },
        api_keys: {
          "sk-test": {
            metrics: {
              spend: 100.5,
              api_requests: 1000,
              successful_requests: 950,
              failed_requests: 50,
              total_tokens: 50000,
              prompt_tokens: 30000,
              completion_tokens: 20000,
              cache_read_input_tokens: 0,
              cache_creation_input_tokens: 0,
            },
            metadata: {
              key_alias: "Test Key",
              team_id: null,
            },
          },
        },
        entities: {},
      },
    },
  ],
  metadata: {
    total_spend: 100.5,
    total_api_requests: 1000,
    total_successful_requests: 950,
    total_failed_requests: 50,
    total_tokens: 50000,
  },
};

const defaultProps = {
  visible: true,
  onCancel: vi.fn(),
  accessToken: "test-token",
  userSpendData: mockUserSpendData,
  dateRange: {
    from: new Date("2025-01-01"),
    to: new Date("2025-01-07"),
  },
};

describe("UsageAIChatModal", () => {
  it("should render the modal when visible", () => {
    renderWithProviders(<UsageAIChatModal {...defaultProps} />);

    expect(screen.getByText("Ask AI about Usage")).toBeInTheDocument();
    expect(
      screen.getByText("Ask questions about your spend, models, API keys, and usage trends")
    ).toBeInTheDocument();
  });

  it("should render model selector", () => {
    renderWithProviders(<UsageAIChatModal {...defaultProps} />);

    expect(screen.getByText("Model")).toBeInTheDocument();
  });

  it("should render empty state message when no conversation", () => {
    renderWithProviders(<UsageAIChatModal {...defaultProps} />);

    expect(screen.getByText("Ask a question about your usage")).toBeInTheDocument();
  });

  it("should render the send button", () => {
    renderWithProviders(<UsageAIChatModal {...defaultProps} />);

    expect(screen.getByText("Send")).toBeInTheDocument();
  });

  it("should render input placeholder", () => {
    renderWithProviders(<UsageAIChatModal {...defaultProps} />);

    expect(screen.getByPlaceholderText("Ask about your usage...")).toBeInTheDocument();
  });

  it("should render clear conversation button", () => {
    renderWithProviders(<UsageAIChatModal {...defaultProps} />);

    expect(screen.getByText("Clear conversation")).toBeInTheDocument();
  });

  it("should not render when not visible", () => {
    renderWithProviders(<UsageAIChatModal {...defaultProps} visible={false} />);

    expect(screen.queryByText("Ask AI about Usage")).not.toBeInTheDocument();
  });

  it("should call onCancel when modal is closed", () => {
    const onCancel = vi.fn();
    renderWithProviders(<UsageAIChatModal {...defaultProps} onCancel={onCancel} />);

    const closeButtons = document.querySelectorAll(".ant-modal-close");
    if (closeButtons.length > 0) {
      act(() => {
        fireEvent.click(closeButtons[0]);
      });
      expect(onCancel).toHaveBeenCalled();
    }
  });
});
