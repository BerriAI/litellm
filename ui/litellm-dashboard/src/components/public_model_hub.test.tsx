import { describe, it, expect, vi, beforeAll, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import PublicModelHub from "./public_model_hub";

vi.mock("next/navigation", () => ({
  useRouter: vi.fn(() => ({
    replace: vi.fn(),
    push: vi.fn(),
    refresh: vi.fn(),
  })),
}));

vi.mock("./networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./networking")>();
  return {
    ...actual,
    modelHubPublicModelsCall: vi.fn().mockResolvedValue([]),
    getPublicModelHubInfo: vi.fn().mockResolvedValue({
      docs_title: "LiteLLM Gateway",
      custom_docs_description: null,
      litellm_version: "1.0.0",
      useful_links: {},
    }),
    agentHubPublicModelsCall: vi.fn().mockResolvedValue([]),
    mcpHubPublicServersCall: vi.fn().mockResolvedValue([]),
    getUiConfig: vi.fn().mockResolvedValue({}),
  };
});

vi.mock("./navbar", () => ({
  default: vi.fn(() => <div data-testid="navbar">Navbar Component</div>),
}));

beforeAll(() => {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  });
});

beforeEach(() => {
  Storage.prototype.getItem = vi.fn(() => "false");
  Storage.prototype.setItem = vi.fn();
  Object.defineProperty(window, "location", {
    writable: true,
    value: {
      pathname: "/",
      origin: "http://localhost:3000",
    },
  });
});

describe("PublicModelHub", () => {
  it("renders", () => {
    const { container } = render(<PublicModelHub />);
    expect(container).toBeInTheDocument();
  });

  it("displays health status correctly for models with health check information", async () => {
    const mockModelsWithHealthChecks = [
      {
        model_group: "gpt-4",
        providers: ["openai"],
        mode: "chat",
        health_status: "healthy",
        health_response_time: 150.5,
        health_checked_at: "2024-01-15T10:30:00Z",
        supports_function_calling: true,
        supports_vision: false,
        supports_parallel_function_calling: false,
      },
      {
        model_group: "claude-3",
        providers: ["anthropic"],
        mode: "chat",
        health_status: "unhealthy",
        health_response_time: 5000.0,
        health_checked_at: "2024-01-15T10:25:00Z",
        supports_function_calling: true,
        supports_vision: false,
        supports_parallel_function_calling: false,
      },
      {
        model_group: "gpt-3.5-turbo",
        providers: ["openai"],
        mode: "chat",
        health_status: undefined,
        health_response_time: undefined,
        health_checked_at: undefined,
        supports_function_calling: false,
        supports_vision: false,
        supports_parallel_function_calling: false,
      },
    ];

    const networkingModule = await import("./networking");
    vi.mocked(networkingModule.modelHubPublicModelsCall).mockResolvedValue(mockModelsWithHealthChecks);

    render(<PublicModelHub />);

    // Wait for the component to load and render the table
    await waitFor(() => {
      expect(screen.getByText("gpt-4")).toBeInTheDocument();
    });

    // Check that health status is displayed for healthy model (gpt-4)
    // Find the row containing "gpt-4" and verify it has "healthy" status
    await waitFor(() => {
      const gpt4Cell = screen.getByText("gpt-4");
      const gpt4Row = gpt4Cell.closest("tr");
      expect(gpt4Row).toBeInTheDocument();

      // Find all cells in the row
      const cells = gpt4Row?.querySelectorAll("td");
      expect(cells).toBeTruthy();

      // Find the cell containing "healthy" text (health status column)
      // The health status is in a Tag component, so look for a Tag containing "healthy"
      const healthyStatus = Array.from(cells || []).find((cell) => {
        const tag = cell.querySelector('[class*="ant-tag"]');
        const text = tag?.textContent?.toLowerCase();
        return text === "healthy";
      });
      expect(healthyStatus).toBeInTheDocument();
    });

    // Check that health status is displayed for unhealthy model (claude-3)
    await waitFor(() => {
      const claude3Cell = screen.getByText("claude-3");
      const claude3Row = claude3Cell.closest("tr");
      expect(claude3Row).toBeInTheDocument();

      // Find all cells in the row
      const cells = claude3Row?.querySelectorAll("td");
      expect(cells).toBeTruthy();

      // Find the cell containing "unhealthy" text (health status column)
      const unhealthyStatus = Array.from(cells || []).find((cell) => {
        const tag = cell.querySelector('[class*="ant-tag"]');
        const text = tag?.textContent?.toLowerCase();
        return text === "unhealthy";
      });
      expect(unhealthyStatus).toBeInTheDocument();
    });

    // Check that "Unknown" is displayed for model without health status (gpt-3.5-turbo)
    await waitFor(() => {
      const gpt35Cell = screen.getByText("gpt-3.5-turbo");
      const gpt35Row = gpt35Cell.closest("tr");
      expect(gpt35Row).toBeInTheDocument();

      // Find all cells in the row
      const cells = gpt35Row?.querySelectorAll("td");
      expect(cells).toBeTruthy();

      // Find the cell containing "Unknown" text (health status column)
      const unknownStatus = Array.from(cells || []).find((cell) => {
        const tag = cell.querySelector('[class*="ant-tag"]');
        const text = tag?.textContent;
        return text === "Unknown";
      });
      expect(unknownStatus).toBeInTheDocument();
    });
  });
  it("handles non-array response gracefully (regression test for e.filter crash)", async () => {
    const networkingModule = await import("./networking");
    // Mock the API to return an object (like an error response) instead of an array
    vi.mocked(networkingModule.modelHubPublicModelsCall).mockResolvedValue({
      detail: "No models configured",
    } as any);

    render(<PublicModelHub />);

    await waitFor(() => {
      expect(screen.getByTestId("navbar")).toBeInTheDocument();
      expect(screen.getByText("Model Hub")).toBeInTheDocument();
    });
  });
});
