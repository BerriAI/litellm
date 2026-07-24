import { describe, it, expect, vi, beforeAll, beforeEach } from "vitest";
import { render, screen, waitFor, within, fireEvent } from "@testing-library/react";
import { flexRender, getCoreRowModel, useReactTable } from "@tanstack/react-table";
import PublicModelHub from "./public_model_hub";
import { getPublicMCPHubColumns, MCPServerData } from "./PublicModelHubTableColumns";

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
    skillHubPublicCall: vi.fn().mockResolvedValue({ plugins: [] }),
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

    // Check the health status badge in each model's row
    await waitFor(() => {
      const gpt4Row = screen.getByText("gpt-4").closest("tr");
      expect(gpt4Row).toBeInTheDocument();
      expect(within(gpt4Row as HTMLElement).getByText("healthy")).toBeInTheDocument();
    });

    await waitFor(() => {
      const claude3Row = screen.getByText("claude-3").closest("tr");
      expect(claude3Row).toBeInTheDocument();
      expect(within(claude3Row as HTMLElement).getByText("unhealthy")).toBeInTheDocument();
    });

    await waitFor(() => {
      const gpt35Row = screen.getByText("gpt-3.5-turbo").closest("tr");
      expect(gpt35Row).toBeInTheDocument();
      expect(within(gpt35Row as HTMLElement).getByText("Unknown")).toBeInTheDocument();
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

const PUBLIC_SERVER_URL = "https://mcp.exa.ai/mcp";

const mockMcpServer: MCPServerData = {
  server_id: "server-1",
  name: "exa_test",
  server_name: "exa_test",
  url: PUBLIC_SERVER_URL,
  transport: "http",
  auth_type: "none",
  mcp_info: { server_name: "exa_test", description: "Fast, intelligent web search and web crawling" },
};

function PublicMcpTestTable({ data }: { data: MCPServerData[] }) {
  const columns = getPublicMCPHubColumns({ onServerClick: vi.fn() });
  const table = useReactTable({ data, columns, getCoreRowModel: getCoreRowModel() });

  return (
    <table>
      <thead>
        {table.getHeaderGroups().map((hg) => (
          <tr key={hg.id}>
            {hg.headers.map((h) => (
              <th key={h.id}>{flexRender(h.column.columnDef.header, h.getContext())}</th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody>
        {table.getRowModel().rows.map((row) => (
          <tr key={row.id}>
            {row.getVisibleCells().map((cell) => (
              <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

describe("publicMCPHubColumns", () => {
  it("keeps the non-sensitive columns", () => {
    render(<PublicMcpTestTable data={[mockMcpServer]} />);
    expect(screen.getByText("Server Name")).toBeInTheDocument();
    expect(screen.getByText("Transport")).toBeInTheDocument();
    expect(screen.getByText("Auth Type")).toBeInTheDocument();
  });

  it("does not expose a URL column header", () => {
    render(<PublicMcpTestTable data={[mockMcpServer]} />);
    expect(screen.queryByText("URL")).not.toBeInTheDocument();
    const columns = getPublicMCPHubColumns({ onServerClick: vi.fn() });
    expect(columns.some((c) => c.header === "URL" || c.meta?.title === "URL")).toBe(false);
  });

  it("does not render the server url anywhere in the table", () => {
    render(<PublicMcpTestTable data={[mockMcpServer]} />);
    expect(screen.queryByText(PUBLIC_SERVER_URL)).not.toBeInTheDocument();
  });
});

describe("public hub MCP details modal", () => {
  it("does not show the upstream url when a server is opened", async () => {
    const networkingModule = await import("./networking");
    vi.mocked(networkingModule.mcpHubPublicServersCall).mockResolvedValue([mockMcpServer]);

    render(<PublicModelHub />);

    fireEvent.click(await screen.findByRole("tab", { name: /MCP Hub/i }));
    fireEvent.click(await screen.findByRole("button", { name: "exa_test" }));

    // "Server Overview" only exists inside the opened MCP details modal,
    // so finding it proves the modal rendered and the url assertion is not vacuous.
    await screen.findByText("Server Overview");
    expect(screen.queryByText(PUBLIC_SERVER_URL)).not.toBeInTheDocument();
  });
});
