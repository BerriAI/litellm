import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import { flexRender, getCoreRowModel, useReactTable } from "@tanstack/react-table";
import { getAgentHubTableColumns, AgentHubData } from "./AgentHubTableColumns";

const mockAgent: AgentHubData = {
  agent_id: "agent-1",
  protocolVersion: "1.0",
  name: "Test Agent",
  description: "A test agent for unit testing",
  url: "https://agent.example.com",
  version: "2.0",
  capabilities: { streaming: true, caching: false },
  defaultInputModes: ["text"],
  defaultOutputModes: ["text", "image"],
  skills: [
    { id: "s1", name: "Skill One", description: "First skill" },
    { id: "s2", name: "Skill Two", description: "Second skill" },
    { id: "s3", name: "Skill Three", description: "Third skill" },
  ],
  is_public: true,
};

function TestTable({
  data,
  publicPage = false,
  showModal = vi.fn(),
  copyToClipboard = vi.fn(),
}: {
  data: AgentHubData[];
  publicPage?: boolean;
  showModal?: ReturnType<typeof vi.fn>;
  copyToClipboard?: ReturnType<typeof vi.fn>;
}) {
  const columns = getAgentHubTableColumns(showModal, copyToClipboard, publicPage);
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

describe("AgentHubTableColumns", () => {
  it("should render", () => {
    render(<TestTable data={[mockAgent]} />);
    expect(screen.getByText("Test Agent")).toBeInTheDocument();
  });

  it("should display the agent description", () => {
    render(<TestTable data={[mockAgent]} />);
    // Description appears in both the description column and the mobile view within agent name column
    expect(screen.getAllByText("A test agent for unit testing").length).toBeGreaterThanOrEqual(1);
  });

  it("should display the version with a 'v' prefix", () => {
    render(<TestTable data={[mockAgent]} />);
    expect(screen.getByText("v2.0")).toBeInTheDocument();
  });

  it("should display the protocol version", () => {
    render(<TestTable data={[mockAgent]} />);
    expect(screen.getByText("1.0")).toBeInTheDocument();
  });

  it("should show skill count with correct pluralization", () => {
    render(<TestTable data={[mockAgent]} />);
    expect(screen.getByText("3 skills")).toBeInTheDocument();
  });

  it("should show first two skills and '+1' for overflow", () => {
    render(<TestTable data={[mockAgent]} />);
    expect(screen.getByText("Skill One")).toBeInTheDocument();
    expect(screen.getByText("Skill Two")).toBeInTheDocument();
    expect(screen.getByText("+1")).toBeInTheDocument();
  });

  it("should show only true capabilities as badges", () => {
    render(<TestTable data={[mockAgent]} />);
    expect(screen.getByText("streaming")).toBeInTheDocument();
    expect(screen.queryByText("caching")).not.toBeInTheDocument();
  });

  it("should display I/O modes", () => {
    render(<TestTable data={[mockAgent]} />);
    // "In:" and "Out:" are in <span> children; getByText with exact:false
    // matches against the element's full textContent across child nodes
    expect(screen.getByText((_, el) =>
      el?.tagName === "P" && el.textContent === "In: text"
    )).toBeInTheDocument();
    expect(screen.getByText((_, el) =>
      el?.tagName === "P" && el.textContent === "Out: text, image"
    )).toBeInTheDocument();
  });

  it("should display 'Yes' badge for public agents", () => {
    render(<TestTable data={[mockAgent]} />);
    expect(screen.getByText("Yes")).toBeInTheDocument();
  });

  it("should display 'No' badge for non-public agents", () => {
    const privateAgent = { ...mockAgent, is_public: false };
    render(<TestTable data={[privateAgent]} />);
    expect(screen.getByText("No")).toBeInTheDocument();
  });

  it("should display a Details button", () => {
    render(<TestTable data={[mockAgent]} />);
    expect(screen.getByRole("button", { name: /details|info/i })).toBeInTheDocument();
  });

  it("should show '-' when agent has no capabilities", () => {
    const noCapAgent = { ...mockAgent, capabilities: {} };
    render(<TestTable data={[noCapAgent]} />);
    // The dash is rendered in the capabilities column
    expect(screen.getByText("-")).toBeInTheDocument();
  });

  it("should show singular 'skill' for one skill", () => {
    const oneSkillAgent = {
      ...mockAgent,
      skills: [{ id: "s1", name: "Only Skill", description: "One" }],
    };
    render(<TestTable data={[oneSkillAgent]} />);
    expect(screen.getByText("1 skill")).toBeInTheDocument();
  });
});
