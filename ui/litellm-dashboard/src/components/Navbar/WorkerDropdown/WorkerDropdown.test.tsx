import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";

// Mock the useWorker hook
const mockUseWorker = vi.fn();
vi.mock("@/hooks/useWorker", () => ({
  useWorker: () => mockUseWorker(),
}));

// Mock antd Select
vi.mock("antd", () => ({
  Select: ({ value, options, onChange, style, disabled, ...props }: any) => (
    <select
      data-testid="worker-select"
      value={value}
      style={style}
      onChange={(e) => onChange?.(e.target.value)}
    >
      {options?.map((opt: any) => (
        <option key={opt.value} value={opt.value} disabled={opt.disabled}>
          {opt.label}
        </option>
      ))}
    </select>
  ),
}));

// Mock icon
vi.mock("@ant-design/icons", () => ({
  CloudServerOutlined: () => <span data-testid="cloud-icon" />,
}));

import WorkerDropdown from "./WorkerDropdown";

describe("WorkerDropdown", () => {
  const mockOnWorkerSwitch = vi.fn();
  const workers = [
    { worker_id: "w1", name: "Worker 1" },
    { worker_id: "w2", name: "Worker 2" },
    { worker_id: "w3", name: "Worker 3" },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders null when isControlPlane is false", () => {
    mockUseWorker.mockReturnValue({
      isControlPlane: false,
      selectedWorker: workers[0],
      workers,
    });

    const { container } = render(<WorkerDropdown onWorkerSwitch={mockOnWorkerSwitch} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders null when selectedWorker is null", () => {
    mockUseWorker.mockReturnValue({
      isControlPlane: true,
      selectedWorker: null,
      workers,
    });

    const { container } = render(<WorkerDropdown onWorkerSwitch={mockOnWorkerSwitch} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders the select when isControlPlane and selectedWorker exist", () => {
    mockUseWorker.mockReturnValue({
      isControlPlane: true,
      selectedWorker: workers[0],
      workers,
    });

    render(<WorkerDropdown onWorkerSwitch={mockOnWorkerSwitch} />);
    expect(screen.getByTestId("worker-select")).toBeInTheDocument();
  });

  it("renders all worker options", () => {
    mockUseWorker.mockReturnValue({
      isControlPlane: true,
      selectedWorker: workers[0],
      workers,
    });

    render(<WorkerDropdown onWorkerSwitch={mockOnWorkerSwitch} />);
    expect(screen.getByText("Worker 1")).toBeInTheDocument();
    expect(screen.getByText("Worker 2")).toBeInTheDocument();
    expect(screen.getByText("Worker 3")).toBeInTheDocument();
  });

  it("sets current worker as selected value", () => {
    mockUseWorker.mockReturnValue({
      isControlPlane: true,
      selectedWorker: workers[1],
      workers,
    });

    render(<WorkerDropdown onWorkerSwitch={mockOnWorkerSwitch} />);
    const select = screen.getByTestId("worker-select") as HTMLSelectElement;
    expect(select.value).toBe("w2");
  });

  it("disables the currently selected worker in options", () => {
    mockUseWorker.mockReturnValue({
      isControlPlane: true,
      selectedWorker: workers[0],
      workers,
    });

    render(<WorkerDropdown onWorkerSwitch={mockOnWorkerSwitch} />);
    const options = screen.getAllByRole("option");
    const selectedOption = options.find((opt) => (opt as HTMLOptionElement).value === "w1");
    expect(selectedOption).toBeDisabled();
  });

  it("calls onWorkerSwitch when selection changes", async () => {
    mockUseWorker.mockReturnValue({
      isControlPlane: true,
      selectedWorker: workers[0],
      workers,
    });

    render(<WorkerDropdown onWorkerSwitch={mockOnWorkerSwitch} />);
    const select = screen.getByTestId("worker-select");

    const { default: userEvent } = await import("@testing-library/user-event");
    const user = userEvent.setup();
    await user.selectOptions(select, "w2");

    expect(mockOnWorkerSwitch).toHaveBeenCalledWith("w2");
  });
});
