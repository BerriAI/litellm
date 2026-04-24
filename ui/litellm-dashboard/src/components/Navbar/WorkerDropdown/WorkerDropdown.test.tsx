import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";

const mockUseWorker = vi.fn();
vi.mock("@/hooks/useWorker", () => ({
  useWorker: () => mockUseWorker(),
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

  it("should render nothing when isControlPlane is false", () => {
    mockUseWorker.mockReturnValue({
      isControlPlane: false,
      selectedWorker: workers[0],
      workers,
    });

    const { container } = render(<WorkerDropdown onWorkerSwitch={mockOnWorkerSwitch} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("should render nothing when selectedWorker is null", () => {
    mockUseWorker.mockReturnValue({
      isControlPlane: true,
      selectedWorker: null,
      workers,
    });

    const { container } = render(<WorkerDropdown onWorkerSwitch={mockOnWorkerSwitch} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("should render the select trigger when isControlPlane and selectedWorker exist", () => {
    mockUseWorker.mockReturnValue({
      isControlPlane: true,
      selectedWorker: workers[0],
      workers,
    });

    render(<WorkerDropdown onWorkerSwitch={mockOnWorkerSwitch} />);
    expect(screen.getByRole("combobox")).toBeInTheDocument();
  });

  it("should display the currently selected worker name in the trigger", () => {
    mockUseWorker.mockReturnValue({
      isControlPlane: true,
      selectedWorker: workers[1],
      workers,
    });

    render(<WorkerDropdown onWorkerSwitch={mockOnWorkerSwitch} />);
    expect(screen.getByRole("combobox")).toHaveTextContent("Worker 2");
  });

  it("should render all worker options when opened", async () => {
    mockUseWorker.mockReturnValue({
      isControlPlane: true,
      selectedWorker: workers[0],
      workers,
    });

    const user = userEvent.setup();
    render(<WorkerDropdown onWorkerSwitch={mockOnWorkerSwitch} />);
    await user.click(screen.getByRole("combobox"));

    expect(await screen.findByRole("option", { name: "Worker 1" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Worker 2" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Worker 3" })).toBeInTheDocument();
  });

  it("should disable the currently selected worker in the options list", async () => {
    mockUseWorker.mockReturnValue({
      isControlPlane: true,
      selectedWorker: workers[0],
      workers,
    });

    const user = userEvent.setup();
    render(<WorkerDropdown onWorkerSwitch={mockOnWorkerSwitch} />);
    await user.click(screen.getByRole("combobox"));

    const selectedOption = await screen.findByRole("option", { name: "Worker 1" });
    expect(selectedOption).toHaveAttribute("data-disabled");
  });

  it("should call onWorkerSwitch when a different worker is picked", async () => {
    mockUseWorker.mockReturnValue({
      isControlPlane: true,
      selectedWorker: workers[0],
      workers,
    });

    const user = userEvent.setup();
    render(<WorkerDropdown onWorkerSwitch={mockOnWorkerSwitch} />);
    await user.click(screen.getByRole("combobox"));

    const option = await screen.findByRole("option", { name: "Worker 2" });
    await user.click(option);

    expect(mockOnWorkerSwitch).toHaveBeenCalledWith("w2");
  });
});
