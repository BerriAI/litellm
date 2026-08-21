import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";

const mockUseWorker = vi.fn();
vi.mock("@/hooks/useWorker", () => ({
  useWorker: () => mockUseWorker(),
}));

import WorkerDropdown from "./WorkerDropdown";

async function openWorkerList(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole("combobox"));
  await waitFor(() => {
    expect(screen.getByRole("combobox")).toHaveAttribute("aria-expanded", "true");
  });
}

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

  it("renders a collapsed worker combobox when isControlPlane and selectedWorker exist", () => {
    mockUseWorker.mockReturnValue({
      isControlPlane: true,
      selectedWorker: workers[1],
      workers,
    });

    render(<WorkerDropdown onWorkerSwitch={mockOnWorkerSwitch} />);
    expect(screen.getByRole("combobox")).toHaveAttribute("aria-expanded", "false");
  });

  it("reveals every worker only once the combobox is opened", async () => {
    mockUseWorker.mockReturnValue({
      isControlPlane: true,
      selectedWorker: workers[1],
      workers,
    });
    const user = userEvent.setup();

    render(<WorkerDropdown onWorkerSwitch={mockOnWorkerSwitch} />);
    expect(screen.queryAllByRole("option")).toHaveLength(0);
    expect(screen.queryByText("Worker 1")).not.toBeInTheDocument();
    expect(screen.queryByText("Worker 3")).not.toBeInTheDocument();

    await openWorkerList(user);

    await waitFor(() => {
      expect(screen.getByText("Worker 1")).toBeInTheDocument();
    });
    expect(screen.getAllByText("Worker 2").length).toBeGreaterThan(0);
    expect(screen.getByText("Worker 3")).toBeInTheDocument();
  });

  it("marks exactly one option as selected, the current worker", async () => {
    mockUseWorker.mockReturnValue({
      isControlPlane: true,
      selectedWorker: workers[1],
      workers,
    });
    const user = userEvent.setup();

    render(<WorkerDropdown onWorkerSwitch={mockOnWorkerSwitch} />);
    await openWorkerList(user);

    await waitFor(() => {
      const selected = screen.getAllByRole("option").filter((o) => o.getAttribute("aria-selected") === "true");
      expect(selected).toHaveLength(1);
      expect(selected[0]).toHaveAccessibleName("Worker 2");
    });
  });

  it("calls onWorkerSwitch with the id of the worker that was picked", async () => {
    mockUseWorker.mockReturnValue({
      isControlPlane: true,
      selectedWorker: workers[1],
      workers,
    });
    const user = userEvent.setup();

    render(<WorkerDropdown onWorkerSwitch={mockOnWorkerSwitch} />);
    await openWorkerList(user);
    await waitFor(() => {
      expect(screen.getByText("Worker 3")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText("Worker 3"));

    expect(mockOnWorkerSwitch).toHaveBeenCalledWith("w3");
  });

  it("does not call onWorkerSwitch when the already-current worker is picked", async () => {
    mockUseWorker.mockReturnValue({
      isControlPlane: true,
      selectedWorker: workers[1],
      workers,
    });
    const user = userEvent.setup();

    render(<WorkerDropdown onWorkerSwitch={mockOnWorkerSwitch} />);
    await openWorkerList(user);
    await waitFor(() => {
      expect(screen.getByText("Worker 3")).toBeInTheDocument();
    });

    for (const currentWorkerNode of screen.getAllByText("Worker 2")) {
      fireEvent.click(currentWorkerNode);
    }

    expect(mockOnWorkerSwitch).not.toHaveBeenCalled();
  });

  it("filters the worker options by the typed search text", async () => {
    mockUseWorker.mockReturnValue({
      isControlPlane: true,
      selectedWorker: workers[1],
      workers,
    });
    const user = userEvent.setup();

    render(<WorkerDropdown onWorkerSwitch={mockOnWorkerSwitch} />);
    await openWorkerList(user);
    await waitFor(() => {
      expect(screen.getByText("Worker 1")).toBeInTheDocument();
    });

    await user.clear(screen.getByRole("combobox"));
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "worker 3" } });

    await waitFor(() => {
      expect(screen.queryByText("Worker 1")).not.toBeInTheDocument();
    });
    expect(screen.getByText("Worker 3")).toBeInTheDocument();
  });
});
