import { renderWithProviders, screen } from "../../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import { vi, describe, it, expect, beforeEach } from "vitest";
import WorkerDropdown from "./WorkerDropdown";

const mockUseWorker = vi.fn();

vi.mock("@/hooks/useWorker", () => ({
  useWorker: () => mockUseWorker(),
}));

const workers = [
  { worker_id: "w1", name: "Worker 1", url: "http://w1" },
  { worker_id: "w2", name: "Worker 2", url: "http://w2" },
  { worker_id: "w3", name: "Worker 3", url: "http://w3" },
];

describe("WorkerDropdown", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("should render", () => {
    mockUseWorker.mockReturnValue({
      isControlPlane: true,
      selectedWorker: workers[0],
      workers,
    });

    renderWithProviders(<WorkerDropdown onWorkerSwitch={vi.fn()} />);
    expect(screen.getByRole("combobox")).toBeInTheDocument();
  });

  it("should return null when not a control plane", () => {
    mockUseWorker.mockReturnValue({
      isControlPlane: false,
      selectedWorker: null,
      workers: [],
    });

    const { container } = renderWithProviders(
      <WorkerDropdown onWorkerSwitch={vi.fn()} />
    );
    expect(container.firstChild).toBeNull();
  });

  it("should return null when there is no selected worker", () => {
    mockUseWorker.mockReturnValue({
      isControlPlane: true,
      selectedWorker: null,
      workers,
    });

    const { container } = renderWithProviders(
      <WorkerDropdown onWorkerSwitch={vi.fn()} />
    );
    expect(container.firstChild).toBeNull();
  });

  it("should call onWorkerSwitch when a different worker is selected", async () => {
    mockUseWorker.mockReturnValue({
      isControlPlane: true,
      selectedWorker: workers[0],
      workers,
    });
    const onWorkerSwitch = vi.fn();
    const user = userEvent.setup();

    renderWithProviders(<WorkerDropdown onWorkerSwitch={onWorkerSwitch} />);

    // Open the dropdown
    await user.click(screen.getByRole("combobox"));
    // Select Worker 2
    await user.click(await screen.findByText("Worker 2"));

    expect(onWorkerSwitch).toHaveBeenCalledWith("w2");
  });
});
