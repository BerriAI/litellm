import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import PaginationStatusAlerts from "./PaginationStatusAlerts";

describe("PaginationStatusAlerts", () => {
  it("shows page progress and wires the Stop button while fetching", () => {
    const cancel = vi.fn();
    render(
      <PaginationStatusAlerts
        isFetchingMore={true}
        cancelled={false}
        progress={{ currentPage: 7, totalPages: 42 }}
        cancel={cancel}
      />,
    );

    expect(screen.getByText(/Currently fetching spend data: fetched 7 \/ 42 pages/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Stop" }));
    expect(cancel).toHaveBeenCalledTimes(1);
  });

  it("shows the partial-data notice after a cancel, frozen at the last fetched page", () => {
    render(
      <PaginationStatusAlerts
        isFetchingMore={false}
        cancelled={true}
        progress={{ currentPage: 7, totalPages: 42 }}
        cancel={vi.fn()}
      />,
    );

    expect(screen.getByText("Showing partial spend data (7/42 pages loaded)")).toBeInTheDocument();
  });

  it("names the subject it is fetching", () => {
    render(
      <PaginationStatusAlerts
        isFetchingMore={true}
        cancelled={false}
        progress={{ currentPage: 1, totalPages: 3 }}
        cancel={vi.fn()}
        subject="agent data"
      />,
    );

    expect(screen.getByText(/Currently fetching agent data: fetched 1 \/ 3 pages/)).toBeInTheDocument();
  });

  it("renders nothing when idle", () => {
    const { container } = render(
      <PaginationStatusAlerts
        isFetchingMore={false}
        cancelled={false}
        progress={{ currentPage: 1, totalPages: 1 }}
        cancel={vi.fn()}
      />,
    );

    expect(container).toBeEmptyDOMElement();
  });
});
