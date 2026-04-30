import { renderWithProviders, screen } from "../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import UsageExportHeader from "./UsageExportHeader";
import type { EntitySpendData } from "./types";

vi.mock("./EntityUsageExportModal", () => ({
  default: ({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) =>
    isOpen ? (
      <div data-testid="export-modal">
        <button onClick={onClose}>Close</button>
      </div>
    ) : null,
}));

const defaultProps = {
  dateValue: { from: new Date("2025-01-01"), to: new Date("2025-01-31") },
  entityType: "team" as const,
  spendData: {
    results: [],
    metadata: {
      total_spend: 0,
      total_api_requests: 0,
      total_successful_requests: 0,
      total_failed_requests: 0,
      total_tokens: 0,
    },
  } satisfies EntitySpendData,
};

describe("UsageExportHeader", () => {
  it("should render", () => {
    renderWithProviders(<UsageExportHeader {...defaultProps} />);
    expect(screen.getByRole("button", { name: /export data/i })).toBeInTheDocument();
  });

  it("should open the export modal when the export button is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<UsageExportHeader {...defaultProps} />);
    await user.click(screen.getByRole("button", { name: /export data/i }));
    expect(screen.getByTestId("export-modal")).toBeInTheDocument();
  });

  it("should close the export modal when onClose is called", async () => {
    const user = userEvent.setup();
    renderWithProviders(<UsageExportHeader {...defaultProps} />);
    await user.click(screen.getByRole("button", { name: /export data/i }));
    await user.click(screen.getByRole("button", { name: /close/i }));
    expect(screen.queryByTestId("export-modal")).not.toBeInTheDocument();
  });

  it("should not show filter dropdown when showFilters is false", () => {
    renderWithProviders(<UsageExportHeader {...defaultProps} showFilters={false} />);
    expect(screen.queryByText(/filter/i)).not.toBeInTheDocument();
  });

  it("should show filter dropdown when showFilters is true and options provided", () => {
    renderWithProviders(
      <UsageExportHeader
        {...defaultProps}
        showFilters
        filterLabel="Team"
        filterPlaceholder="Select teams"
        filterOptions={[
          { label: "Team A", value: "team-a" },
          { label: "Team B", value: "team-b" },
        ]}
        onFiltersChange={vi.fn()}
      />,
    );
    expect(screen.getByText("Team")).toBeInTheDocument();
  });
});
