/**
 * Tests for EntityUsageExportModal component
 *
 * Validates core export functionality:
 * - Renders modal with correct default state (CSV format, daily scope)
 * - User can select export type (daily vs daily_with_models)
 * - User can switch format (CSV vs JSON)
 * - Export button triggers data generation with correct parameters
 * - Modal closes after successful export
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import EntityUsageExportModal from "./EntityUsageExportModal";

// Mock utilities that format/export data so tests stay fast and deterministic
vi.mock("./utils", () => {
  return {
    handleExportCSV: vi.fn(),
    handleExportJSON: vi.fn(),
    generateExportData: vi.fn(() => [{ Date: "2025-10-01" }]),
    generateMetadata: vi.fn(() => ({ meta: true })),
  };
});

// Mock notifications
vi.mock("../molecules/notifications_manager", () => {
  return {
    default: {
      success: vi.fn(),
      fromBackend: vi.fn(),
      info: vi.fn(),
    },
  };
});

// Mock useTeams hook
vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useTeams: vi.fn(() => ({
    data: [],
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  })),
}));

// JSDOM stubs for download flow used by the modal
// @ts-ignore
global.URL.createObjectURL = vi.fn(() => "blob:mock");
// @ts-ignore
global.URL.revokeObjectURL = vi.fn();

describe("EntityUsageExportModal", () => {
  const baseProps = {
    isOpen: true,
    onClose: vi.fn(),
    entityType: "tag" as const,
    spendData: {
      results: [],
      metadata: {
        total_spend: 0,
        total_api_requests: 0,
        total_successful_requests: 0,
        total_failed_requests: 0,
        total_tokens: 0,
      },
    },
    dateRange: { from: new Date("2025-10-01"), to: new Date("2025-10-14") },
    selectedFilters: [],
    customTitle: "Export Tag Usage",
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders default state and exports CSV (daily) successfully", async () => {
    /**
     * Tests the happy path: user opens modal and exports with defaults.
     * Verifies that handleExportCSV is called with correct parameters
     * and modal closes after export completes.
     */
    const user = userEvent.setup();
    const { handleExportCSV } = await import("./utils");

    const { getByRole } = renderWithProviders(<EntityUsageExportModal {...baseProps} />);

    // Default primary action reflects CSV export
    expect(getByRole("button", { name: /Export CSV/i })).toBeInTheDocument();

    // Click export
    await user.click(getByRole("button", { name: /Export CSV/i }));

    // Verifies export function was invoked with correct parameters
    expect(handleExportCSV).toHaveBeenCalledWith(baseProps.spendData, "daily", "Tag", "tag", {});

    // Modal closes after export
    expect(baseProps.onClose).toHaveBeenCalled();
  });

  it("exports with 'day-by-day by tag and model' scope when selected", async () => {
    /**
     * Tests that user can change export type (scope).
     * Verifies handleExportCSV receives 'daily_with_models' scope
     * when the second radio option is selected.
     */
    const user = userEvent.setup();
    const { handleExportCSV } = await import("./utils");

    const { getByText, getByRole } = renderWithProviders(<EntityUsageExportModal {...baseProps} />);

    // Choose the alternate export type - click the label to trigger radio
    const dailyModelLabel = getByText(/Day-by-day by tag and model/i);
    await user.click(dailyModelLabel);

    // Export with default CSV format
    const exportBtn = getByRole("button", { name: /Export CSV/i });
    await user.click(exportBtn);

    // Ensure the selected scope flowed through
    expect(handleExportCSV).toHaveBeenCalledWith(baseProps.spendData, "daily_with_models", "Tag", "tag", {});

    // Modal closes after export
    expect(baseProps.onClose).toHaveBeenCalled();
  });
});
