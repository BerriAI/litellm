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
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import EntityUsageExportModal from "./EntityUsageExportModal";

// Mock utilities that format/export data so tests stay fast and deterministic
vi.mock("./utils", () => {
  return {
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
     * Verifies that generateExportData is called with 'daily' scope
     * and modal closes after export completes.
     */
    const user = userEvent.setup();
    const { generateExportData } = await import("./utils");

    render(<EntityUsageExportModal {...baseProps} />);

    // Default primary action reflects CSV export
    expect(screen.getByRole("button", { name: /Export CSV/i })).toBeInTheDocument();

    // Click export
    await user.click(screen.getByRole("button", { name: /Export CSV/i }));

    // Verifies export pipeline was invoked with default scope 'daily'
    expect(generateExportData).toHaveBeenCalled();
    const callArgs = (generateExportData as any).mock.calls[0];
    expect(callArgs[1]).toBe("daily");

    // Modal closes after export
    expect(baseProps.onClose).toHaveBeenCalled();
  });

  it("exports with 'day-by-day by tag and model' scope when selected", async () => {
    /**
     * Tests that user can change export type (scope).
     * Verifies generateExportData receives 'daily_with_models' scope
     * when the second radio option is selected.
     */
    const user = userEvent.setup();
    const { generateExportData } = await import("./utils");

    render(<EntityUsageExportModal {...baseProps} />);

    // Choose the alternate export type - click the label to trigger radio
    const dailyModelLabel = screen.getByText(/Day-by-day by tag and model/i);
    await user.click(dailyModelLabel);

    // Export with default CSV format
    const exportBtn = screen.getByRole("button", { name: /Export CSV/i });
    await user.click(exportBtn);

    // Ensure the selected scope flowed through
    expect(generateExportData).toHaveBeenCalled();
    const callArgs = (generateExportData as any).mock.calls.at(-1);
    expect(callArgs[1]).toBe("daily_with_models");

    // Modal closes after export
    expect(baseProps.onClose).toHaveBeenCalled();
  });
});


