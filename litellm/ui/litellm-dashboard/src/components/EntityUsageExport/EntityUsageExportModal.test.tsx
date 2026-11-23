import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import EntityUsageExportModal from "./EntityUsageExportModal";

const downloadBlob = vi.fn();

vi.mock("./utils", () => ({
  downloadBlob: (...args: unknown[]) => downloadBlob(...args),
  exportFilename: vi.fn(() => "tag_usage_daily_2025-10-01_2025-10-14.csv"),
}));

describe("EntityUsageExportModal", () => {
  const baseProps = {
    isOpen: true,
    onClose: vi.fn(),
    entityType: "tag" as const,
    onExport: vi.fn().mockResolvedValue(new Blob(["data"])),
    dateRange: { from: new Date("2025-10-01"), to: new Date("2025-10-14") },
    selectedFilters: [],
    customTitle: "Export Tag Usage",
    canIncludeMetadata: true,
  };

  beforeEach(() => {
    vi.clearAllMocks();
    baseProps.onExport.mockResolvedValue(new Blob(["data"]));
  });

  it("exports through the onExport callback with the selected type and format, then closes", async () => {
    const user = userEvent.setup();

    renderWithProviders(<EntityUsageExportModal {...baseProps} />);

    expect(screen.getByRole("button", { name: /Export CSV/i })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Export CSV/i }));

    expect(baseProps.onExport).toHaveBeenCalledWith("daily", "csv", false);
    expect(downloadBlob).toHaveBeenCalledWith(expect.any(Blob), "tag_usage_daily_2025-10-01_2025-10-14.csv");
    expect(baseProps.onClose).toHaveBeenCalled();
  });

  it("forwards the export type the user picks", async () => {
    const user = userEvent.setup();

    renderWithProviders(<EntityUsageExportModal {...baseProps} />);

    await user.click(screen.getByRole("radio", { name: /Day-by-day breakdown by tag and key/i }));
    await user.click(screen.getByRole("button", { name: /Export CSV/i }));

    expect(baseProps.onExport).toHaveBeenCalledWith("daily_with_keys", "csv", false);
    expect(baseProps.onClose).toHaveBeenCalled();
  });

  it("asks for key and user metadata only when the checkbox is ticked", async () => {
    const user = userEvent.setup();

    renderWithProviders(<EntityUsageExportModal {...baseProps} />);

    const checkbox = screen.getByRole("checkbox", { name: /Include key and user metadata/i });
    expect(checkbox).not.toBeChecked();
    await user.click(checkbox);
    await user.click(screen.getByRole("button", { name: /Export CSV/i }));

    expect(baseProps.onExport).toHaveBeenCalledWith("daily", "csv", true);
  });

  it("warns next to the metadata checkbox that exported metadata may contain credentials", () => {
    renderWithProviders(<EntityUsageExportModal {...baseProps} />);

    expect(screen.getByRole("checkbox", { name: /Include key and user metadata/i })).toHaveAccessibleDescription(
      "Metadata is exported as stored and may contain credentials or other sensitive values. Known credential patterns are redacted, but not every secret can be caught.",
    );
  });

  it("hides the metadata option from callers who cannot export metadata and never asks for it", async () => {
    const user = userEvent.setup();

    renderWithProviders(<EntityUsageExportModal {...baseProps} canIncludeMetadata={false} />);

    expect(screen.queryByRole("checkbox", { name: /Include key and user metadata/i })).not.toBeInTheDocument();
    expect(screen.queryByText(/may contain credentials/i)).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Export CSV/i }));

    expect(baseProps.onExport).toHaveBeenCalledWith("daily", "csv", false);
  });

  it("keeps the modal open and reports the error when onExport rejects", async () => {
    const user = userEvent.setup();
    baseProps.onExport.mockRejectedValueOnce(new Error("export failed"));

    renderWithProviders(<EntityUsageExportModal {...baseProps} />);

    await user.click(screen.getByRole("button", { name: /Export CSV/i }));

    expect(baseProps.onExport).toHaveBeenCalled();
    expect(downloadBlob).not.toHaveBeenCalled();
    expect(baseProps.onClose).not.toHaveBeenCalled();
  });
});
