import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import EntityUsageExportModal from "./EntityUsageExportModal";

const exportRow = (overrides: Record<string, unknown>) => ({
  date: "2025-10-02",
  entity_id: "team-a",
  entity_alias: "Team A",
  api_key: "hashed-key",
  key_alias: null,
  user_id: null,
  user_email: null,
  model: "gpt-4o",
  spend: 0,
  flat_cost: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  api_requests: 0,
  successful_requests: 0,
  failed_requests: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
  ...overrides,
});

const jsonBlob = (rows: readonly unknown[]) => {
  const text = JSON.stringify(rows);
  return Object.assign(new Blob([text], { type: "application/json" }), { text: async () => text });
};

const reportDocument = (): Document => {
  const frame = screen.getByTitle("Usage report");
  if (!(frame instanceof HTMLIFrameElement) || !frame.contentDocument) throw new Error("Usage report frame missing");
  return frame.contentDocument;
};

describe("EntityUsageExportModal PDF export", () => {
  afterEach(() => {
    screen.queryByTitle("Usage report")?.remove();
  });

  it("builds a printable report from the JSON export of the chosen type and filters", async () => {
    const user = userEvent.setup();
    const gptDay = {
      model: "gpt-4o",
      spend: 1.5,
      prompt_tokens: 100,
      completion_tokens: 50,
      api_requests: 3,
      successful_requests: 2,
      failed_requests: 1,
    };
    const claudeDay = {
      date: "2025-10-03",
      model: "claude-sonnet",
      spend: 2.25,
      prompt_tokens: 200,
      completion_tokens: 25,
      api_requests: 4,
      successful_requests: 4,
    };
    const onExport = vi.fn().mockResolvedValue(jsonBlob([exportRow(gptDay), exportRow(claudeDay)]));
    const onClose = vi.fn();

    const dateRange = { from: new Date(2025, 9, 1), to: new Date(2025, 9, 14) };

    renderWithProviders(
      <EntityUsageExportModal
        isOpen
        onClose={onClose}
        entityType="team"
        onExport={onExport}
        dateRange={dateRange}
        selectedFilters={["team-a"]}
        canIncludeMetadata
      />,
    );

    await user.click(screen.getByRole("radio", { name: /Day-by-day by team and model/i }));
    await user.click(screen.getByRole("checkbox", { name: /Include key and user metadata/i }));
    await user.click(screen.getByRole("combobox"));
    await user.click(await screen.findByRole("option", { name: /PDF/ }));
    expect(screen.queryByRole("checkbox", { name: /Include key and user metadata/i })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Export PDF" }));

    expect(onExport).toHaveBeenCalledWith("daily_with_models", "json", false);
    await waitFor(() => expect(onClose).toHaveBeenCalled());
    const report = reportDocument();
    expect(report.title).toBe("team_usage_daily_with_models_2025-10-01_2025-10-14");
    const page = within(report.body);
    const summary = page.getByTestId("usage-report-summary");
    expect(summary).toHaveTextContent("Total spend$3.75");
    expect(summary).toHaveTextContent("Requests7");
    expect(summary).toHaveTextContent("Total tokens375");
    expect(page.getByText("team-a")).toBeInTheDocument();
    const rows = page
      .getAllByRole("row")
      .slice(1)
      .map((row) =>
        within(row)
          .getAllByRole("cell")
          .map((cell) => cell.textContent),
      );
    expect(rows).toEqual([
      ["2025-10-02", "Team A", "gpt-4o", "$1.50", "3", "2", "1", "100", "50"],
      ["2025-10-03", "Team A", "claude-sonnet", "$2.25", "4", "4", "0", "200", "25"],
    ]);
  });
});
