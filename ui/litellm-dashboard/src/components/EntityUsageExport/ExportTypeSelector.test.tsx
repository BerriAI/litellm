import { renderWithProviders, screen } from "../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import ExportTypeSelector from "./ExportTypeSelector";

describe("ExportTypeSelector", () => {
  it("should render", () => {
    renderWithProviders(<ExportTypeSelector value="daily" onChange={vi.fn()} entityType="team" />);
    expect(screen.getByText("Export type")).toBeInTheDocument();
  });

  it("should display entity type in radio labels", () => {
    renderWithProviders(<ExportTypeSelector value="daily" onChange={vi.fn()} entityType="team" />);
    expect(screen.getByText(/Day-by-day breakdown by team$/)).toBeInTheDocument();
    expect(screen.getByText(/Day-by-day breakdown by team and key/)).toBeInTheDocument();
    expect(screen.getByText(/Day-by-day by team and model/)).toBeInTheDocument();
  });

  it("should display the correct entity type for different entities", () => {
    renderWithProviders(<ExportTypeSelector value="daily" onChange={vi.fn()} entityType="organization" />);
    expect(screen.getByText(/Day-by-day breakdown by organization$/)).toBeInTheDocument();
  });

  it("should call onChange when a radio option is selected", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderWithProviders(<ExportTypeSelector value="daily" onChange={onChange} entityType="team" />);
    await user.click(screen.getByRole("radio", { name: /Day-by-day breakdown by team and key/i }));
    expect(onChange).toHaveBeenCalledWith("daily_with_keys");
  });

  it("should have the correct radio checked", () => {
    renderWithProviders(<ExportTypeSelector value="daily_with_models" onChange={vi.fn()} entityType="team" />);
    expect(screen.getByRole("radio", { name: /Day-by-day by team and model/i })).toBeChecked();
  });

  it("should offer the per-user scope for teams and call onChange with daily_with_users", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderWithProviders(<ExportTypeSelector value="daily" onChange={onChange} entityType="team" />);

    const option = screen.getByRole("radio", { name: /Day-by-day breakdown by team and user/i });
    await user.click(option);

    expect(onChange).toHaveBeenCalledWith("daily_with_users");
    expect(screen.getByText("Daily metrics for each team, split by key owner")).toBeInTheDocument();
  });

  it("should hide the per-user scope for user exports while keeping the other scopes", () => {
    renderWithProviders(<ExportTypeSelector value="daily" onChange={vi.fn()} entityType="user" />);

    expect(screen.queryByRole("radio", { name: /and user/i })).not.toBeInTheDocument();
    expect(
      screen.getByRole("radio", { name: /Day-by-day breakdown by user Daily metrics for each user$/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /Day-by-day breakdown by user and key/i })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /Day-by-day by user and model/i })).toBeInTheDocument();
  });

  it("should offer the per-user scope for tags", () => {
    renderWithProviders(<ExportTypeSelector value="daily" onChange={vi.fn()} entityType="tag" />);

    expect(screen.getByRole("radio", { name: /Day-by-day breakdown by tag and user/i })).toBeInTheDocument();
  });
});
