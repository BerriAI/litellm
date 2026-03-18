import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import ExportTypeSelector from "./ExportTypeSelector";

describe("ExportTypeSelector", () => {
  it("should render", () => {
    render(<ExportTypeSelector value="daily" onChange={vi.fn()} entityType="team" />);
    expect(screen.getByText("Export type")).toBeInTheDocument();
  });

  it("should render all three radio options", () => {
    render(<ExportTypeSelector value="daily" onChange={vi.fn()} entityType="team" />);
    expect(screen.getAllByRole("radio")).toHaveLength(3);
  });

  it("should interpolate entity type in labels", () => {
    render(<ExportTypeSelector value="daily" onChange={vi.fn()} entityType="organization" />);
    expect(screen.getByText(/Day-by-day breakdown by organization$/)).toBeInTheDocument();
    expect(screen.getByText(/organization and key/)).toBeInTheDocument();
    expect(screen.getByText(/organization and model/)).toBeInTheDocument();
  });

  it("should call onChange when a different option is selected", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<ExportTypeSelector value="daily" onChange={onChange} entityType="team" />);
    await user.click(screen.getByText(/by team and key/));
    expect(onChange).toHaveBeenCalledWith("daily_with_keys");
  });

  it("should have the correct radio checked based on value prop", () => {
    render(<ExportTypeSelector value="daily_with_models" onChange={vi.fn()} entityType="team" />);
    const radios = screen.getAllByRole("radio");
    expect(radios[2]).toBeChecked();
  });
});
