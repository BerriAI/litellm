import { renderWithProviders, screen } from "../../../tests/test-utils";
import { vi } from "vitest";
import ExportFormatSelector from "./ExportFormatSelector";

describe("ExportFormatSelector", () => {
  it("should render", () => {
    renderWithProviders(<ExportFormatSelector value="csv" onChange={vi.fn()} />);
    expect(screen.getByText("Format")).toBeInTheDocument();
  });

  it("should display the current value", () => {
    renderWithProviders(<ExportFormatSelector value="csv" onChange={vi.fn()} />);
    expect(screen.getByText("CSV (Excel, Google Sheets)")).toBeInTheDocument();
  });

  it("should display JSON label when json is selected", () => {
    renderWithProviders(<ExportFormatSelector value="json" onChange={vi.fn()} />);
    expect(screen.getByText("JSON (includes metadata)")).toBeInTheDocument();
  });
});
