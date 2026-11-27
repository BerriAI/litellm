import userEvent from "@testing-library/user-event";
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

  it("offers CSV, JSON and PDF and reports the picked format", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderWithProviders(<ExportFormatSelector value="csv" onChange={onChange} />);

    await user.click(screen.getByRole("combobox"));
    const options = await screen.findAllByRole("option");
    expect(options.map((option) => option.textContent)).toEqual([
      "CSV (Excel, Google Sheets)",
      "JSON (includes metadata)",
      "PDF (printable report)",
    ]);
    await user.click(options[2]);
    expect(onChange).toHaveBeenCalledWith("pdf");
  });
});
