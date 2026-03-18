import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import ExportFormatSelector from "./ExportFormatSelector";

describe("ExportFormatSelector", () => {
  it("should render", () => {
    render(<ExportFormatSelector value="csv" onChange={vi.fn()} />);
    expect(screen.getByText("Format")).toBeInTheDocument();
  });

  it("should display the current value as csv", () => {
    render(<ExportFormatSelector value="csv" onChange={vi.fn()} />);
    expect(screen.getByText("CSV (Excel, Google Sheets)")).toBeInTheDocument();
  });

  it("should display the current value as json", () => {
    render(<ExportFormatSelector value="json" onChange={vi.fn()} />);
    expect(screen.getByText("JSON (includes metadata)")).toBeInTheDocument();
  });

  it("should call onChange when a different format is selected", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<ExportFormatSelector value="csv" onChange={onChange} />);

    // Open the Ant Design Select dropdown
    await user.click(screen.getByText("CSV (Excel, Google Sheets)"));
    // Select JSON option from the dropdown
    const jsonOption = await screen.findByText("JSON (includes metadata)");
    await user.click(jsonOption);
    expect(onChange).toHaveBeenCalledWith("json", expect.anything());
  });
});
