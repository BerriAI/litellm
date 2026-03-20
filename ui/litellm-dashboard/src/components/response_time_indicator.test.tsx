import { render, screen } from "@testing-library/react";
import { ResponseTimeIndicator } from "./response_time_indicator";

describe("ResponseTimeIndicator", () => {
  it("should render", () => {
    render(<ResponseTimeIndicator responseTimeMs={150} />);
    expect(screen.getByText("150ms")).toBeInTheDocument();
  });

  it("should return null when responseTimeMs is null", () => {
    const { container } = render(<ResponseTimeIndicator responseTimeMs={null} />);
    expect(container.innerHTML).toBe("");
  });

  it("should round the displayed time to the nearest integer", () => {
    render(<ResponseTimeIndicator responseTimeMs={123.456} />);
    expect(screen.getByText("123ms")).toBeInTheDocument();
  });

  it("should display 0ms for a zero response time", () => {
    render(<ResponseTimeIndicator responseTimeMs={0} />);
    expect(screen.getByText("0ms")).toBeInTheDocument();
  });
});
