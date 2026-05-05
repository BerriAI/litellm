import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ConfigInfoMessage } from "./ConfigInfoMessage";

describe("ConfigInfoMessage", () => {
  it("should render the info message when show is true", () => {
    render(<ConfigInfoMessage show={true} />);
    expect(screen.getByText("Request/Response Data Not Available")).toBeInTheDocument();
  });

  it("should render nothing when show is false", () => {
    const { container } = render(<ConfigInfoMessage show={false} />);
    expect(container.innerHTML).toBe("");
  });

  it("should display the YAML config snippet", () => {
    render(<ConfigInfoMessage show={true} />);
    expect(screen.getByText(/store_prompts_in_spend_logs: true/)).toBeInTheDocument();
  });

  it("should reference Admin Settings \u2192 Logging Settings", () => {
    render(<ConfigInfoMessage show={true} />);
    expect(screen.getByText(/Admin Settings → Logging Settings/)).toBeInTheDocument();
  });
});
