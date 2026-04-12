import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
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

  it("should render the settings button when onOpenSettings is provided", () => {
    render(<ConfigInfoMessage show={true} onOpenSettings={() => {}} />);
    expect(screen.getByText("open the settings")).toBeInTheDocument();
  });

  it("should not render the settings button when onOpenSettings is omitted", () => {
    render(<ConfigInfoMessage show={true} />);
    expect(screen.queryByText("open the settings")).not.toBeInTheDocument();
  });

  it("should call onOpenSettings when the settings button is clicked", async () => {
    const user = userEvent.setup();
    const onOpenSettings = vi.fn();

    render(<ConfigInfoMessage show={true} onOpenSettings={onOpenSettings} />);
    await user.click(screen.getByText("open the settings"));

    expect(onOpenSettings).toHaveBeenCalledOnce();
  });
});
