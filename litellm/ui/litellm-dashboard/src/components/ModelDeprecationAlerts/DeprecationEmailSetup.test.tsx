import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import DeprecationEmailSetup from "./DeprecationEmailSetup";

describe("deprecation email setup", () => {
  it("should render an ambient setup action", () => {
    render(<DeprecationEmailSetup />);
    expect(screen.getByRole("button", { name: "Set up email reminders" })).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
  it("should explain the three setup steps and link to Email Settings", async () => {
    const user = userEvent.setup();
    render(<DeprecationEmailSetup />);
    await user.click(screen.getByRole("button", { name: "Set up email reminders" }));
    expect(await screen.findByRole("dialog", { name: "Set up alert emails" })).toBeInTheDocument();
    expect(screen.getByText("Connect email delivery")).toBeInTheDocument();
    expect(screen.getByText("Choose your email alerts")).toBeInTheDocument();
    expect(screen.getByText("Know what gets sent")).toBeInTheDocument();
    expect(screen.getByText(/Both are off by default/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open Email Settings" })).toHaveAttribute(
      "href",
      "/ui/logging-and-alerts?tab=email-alerts",
    );
  });
});
