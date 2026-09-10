import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Alert, AlertAction, AlertDescription, AlertTitle } from "./Alert";

describe("Alert", () => {
  it("exposes the alert role plus the slot and variant hooks", () => {
    render(
      <Alert variant="warning">
        <AlertTitle>License expiring</AlertTitle>
        <AlertDescription>Renew before it lapses.</AlertDescription>
      </Alert>,
    );

    const alert = screen.getByRole("alert");
    expect(alert).toHaveAttribute("data-slot", "alert");
    expect(alert).toHaveAttribute("data-variant", "warning");
    expect(alert).toHaveTextContent("License expiring");
    expect(alert).toHaveTextContent("Renew before it lapses.");
  });

  it("reports the default variant when none is given", () => {
    render(<Alert>Heads up</Alert>);
    expect(screen.getByRole("alert")).toHaveAttribute("data-variant", "default");
  });

  it.each([
    ["info", "text-info"],
    ["success", "text-success"],
    ["warning", "text-warning"],
    ["error", "text-destructive"],
    ["destructive", "text-destructive"],
  ] as const)("paints the %s variant with its own token color", (variant, tokenClass) => {
    render(<Alert variant={variant}>message</Alert>);
    const alert = screen.getByRole("alert");
    expect(alert).toHaveAttribute("data-variant", variant);
    expect(alert).toHaveClass(tokenClass);
  });

  it("keeps success off the warning color so a save does not read as a problem", () => {
    render(<Alert variant="success">saved</Alert>);
    expect(screen.getByRole("alert")).not.toHaveClass("text-warning");
  });

  it("lets className win over the variant classes through twMerge", () => {
    render(
      <Alert variant="info" className="bg-card">
        message
      </Alert>,
    );
    const alert = screen.getByRole("alert");
    expect(alert).toHaveClass("bg-card");
    expect(alert).not.toHaveClass("bg-info/5");
  });

  it("renders the action slot so a dismiss control stays reachable", () => {
    render(
      <Alert variant="info">
        <AlertTitle>SSO enabled</AlertTitle>
        <AlertAction>
          <button type="button" aria-label="Close" />
        </AlertAction>
      </Alert>,
    );

    const action = screen.getByRole("button", { name: "Close" }).parentElement;
    expect(action).toHaveAttribute("data-slot", "alert-action");
  });

  it("tags the title and description parts for the layout selectors", () => {
    render(
      <Alert>
        <AlertTitle>Title</AlertTitle>
        <AlertDescription>Description</AlertDescription>
      </Alert>,
    );

    expect(screen.getByText("Title")).toHaveAttribute("data-slot", "alert-title");
    expect(screen.getByText("Description")).toHaveAttribute("data-slot", "alert-description");
  });
});
