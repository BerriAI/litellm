import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { BuilderInsightsDemoBanner } from "./BuilderInsightsDemoBanner";

const builders = [{ email: "ava@example.com" }, { email: "ben@example.com" }] as const;

describe("BuilderInsightsDemoBanner", () => {
  it("guides setup through repository and user mapping, then saves without exposing the token", () => {
    render(<BuilderInsightsDemoBanner builders={builders} />);

    fireEvent.click(screen.getByRole("button", { name: "How to connect" }));
    fireEvent.click(screen.getByRole("button", { name: /Personal access token/ }));

    const organization = screen.getByRole("textbox", { name: "GitHub organization" });
    const token = screen.getByLabelText("Personal access token");
    const continueButton = screen.getByRole("button", { name: "Continue" });

    expect(continueButton).toBeDisabled();
    fireEvent.change(organization, { target: { value: "acme-eng" } });
    expect(continueButton).toBeDisabled();
    fireEvent.change(token, { target: { value: "ghp_demo" } });
    expect(continueButton).toBeEnabled();

    fireEvent.click(continueButton);
    expect(screen.getByText("acme-eng/api")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("checkbox", { name: "Select all" }));
    expect(screen.getByRole("checkbox", { name: "Select all" })).toBeChecked();
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    const avaLogin = screen.getByRole("textbox", { name: "GitHub login for ava@example.com" });
    expect(avaLogin).toHaveValue("ava");
    fireEvent.change(avaLogin, { target: { value: "ava-gh" } });
    expect(screen.getByText("2 of 2 matched")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    expect(screen.getByText("Personal access token")).toBeInTheDocument();
    expect(screen.getByText("acme-eng")).toBeInTheDocument();
    expect(screen.getByText("4 repos")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Finish" }));

    expect(screen.getByRole("status")).toHaveTextContent(
      "GitHub setup saved for acme-eng. Sample data shows until sync ships.",
    );
    expect(screen.getByRole("status")).not.toHaveTextContent("ghp_demo");
    expect(screen.queryByLabelText("Personal access token")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "How to connect" }));
    fireEvent.click(screen.getByRole("button", { name: /Personal access token/ }));
    expect(screen.getByLabelText("Personal access token")).toHaveValue("");
  });

  it("requires an organization before continuing with the GitHub App", () => {
    render(<BuilderInsightsDemoBanner builders={builders} />);

    fireEvent.click(screen.getByRole("button", { name: "How to connect" }));
    const continueButton = screen.getByRole("button", { name: "Continue" });

    expect(continueButton).toBeDisabled();
    fireEvent.change(screen.getByRole("textbox", { name: "GitHub organization" }), {
      target: { value: "acme-eng" },
    });
    expect(continueButton).toBeEnabled();
  });
});
