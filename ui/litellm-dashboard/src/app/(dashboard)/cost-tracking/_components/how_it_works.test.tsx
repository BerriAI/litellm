import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../../../tests/test-utils";
import HowItWorks from "./how_it_works";

vi.mock("@/components/CodeBlock", () => ({
  default: ({ code }: { code: string }) => <pre data-testid="code-block">{code}</pre>,
}));

describe("HowItWorks", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render", () => {
    renderWithProviders(<HowItWorks />);
    expect(screen.getByText("Cost Calculation")).toBeInTheDocument();
  });

  it("should display the cost calculation formula", () => {
    renderWithProviders(<HowItWorks />);
    expect(screen.getByText(/final_cost = base_cost/i)).toBeInTheDocument();
  });

  it("should display the valid range information", () => {
    renderWithProviders(<HowItWorks />);
    expect(screen.getByText(/0% and 100%/i)).toBeInTheDocument();
  });

  it("should render the code block with a curl example", () => {
    renderWithProviders(<HowItWorks />);
    expect(screen.getByTestId("code-block")).toBeInTheDocument();
    expect(screen.getByTestId("code-block")).toHaveTextContent(/curl/);
  });

  it("should show the response header names for discount verification", () => {
    renderWithProviders(<HowItWorks />);
    expect(screen.getByText("x-litellm-response-cost")).toBeInTheDocument();
    expect(screen.getByText("x-litellm-response-cost-original")).toBeInTheDocument();
    expect(screen.getByText("x-litellm-response-cost-discount-amount")).toBeInTheDocument();
  });

  it("should not show calculated results initially when no input is provided", () => {
    renderWithProviders(<HowItWorks />);
    expect(screen.queryByText("Calculated Results")).not.toBeInTheDocument();
  });

  it("should not show calculated results when only response cost is entered", async () => {
    const user = userEvent.setup();
    renderWithProviders(<HowItWorks />);

    const responseCostInput = screen.getByPlaceholderText("0.0171938125");
    fireEvent.change(responseCostInput, { target: { value: "0.01" } });

    expect(screen.queryByText("Calculated Results")).not.toBeInTheDocument();
  });

  it("should not show calculated results when only discount amount is entered", async () => {
    const user = userEvent.setup();
    renderWithProviders(<HowItWorks />);

    const discountAmountInput = screen.getByPlaceholderText("0.0009049375");
    fireEvent.change(discountAmountInput, { target: { value: "0.001" } });

    expect(screen.queryByText("Calculated Results")).not.toBeInTheDocument();
  });

  it("should show calculated results when both fields are filled", async () => {
    const user = userEvent.setup();
    renderWithProviders(<HowItWorks />);

    const responseCostInput = screen.getByPlaceholderText("0.0171938125");
    const discountAmountInput = screen.getByPlaceholderText("0.0009049375");

    fireEvent.change(responseCostInput, { target: { value: "0.0171938125" } });
    fireEvent.change(discountAmountInput, { target: { value: "0.0009049375" } });

    expect(await screen.findByText("Calculated Results")).toBeInTheDocument();
  });

  it("should show original cost, final cost, and discount amount in results", async () => {
    const user = userEvent.setup();
    renderWithProviders(<HowItWorks />);

    fireEvent.change(screen.getByPlaceholderText("0.0171938125"), { target: { value: "0.0171938125" } });
    fireEvent.change(screen.getByPlaceholderText("0.0009049375"), { target: { value: "0.0009049375" } });

    expect(await screen.findByText("Original Cost:")).toBeInTheDocument();
    expect(screen.getByText("Final Cost:")).toBeInTheDocument();
    expect(screen.getByText("Discount Amount:")).toBeInTheDocument();
    expect(screen.getByText("Discount Applied:")).toBeInTheDocument();
  });
});
