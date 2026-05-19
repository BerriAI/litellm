import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import ReliabilityRetriesSection from "./ReliabilityRetriesSection";

const baseSettings = {
  num_retries: 3,
  timeout: 30,
  allowed_fails: 2,
  fallbacks: ["gpt-3.5"],
  context_window_fallbacks: [],
  routing_strategy_args: { ttl: 3600 },
  routing_strategy: "simple-shuffle",
  enable_tag_filtering: false,
};

describe("ReliabilityRetriesSection", () => {
  it("should render the section heading", () => {
    render(<ReliabilityRetriesSection routerSettings={{}} routerFieldsMetadata={{}} />);
    expect(screen.getByText("Reliability & Retries")).toBeInTheDocument();
  });

  it("should render input fields for non-excluded settings", () => {
    render(<ReliabilityRetriesSection routerSettings={baseSettings} routerFieldsMetadata={{}} />);
    expect(screen.getByDisplayValue("3")).toBeInTheDocument();   // num_retries
    expect(screen.getByDisplayValue("30")).toBeInTheDocument();  // timeout
    expect(screen.getByDisplayValue("2")).toBeInTheDocument();   // allowed_fails
  });

  it("should not render inputs for excluded keys", () => {
    render(<ReliabilityRetriesSection routerSettings={baseSettings} routerFieldsMetadata={{}} />);
    // Each excluded key must not produce a visible input value
    const inputs = screen.queryAllByRole("textbox");
    const inputNames = inputs.map((el) => el.getAttribute("name"));
    expect(inputNames).not.toContain("fallbacks");
    expect(inputNames).not.toContain("context_window_fallbacks");
    expect(inputNames).not.toContain("routing_strategy_args");
    expect(inputNames).not.toContain("routing_strategy");
    expect(inputNames).not.toContain("enable_tag_filtering");
  });

  it("should use ui_field_name from metadata as the label", () => {
    const metadata = {
      num_retries: { ui_field_name: "Number of Retries", field_description: "How many times to retry" },
    };
    render(
      <ReliabilityRetriesSection routerSettings={{ num_retries: 3 }} routerFieldsMetadata={metadata} />
    );
    expect(screen.getByText("Number of Retries")).toBeInTheDocument();
  });

  it("should fall back to the raw param name when no metadata label is available", () => {
    render(
      <ReliabilityRetriesSection routerSettings={{ num_retries: 3 }} routerFieldsMetadata={{}} />
    );
    expect(screen.getByText("num_retries")).toBeInTheDocument();
  });

  it("should render null values as an empty input", () => {
    render(
      <ReliabilityRetriesSection routerSettings={{ timeout: null }} routerFieldsMetadata={{}} />
    );
    const input = screen.getByRole("textbox", { name: /timeout/i }) as HTMLInputElement;
    expect(input.value).toBe("");
  });

  it("should render object values stringified into the input", () => {
    const settings = { retry_policy: { "rate-limited": 2 } };
    render(<ReliabilityRetriesSection routerSettings={settings} routerFieldsMetadata={{}} />);
    // HTML input type=text strips newlines, so check that the key/value appears
    const input = screen.getByRole("textbox", { name: /retry_policy/i }) as HTMLInputElement;
    expect(input.value).toContain('"rate-limited"');
    expect(input.value).toContain('2');
  });

  it("should render no inputs when routerSettings is empty", () => {
    render(<ReliabilityRetriesSection routerSettings={{}} routerFieldsMetadata={{}} />);
    expect(screen.queryAllByRole("textbox")).toHaveLength(0);
  });
});
