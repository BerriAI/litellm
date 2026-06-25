import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import LoggingExportersSelect from "./LoggingExportersSelect";

const mockUseCredentials = vi.fn();

vi.mock("@/app/(dashboard)/hooks/credentials/useCredentials", () => ({
  useCredentials: () => mockUseCredentials(),
}));

vi.mock("antd", async () => {
  const React = await import("react");
  function Select(props: any) {
    const { value, onChange, options, notFoundContent } = props;
    return React.createElement(
      "div",
      { "data-testid": "logging-exporters-select" },
      React.createElement(
        "ul",
        null,
        (options ?? []).map((opt: any) =>
          React.createElement("li", { key: opt.value, "data-testid": "option" }, opt.label),
        ),
      ),
      options && options.length === 0 ? React.createElement("div", { "data-testid": "empty" }, notFoundContent) : null,
      React.createElement(
        "button",
        { "data-testid": "pick-first", onClick: () => onChange?.(options?.[0] ? [options[0].value] : []) },
        "pick first",
      ),
      React.createElement("div", { "data-testid": "value" }, JSON.stringify(value ?? [])),
    );
  }
  return { Select };
});

describe("LoggingExportersSelect", () => {
  it("only surfaces credentials whose credential_type is 'logging'", () => {
    mockUseCredentials.mockReturnValue({
      data: {
        credentials: [
          {
            credential_name: "poc-langfuse",
            credential_info: { credential_type: "logging", host: "https://cloud.langfuse.com" },
          },
          {
            credential_name: "poc-arize",
            credential_info: { credential_type: "logging" },
          },
          {
            credential_name: "openai-prod",
            credential_info: { custom_llm_provider: "openai" },
          },
        ],
      },
    });

    render(<LoggingExportersSelect value={[]} onChange={() => {}} />);

    const options = screen.getAllByTestId("option").map((el) => el.textContent);
    expect(options).toEqual(["poc-langfuse (https://cloud.langfuse.com)", "poc-arize"]);
  });

  it("renders empty-state copy when no logging destinations exist", () => {
    mockUseCredentials.mockReturnValue({
      data: {
        credentials: [
          {
            credential_name: "openai-prod",
            credential_info: { custom_llm_provider: "openai" },
          },
        ],
      },
    });

    render(<LoggingExportersSelect value={[]} onChange={() => {}} />);

    expect(screen.queryAllByTestId("option")).toHaveLength(0);
    expect(screen.getByTestId("empty").textContent).toMatch(/proxy admin/i);
  });

  it("works for a team-admin caller: backend already filters to logging-typed; component renders whatever it gets", () => {
    // Simulates what /credentials returns to a team-admin via the backend
    // route widening: provider credentials are dropped server-side; only
    // logging-typed entries reach the component.
    mockUseCredentials.mockReturnValue({
      data: {
        credentials: [{ credential_name: "poc-langfuse", credential_info: { credential_type: "logging" } }],
      },
    });

    render(<LoggingExportersSelect value={["poc-langfuse"]} onChange={() => {}} />);

    expect(screen.getByTestId("option").textContent).toBe("poc-langfuse");
    expect(screen.getByTestId("value").textContent).toBe(JSON.stringify(["poc-langfuse"]));
  });
});
