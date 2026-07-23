import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

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

beforeEach(() => {
  mockUseCredentials.mockReset();
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

  it("shows exactly the logging destinations the backend returned, without any client-side scope filtering", () => {
    // GET /credentials is already scoped server-side (proxy admin -> all; team/org
    // admin -> only in-scope destinations) by the same predicate the assignment gate
    // and the resolver use. The picker must therefore render every logging-typed
    // destination in the response verbatim; re-filtering here by role/scope on the
    // client would risk disagreeing with the authoritative backend in either
    // direction. This response mixes access shapes to prove none are dropped locally.
    mockUseCredentials.mockReturnValue({
      data: {
        credentials: [
          { credential_name: "team-scoped", credential_info: { credential_type: "logging", access: { teams: ["t"] } } },
          { credential_name: "org-scoped", credential_info: { credential_type: "logging", access: { orgs: ["o"] } } },
          { credential_name: "everyone", credential_info: { credential_type: "logging", access: { global: true } } },
          { credential_name: "always-on", credential_info: { credential_type: "logging", auto_enable: true } },
          { credential_name: "provider", credential_info: { custom_llm_provider: "openai" } },
        ],
      },
    });

    render(<LoggingExportersSelect value={[]} onChange={() => {}} />);

    const options = screen.getAllByTestId("option").map((el) => el.textContent);
    // every logging destination the backend returned, and only those (provider dropped)
    expect(options).toEqual(["team-scoped", "org-scoped", "everyone", "always-on"]);
  });
});
