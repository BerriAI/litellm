import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import LoggingExportersSelect from "./LoggingExportersSelect";

const mockUseCredentials = vi.fn();
const mockUseAuthorized = vi.fn();
const mockUseTeams = vi.fn();
const mockUseOrganizations = vi.fn();

vi.mock("@/app/(dashboard)/hooks/credentials/useCredentials", () => ({
  useCredentials: () => mockUseCredentials(),
}));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));
vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useTeams: () => mockUseTeams(),
}));
vi.mock("@/app/(dashboard)/hooks/organizations/useOrganizations", () => ({
  useOrganizations: () => mockUseOrganizations(),
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
  // Default: a proxy admin (formatted role "Admin"), no team/org membership needed.
  mockUseAuthorized.mockReturnValue({ userRole: "Admin" });
  mockUseTeams.mockReturnValue({ data: [] });
  mockUseOrganizations.mockReturnValue({ data: [] });
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

  it("scopes a non-admin caller to destinations granted to their team/org (plus global/auto_enable)", () => {
    // An internal_user who is a member of team-a. They must see only what they could
    // actually assign: the team-a destination, the global one, and the auto_enable
    // default -- never the team-b destination or the foreign-org one. This mirrors the
    // backend assignment gate; the backend stays the authoritative check.
    mockUseAuthorized.mockReturnValue({ userRole: "Internal User" });
    mockUseTeams.mockReturnValue({ data: [{ team_id: "team-a" }] });
    mockUseOrganizations.mockReturnValue({ data: [{ organization_id: "org-a" }] });
    mockUseCredentials.mockReturnValue({
      data: {
        credentials: [
          {
            credential_name: "mine-team",
            credential_info: { credential_type: "logging", access: { teams: ["team-a"] } },
          },
          {
            credential_name: "foreign-team",
            credential_info: { credential_type: "logging", access: { teams: ["team-b"] } },
          },
          { credential_name: "mine-org", credential_info: { credential_type: "logging", access: { orgs: ["org-a"] } } },
          {
            credential_name: "foreign-org",
            credential_info: { credential_type: "logging", access: { orgs: ["org-z"] } },
          },
          { credential_name: "everyone", credential_info: { credential_type: "logging", access: { global: true } } },
          { credential_name: "always-on", credential_info: { credential_type: "logging", auto_enable: true } },
        ],
      },
    });

    render(<LoggingExportersSelect value={[]} onChange={() => {}} />);

    const options = screen.getAllByTestId("option").map((el) => el.textContent);
    expect(options).toEqual(["mine-team", "mine-org", "everyone", "always-on"]);
  });

  it("shows every logging destination to a proxy admin regardless of access scope", () => {
    mockUseAuthorized.mockReturnValue({ userRole: "Admin" });
    mockUseTeams.mockReturnValue({ data: [] });
    mockUseCredentials.mockReturnValue({
      data: {
        credentials: [
          {
            credential_name: "team-b-only",
            credential_info: { credential_type: "logging", access: { teams: ["team-b"] } },
          },
          { credential_name: "no-access", credential_info: { credential_type: "logging" } },
        ],
      },
    });

    render(<LoggingExportersSelect value={[]} onChange={() => {}} />);

    const options = screen.getAllByTestId("option").map((el) => el.textContent);
    expect(options).toEqual(["team-b-only", "no-access"]);
  });
});
