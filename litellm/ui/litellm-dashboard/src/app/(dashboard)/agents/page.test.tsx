import { screen } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "../../../../tests/test-utils";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import Agents from "./page";

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: vi.fn() }));
vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({ useTeams: () => ({ data: [] }) }));
vi.mock("./_components/AgentsPanel", () => ({ default: () => <div>Registered agents panel</div> }));
vi.mock("@/components/agents/external/ExternalAgentsPanel", () => ({
  default: () => <div>External agents panel</div>,
}));

beforeEach(() => testQueryClient.clear());

it.each(["Admin", "proxy_admin"])("permits proxy administrator %s to open external agents", (userRole) => {
  vi.mocked(useAuthorized).mockReturnValue({ accessToken: "test-token", userRole } as ReturnType<typeof useAuthorized>);
  renderWithProviders(<Agents />, { searchParams: { agent_section: "external" } });
  expect(screen.getByText("External agents panel")).toBeInTheDocument();
});

it.each(["org_admin", "proxy_admin_viewer", "internal_user"])(
  "does not expose external agents to %s even with a deep link",
  (userRole) => {
    vi.mocked(useAuthorized).mockReturnValue({ accessToken: "test-token", userRole } as ReturnType<
      typeof useAuthorized
    >);
    renderWithProviders(<Agents />, { searchParams: { agent_section: "external" } });
    expect(screen.queryByText("External agents panel")).not.toBeInTheDocument();
    expect(screen.getByText("Registered agents panel")).toBeInTheDocument();
  },
);
