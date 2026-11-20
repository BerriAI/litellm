import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { chooseSelectOption, renderWithProviders, testQueryClient } from "../../../tests/test-utils";
import { switchToWorkerUrl } from "@/components/networking";
import { registerAuthTokenGetter } from "@/lib/http/runtime";
import { GatewayWorkspaceSwitcher } from "./GatewayWorkspaceSwitcher";

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "test-session", userId: "test-user", userRole: "Admin", premiumUser: true }),
}));

const KEY_HASH = "a".repeat(64);
const organization = { organization_id: "org-a", organization_alias: "Engineering organization" };
const team = { team_id: "team-a", team_alias: "Platform team", organization_id: organization.organization_id };
const project = { project_id: "project-a", project_alias: "Coding project", team_id: team.team_id };
const key = {
  token: KEY_HASH,
  key_alias: "Coding key",
  api_key: "private-field-fixture",
  organization_id: organization.organization_id,
  team_id: team.team_id,
  project_id: project.project_id,
};
const keyListResponse = {
  keys: [key, { token: "raw-key-material-fixture", key_alias: "Unsupported credential" }],
  current_page: 1,
  total_pages: 1,
  total_count: 2,
};
const network = vi.fn<typeof fetch>();

beforeEach(() => {
  testQueryClient.clear();
  network.mockReset();
  switchToWorkerUrl("https://gateway.test");
  registerAuthTokenGetter(() => "test-session");
  vi.stubGlobal("fetch", network);
  network.mockImplementation(async (input) => {
    const url = new URL(input instanceof Request ? input.url : String(input));
    switch (url.pathname) {
      case "/organization/list":
        return Response.json([organization]);
      case "/organization/info":
        return Response.json(organization);
      case "/v2/team/list":
        return Response.json({ teams: [team], total_pages: 1 });
      case "/team/info":
        return Response.json({ team_info: team });
      case "/project/list":
        return Response.json([project]);
      case "/project/info":
        return Response.json(project);
      case "/key/list":
        return Response.json(keyListResponse);
      case "/key/info":
        return Response.json({ info: key });
      default:
        throw new Error(`Unexpected workspace request ${url.pathname}`);
    }
  });
});

afterEach(() => {
  switchToWorkerUrl(null);
  vi.unstubAllGlobals();
});

const lastParams = (onUrlUpdate: ReturnType<typeof vi.fn>) =>
  new URLSearchParams(String(onUrlUpdate.mock.lastCall?.[0].queryString ?? ""));

it("selects a gateway key and fills its organization, team, and project without exposing credential fields", async () => {
  const user = userEvent.setup();
  const onUrlUpdate = vi.fn();
  renderWithProviders(<GatewayWorkspaceSwitcher />, { searchParams: "?tab=traces", onUrlUpdate });
  await user.click(screen.getByRole("combobox", { name: "Key workspace" }));
  const option = await screen.findByRole("option", { name: key.key_alias });
  expect(screen.queryByRole("option", { name: "Unsupported credential" })).not.toBeInTheDocument();
  expect(screen.queryByText(key.api_key)).not.toBeInTheDocument();
  expect(screen.queryByText("raw-key-material-fixture")).not.toBeInTheDocument();
  await user.click(option);
  await waitFor(() => expect(lastParams(onUrlUpdate).get("api_key_hash")).toBe(KEY_HASH));
  expect(lastParams(onUrlUpdate).get("organization_id")).toBe(organization.organization_id);
  expect(lastParams(onUrlUpdate).get("team_id")).toBe(team.team_id);
  expect(lastParams(onUrlUpdate).get("project_id")).toBe(project.project_id);
  await waitFor(() =>
    expect(screen.getByRole("combobox", { name: "Organization workspace" })).toHaveTextContent(
      organization.organization_alias,
    ),
  );
  expect(screen.getByRole("combobox", { name: "Team workspace" })).toHaveTextContent(team.team_alias);
  expect(screen.getByRole("combobox", { name: "Project workspace" })).toHaveTextContent(project.project_alias);
  expect(screen.getByRole("combobox", { name: "Key workspace" })).toHaveTextContent(key.key_alias);
});

it("keeps the resolved parents when showing all keys from a key-only deep link", async () => {
  const user = userEvent.setup();
  const onUrlUpdate = vi.fn();
  renderWithProviders(<GatewayWorkspaceSwitcher />, {
    searchParams: `?tab=traces&api_key_hash=${KEY_HASH}`,
    onUrlUpdate,
  });
  await waitFor(() =>
    expect(screen.getByRole("combobox", { name: "Organization workspace" })).toHaveTextContent(
      organization.organization_alias,
    ),
  );
  expect(screen.getByRole("combobox", { name: "Team workspace" })).toHaveTextContent(team.team_alias);
  expect(screen.getByRole("combobox", { name: "Project workspace" })).toHaveTextContent(project.project_alias);
  await chooseSelectOption(user, screen.getByRole("combobox", { name: "Key workspace" }), "All keys");
  await waitFor(() => expect(onUrlUpdate).toHaveBeenCalledOnce());
  const expectedScope = {
    tab: "traces",
    organization_id: organization.organization_id,
    team_id: team.team_id,
    project_id: project.project_id,
  };
  expect(Object.fromEntries(lastParams(onUrlUpdate))).toEqual(expectedScope);
  expect(screen.getByRole("combobox", { name: "Key workspace" })).toHaveTextContent("All keys");
});

it("resolves the team and organization when landing on a project-only deep link", async () => {
  renderWithProviders(<GatewayWorkspaceSwitcher />, { searchParams: `?tab=traces&project_id=${project.project_id}` });
  await waitFor(() =>
    expect(screen.getByRole("combobox", { name: "Organization workspace" })).toHaveTextContent(
      organization.organization_alias,
    ),
  );
  expect(screen.getByRole("combobox", { name: "Team workspace" })).toHaveTextContent(team.team_alias);
  expect(screen.getByRole("combobox", { name: "Project workspace" })).toHaveTextContent(project.project_alias);
  expect(screen.getByRole("combobox", { name: "Key workspace" })).toHaveTextContent("All keys");
});
