import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "@/../tests/test-utils";
import type { Plugin } from "@/components/claude_code_plugins/types";
import SkillHubDashboard from "./SkillHubDashboard";

const skills: Plugin[] = [
  { id: "alpha", name: "alpha", source: {}, category: "Writing", domain: "Support", enabled: false, download_count: 1 },
  {
    id: "beta",
    name: "beta",
    source: {},
    category: "Analysis",
    domain: "Engineering",
    enabled: true,
    download_count: 7,
  },
];

beforeEach(() => testQueryClient.clear());
afterEach(() => vi.unstubAllGlobals());

it("shows creator email in the authenticated hub and keeps directory data off the public view", async () => {
  const creator = { user_id: "creator-uuid", user_email: "creator@example.test", user_alias: "Creator Name" };
  const fetchMock = vi.fn<typeof fetch>(async (input) => {
    const url = new URL(input instanceof Request ? input.url : String(input), "http://localhost");
    if (url.pathname === "/management/v1/users/me")
      return Response.json({ user_id: "admin", user_email: null, user_role: "proxy_admin" });
    if (url.pathname === "/v2/user/info") return Response.json(creator);
    return Response.json({ detail: "Unexpected request" }, { status: 404 });
  });
  vi.stubGlobal("fetch", fetchMock);
  const catalog = [{ ...skills[0], created_by: creator.user_id }];
  const { rerender } = renderWithProviders(
    <SkillHubDashboard skills={catalog} isLoading={false} accessToken="admin-token" />,
  );
  expect(await screen.findByRole("cell", { name: creator.user_email })).toBeInTheDocument();
  fetchMock.mockClear();
  rerender(<SkillHubDashboard skills={catalog} isLoading={false} accessToken="admin-token" publicPage />);
  expect(screen.getByRole("cell", { name: creator.user_id })).toBeInTheDocument();
  expect(screen.queryByText(creator.user_email)).not.toBeInTheDocument();
  expect(fetchMock).not.toHaveBeenCalled();
});

it("links a pinned GitLab source and renders SSH sources as text", () => {
  const sha = "a".repeat(40);
  const catalog: Plugin[] = [
    {
      ...skills[0],
      source: { source: "git-subdir", url: "https://gitlab.com/team/nested/repo.git", sha, path: "plugins/example" },
    },
    { ...skills[1], source: { source: "url", url: "git@git.example.test:team/repo.git" } },
  ];
  renderWithProviders(<SkillHubDashboard skills={catalog} isLoading={false} />);
  expect(within(screen.getByRole("row", { name: /alpha/ })).getByRole("link")).toHaveAttribute(
    "href",
    `https://gitlab.com/team/nested/repo/-/tree/${sha}/plugins/example`,
  );
  const sshRow = screen.getByRole("row", { name: /beta/ });
  expect(sshRow).toHaveTextContent("git@git.example.test:team/repo.git");
  expect(within(sshRow).queryByRole("link")).not.toBeInTheDocument();
});

describe("Skills Hub sorting", () => {
  it.each(["Category", "Domain", "Status", "Downloads"])("sorts by %s through the table header", async (header) => {
    const user = userEvent.setup();
    renderWithProviders(<SkillHubDashboard skills={skills} isLoading={false} />);
    expect(screen.getAllByRole("row")[1]).toHaveTextContent("alpha");
    await user.click(screen.getByRole("button", { name: header, exact: true }));
    await waitFor(() => expect(screen.getAllByRole("row")[1]).toHaveTextContent("beta"));
    await user.click(screen.getByRole("button", { name: header, exact: true }));
    await waitFor(() => expect(screen.getAllByRole("row")[1]).toHaveTextContent("alpha"));
  });
});
