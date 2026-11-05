import { fireEvent, screen, waitFor } from "@testing-library/react";
import { renderWithProviders, testQueryClient } from "@/../tests/test-utils";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { getProxyBaseUrl, setGlobalLitellmHeaderName } from "@/components/networking";
import SkillDetail, { skillHubDetail } from "./skill_detail";
import type { Plugin, PluginDownloadEvent, PluginDownloadsResponse } from "./types";

const digest = "a".repeat(64);
const archivePath = `/claude-code/plugins/my-skill/archive/${digest}.zip`;
const url = `${getProxyBaseUrl()}${archivePath}`;
const skill: Plugin = {
  id: "private-skill",
  name: "my-skill",
  source: { source: "archive", url, sha256: digest },
  enabled: true,
  version: null,
  description: null,
  created_at: null,
  updated_at: null,
};
const fetchMock = vi.fn<typeof fetch>();
const skillRequestMock = vi.fn<typeof fetch>();
const renderSkill = (source = skill.source, isAdmin = false) =>
  renderWithProviders(
    <SkillDetail skill={{ ...skill, source }} onBack={vi.fn()} accessToken="dashboard-token" isAdmin={isAdmin} />,
  );
const downloadsUrl = `${getProxyBaseUrl()}/skill-hub/skills/my-skill/downloads?limit=25`;
const event = {
  id: "9007199254740993",
  downloaded_at: "2026-10-01T10:00:00Z",
  version: "1.2.0",
  digest,
  authenticated: true,
  user_id: "reader-id",
  user_email: "reader@example.test",
  key_alias: "development",
  team_id: "team-id",
  team_alias: "Engineering",
} satisfies PluginDownloadEvent;
const page = (
  downloads: PluginDownloadEvent[] = [],
  total_downloads = downloads.length,
  next_cursor: string | null = null,
  anonymous_downloads = 0,
): PluginDownloadsResponse => ({ downloads, total_downloads, next_cursor, anonymous_downloads });

beforeEach(() => {
  vi.clearAllMocks();
  testQueryClient.clear();
  skillRequestMock.mockReset().mockRejectedValue(new Error("Unexpected skill request"));
  fetchMock.mockReset().mockImplementation((input, init) => {
    const requestUrl = input instanceof Request ? input.url : String(input);
    if (requestUrl.includes("/.well-known/litellm-ui-config")) {
      return Promise.resolve(Response.json({ claude_code_marketplace_auth_required: true }));
    }
    return skillRequestMock(input, init);
  });
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  setGlobalLitellmHeaderName();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

it.each(["legacy", "portable"])(
  "retries a private %s ZIP download through the configured auth header and local route",
  async (kind) => {
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click");
    URL.createObjectURL = vi.fn(() => "blob:skill-download");
    setGlobalLitellmHeaderName("X-LiteLLM-Key");
    const expectedUrl = kind === "portable" ? `${getProxyBaseUrl()}/skill-hub/skills/my-skill/archive` : url;
    const detail =
      kind === "portable"
        ? skillHubDetail({ ...skill, visibility: "private", download_url: "/skill-hub/skills/my-skill/archive" })
        : skill;
    renderWithProviders(<SkillDetail skill={detail} onBack={vi.fn()} accessToken="dashboard-token" />);
    skillRequestMock.mockResolvedValueOnce(Response.json({ detail: "Archive not found" }, { status: 404 }));
    fireEvent.click(screen.getByRole("button", { name: "Download ZIP" }));
    expect(screen.getByRole("button", { name: "Downloading…" })).toBeDisabled();
    expect(await screen.findByRole("alert")).toHaveTextContent("Archive not found");
    expect(click).not.toHaveBeenCalled();
    skillRequestMock.mockResolvedValueOnce(
      new Response("zip-bytes", { headers: { "Content-Type": "application/zip" } }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Download ZIP" }));
    await waitFor(() => expect(click).toHaveBeenCalledOnce());
    expect((click.mock.contexts[0] as HTMLAnchorElement).download).toBe("my-skill.zip");
    expect(skillRequestMock).toHaveBeenLastCalledWith(
      expectedUrl,
      expect.objectContaining({
        method: "GET",
        headers: expect.objectContaining({ "X-LiteLLM-Key": "Bearer dashboard-token" }),
        redirect: "error",
      }),
    );
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  },
);

it("keeps an external archive as a link while retaining stored download history", async () => {
  const externalUrl = `https://external.example.com${archivePath}`;
  renderSkill({ ...skill.source, url: externalUrl }, true);
  expect(screen.getByRole("link")).toHaveAttribute("href", externalUrl);
  expect(screen.getByRole("link")).toHaveAttribute("target", "_blank");
  expect(screen.queryByRole("button", { name: "Download ZIP" })).not.toBeInTheDocument();
  skillRequestMock.mockResolvedValueOnce(Response.json(page([event])));
  fireEvent.click(screen.getByRole("tab", { name: "Downloads" }));
  expect(await screen.findByRole("cell", { name: event.user_email })).toBeInTheDocument();
  expect(screen.getByText(/External downloads are not tracked/)).toBeInTheDocument();
});

it("shows attribution and string cursors, retains rows on failure, and hides populated history on role loss", async () => {
  setGlobalLitellmHeaderName("X-LiteLLM-Key");
  const view = renderSkill(skill.source, true);
  skillRequestMock.mockResolvedValueOnce(Response.json(page([event], 7, event.id, 5)));
  fireEvent.click(screen.getByRole("tab", { name: "Downloads" }));
  const row = await screen.findByRole("row", { name: /reader@example.test/ });
  for (const value of ["development", "Engineering", "1.2.0", new Date(event.downloaded_at).toLocaleString("en")]) {
    expect(row).toHaveTextContent(value);
  }
  expect(screen.getByRole("definition", { name: "Total downloads" })).toHaveTextContent("7");
  expect(screen.getByRole("definition", { name: "Anonymous downloads" })).toHaveTextContent("5");
  expect(skillRequestMock).toHaveBeenLastCalledWith(
    downloadsUrl,
    expect.objectContaining({
      headers: expect.objectContaining({ "X-LiteLLM-Key": "Bearer dashboard-token" }),
    }),
  );
  skillRequestMock.mockResolvedValueOnce(Response.json({ detail: "Unavailable" }, { status: 500 }));
  fireEvent.click(screen.getByRole("button", { name: "Load more" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Could not load download history");
  expect(screen.getByRole("cell", { name: event.user_email })).toBeInTheDocument();
  skillRequestMock.mockResolvedValueOnce(Response.json(page([{ ...event, id: "2", user_email: null }], 7, null, 5)));
  fireEvent.click(screen.getByRole("button", { name: "Load more" }));
  expect(await screen.findByRole("cell", { name: event.user_id })).toBeInTheDocument();
  expect(skillRequestMock).toHaveBeenLastCalledWith(`${downloadsUrl}&cursor=${event.id}`, expect.anything());
  expect(screen.queryByRole("button", { name: "Load more" })).not.toBeInTheDocument();
  view.rerender(<SkillDetail skill={skill} onBack={vi.fn()} accessToken="dashboard-token" isAdmin={false} />);
  expect(screen.queryByRole("tab", { name: "Downloads" })).not.toBeInTheDocument();
  expect(screen.queryByRole("table", { name: "Download history" })).not.toBeInTheDocument();
});

it.each([
  { nextSkill: { ...skill, id: "recreated-skill" }, token: "dashboard-token" },
  { nextSkill: skill, token: "another-admin-token" },
])("clears old history while a new scope fails to load", async ({ nextSkill, token }) => {
  const view = renderSkill(skill.source, true);
  skillRequestMock.mockResolvedValueOnce(Response.json(page([event])));
  fireEvent.click(screen.getByRole("tab", { name: "Downloads" }));
  expect(await screen.findByRole("cell", { name: event.user_email })).toBeInTheDocument();
  const nextPage = Promise.withResolvers<Response>();
  skillRequestMock.mockReturnValueOnce(nextPage.promise);
  view.rerender(<SkillDetail skill={nextSkill} onBack={vi.fn()} accessToken={token} isAdmin />);
  expect(screen.getByRole("status")).toHaveTextContent("Loading download history");
  expect(screen.queryByRole("table", { name: "Download history" })).not.toBeInTheDocument();
  nextPage.resolve(Response.json({ detail: "Unavailable" }, { status: 500 }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Could not load download history");
  expect(screen.queryByText("Total downloads")).not.toBeInTheDocument();
});

it.each([skill.source, { source: "archive", url: `${getProxyBaseUrl()}/skill-hub/skills/my-skill/archive` }])(
  "refreshes canonical history after a successful ZIP download and refreshes anonymous-only totals",
  async (source) => {
    URL.createObjectURL = vi.fn(() => "blob:skill-download");
    renderSkill(source, true);
    skillRequestMock.mockResolvedValueOnce(Response.json(page()));
    fireEvent.click(screen.getByRole("tab", { name: "Downloads" }));
    expect(await screen.findByText("No authenticated download history yet.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "Overview" }));
    skillRequestMock
      .mockResolvedValueOnce(new Response("zip-bytes"))
      .mockImplementation(() => Promise.resolve(Response.json(page([event]))));
    fireEvent.click(screen.getByRole("button", { name: "Download ZIP" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Download ZIP" })).toBeEnabled());
    expect(skillRequestMock).toHaveBeenLastCalledWith(downloadsUrl, expect.anything());
    fireEvent.click(screen.getByRole("tab", { name: "Downloads" }));
    expect(await screen.findByRole("cell", { name: event.user_email })).toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("button", { name: "Refresh" })).toBeEnabled());
    skillRequestMock.mockResolvedValue(Response.json(page([], 5, null, 5)));
    fireEvent.click(screen.getByRole("button", { name: "Refresh" }));
    expect(await screen.findByText("No authenticated download history yet.")).toBeInTheDocument();
    expect(screen.getByRole("definition", { name: "Total downloads" })).toHaveTextContent("5");
    skillRequestMock.mockResolvedValueOnce(Response.json({ versions: [], next_offset: null }));
    fireEvent.click(screen.getByRole("tab", { name: "Version history" }));
    expect(await screen.findByText(/No recorded releases yet/)).toBeInTheDocument();
  },
);
