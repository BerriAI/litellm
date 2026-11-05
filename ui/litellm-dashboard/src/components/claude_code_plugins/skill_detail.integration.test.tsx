import { fireEvent, screen, waitFor } from "@testing-library/react";
import { renderWithProviders, testQueryClient } from "@/../tests/test-utils";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { getProxyBaseUrl, setGlobalLitellmHeaderName } from "@/components/networking";
import SkillDetail, { skillHubDetail } from "./skill_detail";
import type { Plugin } from "./types";

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
const renderSkill = (source = skill.source) =>
  renderWithProviders(<SkillDetail skill={{ ...skill, source }} onBack={vi.fn()} accessToken="dashboard-token" />);

beforeEach(() => {
  vi.clearAllMocks();
  testQueryClient.clear();
  fetchMock.mockReset().mockResolvedValue(Response.json({ claude_code_marketplace_auth_required: true }));
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
    fetchMock.mockResolvedValueOnce(Response.json({ detail: "Archive not found" }, { status: 404 }));
    fireEvent.click(screen.getByRole("button", { name: "Download ZIP" }));
    expect(screen.getByRole("button", { name: "Downloading…" })).toBeDisabled();
    expect(await screen.findByRole("alert")).toHaveTextContent("Archive not found");
    expect(click).not.toHaveBeenCalled();
    fetchMock.mockResolvedValueOnce(new Response("zip-bytes", { headers: { "Content-Type": "application/zip" } }));
    fireEvent.click(screen.getByRole("button", { name: "Download ZIP" }));
    await waitFor(() => expect(click).toHaveBeenCalledOnce());
    expect((click.mock.contexts[0] as HTMLAnchorElement).download).toBe("my-skill.zip");
    expect(fetchMock).toHaveBeenLastCalledWith(
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

it("keeps an external archive as an ordinary link even when its path matches the gateway", () => {
  const externalUrl = `https://external.example.com${archivePath}`;
  renderSkill({ ...skill.source, url: externalUrl });
  expect(screen.getByRole("link")).toHaveAttribute("href", externalUrl);
  expect(screen.getByRole("link")).toHaveAttribute("target", "_blank");
  expect(screen.queryByRole("button", { name: "Download ZIP" })).not.toBeInTheDocument();
});
