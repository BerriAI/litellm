import { fireEvent, screen } from "@testing-library/react";
import { renderWithProviders, testQueryClient } from "@/../tests/test-utils";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import SkillVersionHistory from "./SkillVersionHistory";

const skill = {
  id: "versioned-id",
  name: "versioned",
};
const fetchMock = vi.fn<typeof fetch>();
beforeEach(() => {
  testQueryClient.clear();
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => vi.unstubAllGlobals());

it("distinguishes retained bytes from external metadata and paginates publications", async () => {
  fetchMock.mockResolvedValueOnce(
    Response.json({
      versions: [
        {
          id: "new",
          version: "1.0.1",
          description: "Revised instructions",
          manifest: { source: { source: "github", repo: "example/skill" } },
          published_at: null,
          published_by: null,
          content_kind: "source_reference",
          download_url: null,
        },
        {
          id: "old",
          version: "1.0.0",
          description: null,
          manifest: {},
          published_at: null,
          published_by: null,
          content_kind: "stored_archive",
          download_url: "/skill-hub/skills/versioned/versions/old/archive",
        },
      ],
      next_offset: 20,
    }),
  );
  renderWithProviders(<SkillVersionHistory skill={skill} accessToken="test-token" />);
  expect(await screen.findByText("1.0.1")).toBeInTheDocument();
  expect(String(fetchMock.mock.calls[0][0])).toContain("/skill-hub/skills/versioned/versions");
  expect(screen.getByText(/External files may change/)).toBeInTheDocument();
  expect(screen.getByText(/Uploaded ZIP retained/)).toBeInTheDocument();
  expect(screen.getAllByRole("button", { name: "Download retained ZIP" })).toHaveLength(1);
  fetchMock.mockResolvedValueOnce(Response.json({ versions: [], next_offset: null }));
  fireEvent.click(screen.getByRole("button", { name: "Older" }));
  expect(await screen.findByText(/No recorded releases yet/)).toBeInTheDocument();
  expect(screen.queryByText("1.0.1")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Older" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "Newer" })).toBeEnabled();
});

it("downloads through the neutral API client and reports a failed retained download", async () => {
  fetchMock
    .mockResolvedValueOnce(
      Response.json({
        versions: [
          {
            id: "retained",
            version: "1.0.0",
            description: null,
            manifest: {},
            published_at: null,
            published_by: null,
            content_kind: "stored_archive",
            download_url: "https://proxy.example/skill-hub/skills/versioned/versions/retained/archive",
          },
        ],
        next_offset: null,
      }),
    )
    .mockResolvedValueOnce(new Response("Archive unavailable", { status: 503 }));
  renderWithProviders(<SkillVersionHistory skill={skill} accessToken="test-token" />);
  fireEvent.click(await screen.findByRole("button", { name: "Download retained ZIP" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Could not download this version.");
  expect(String(fetchMock.mock.calls[1][0])).toContain("/skill-hub/skills/versioned/versions/retained/archive");
  expect(String(fetchMock.mock.calls[1][0])).not.toContain("https://proxy.example");
});

const retained = {
  id: "retained",
  version: "1.0.0",
  description: "Original instructions",
  manifest: {},
  published_at: null,
  published_by: null,
  content_kind: "stored_archive",
  download_url: "/skill-hub/skills/versioned/versions/retained/archive",
};
const currentSkill = { ...skill, version: "2.0.0", updated_at: "2026-01-02T00:00:00Z" };

it.each([false, true])("only offers retained older versions to a writable admin (%s)", async (isAdmin) => {
  fetchMock.mockResolvedValueOnce(
    Response.json({
      versions: [
        retained,
        { ...retained, id: "current", version: "2.0.0" },
        { ...retained, id: "external", version: "0.1.0", content_kind: "source_reference", download_url: null },
      ],
      next_offset: null,
    }),
  );
  renderWithProviders(
    <SkillVersionHistory skill={currentSkill} accessToken="test-token" isAdmin={isAdmin} onRestored={vi.fn()} />,
  );
  expect(await screen.findByText("1.0.0")).toBeInTheDocument();
  expect(screen.queryAllByRole("button", { name: "Restore this version" })).toHaveLength(isAdmin ? 1 : 0);
});

it("confirms a restore with the editor snapshot, updates the detail and refreshes history", async () => {
  const onRestored = vi.fn();
  const restored = { ...currentSkill, version: "2.0.1", description: "Original instructions" };
  fetchMock
    .mockResolvedValueOnce(Response.json({ versions: [retained], next_offset: null }))
    .mockResolvedValueOnce(Response.json(restored))
    .mockResolvedValueOnce(
      Response.json({ versions: [{ ...retained, id: "restored", version: "2.0.1" }, retained], next_offset: null }),
    );
  renderWithProviders(
    <SkillVersionHistory skill={currentSkill} accessToken="test-token" isAdmin onRestored={onRestored} />,
  );
  fireEvent.click(await screen.findByRole("button", { name: "Restore this version" }));
  expect(screen.getByRole("dialog")).toHaveTextContent("Existing versions, access permissions");
  expect(fetchMock).toHaveBeenCalledTimes(1);
  fireEvent.change(screen.getByRole("textbox", { name: "New version (optional)" }), { target: { value: "2.0.1" } });
  fireEvent.click(screen.getByRole("button", { name: "Restore as new version" }));
  expect(await screen.findByText("2.0.1")).toBeInTheDocument();
  expect(onRestored).toHaveBeenCalledWith(restored);
  expect(String(fetchMock.mock.calls[1][0])).toContain("/skill-hub/skills/versioned/versions/retained/restore");
  expect(fetchMock.mock.calls[1][1]).toMatchObject({
    method: "POST",
    headers: { Authorization: "Bearer test-token" },
    body: JSON.stringify({
      expected_skill_id: skill.id,
      expected_updated_at: currentSkill.updated_at,
      version: "2.0.1",
    }),
  });
});

it("shows stale-restore errors without reporting success", async () => {
  const onRestored = vi.fn();
  fetchMock
    .mockResolvedValueOnce(Response.json({ versions: [retained], next_offset: null }))
    .mockResolvedValueOnce(
      Response.json({ detail: "The published skill changed. Reload it before restoring a version" }, { status: 409 }),
    );
  renderWithProviders(
    <SkillVersionHistory skill={currentSkill} accessToken="test-token" isAdmin onRestored={onRestored} />,
  );
  fireEvent.click(await screen.findByRole("button", { name: "Restore this version" }));
  fireEvent.click(screen.getByRole("button", { name: "Restore as new version" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("The published skill changed");
  expect(onRestored).not.toHaveBeenCalled();
});
