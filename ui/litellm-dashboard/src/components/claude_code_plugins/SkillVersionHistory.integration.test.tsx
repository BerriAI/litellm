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
