import { describe, expect, it, vi } from "vitest";
import { fetchTeamFilterOptions } from "./filter_helpers";

const mockKeyListCall = vi.fn();

vi.mock("@/components/networking", () => ({
  keyListCall: (...args: unknown[]) => mockKeyListCall(...args),
  teamListCall: vi.fn(),
  organizationListCall: vi.fn(),
}));

describe("fetchTeamFilterOptions", () => {
  it("should return empty arrays when accessToken is null", async () => {
    const result = await fetchTeamFilterOptions(null, "team-1");

    expect(result).toEqual({ keyAliases: [], organizationIds: [], userIds: [] });
    expect(mockKeyListCall).not.toHaveBeenCalled();
  });

  it("should return empty arrays when teamId is empty", async () => {
    const result = await fetchTeamFilterOptions("tok-123", "");

    expect(result).toEqual({ keyAliases: [], organizationIds: [], userIds: [] });
    expect(mockKeyListCall).not.toHaveBeenCalled();
  });

  it("should return sorted key aliases from fetched keys", async () => {
    mockKeyListCall.mockResolvedValue({
      keys: [
        { key_alias: "zeta-key" },
        { key_alias: "alpha-key" },
        { key_alias: "mid-key" },
      ],
      total_pages: 1,
    });

    const result = await fetchTeamFilterOptions("tok-123", "team-1");

    expect(result.keyAliases).toEqual(["alpha-key", "mid-key", "zeta-key"]);
  });

  it("should deduplicate organization IDs across pages", async () => {
    mockKeyListCall
      .mockResolvedValueOnce({
        keys: [
          { organization_id: "org-b" },
          { organization_id: "org-a" },
        ],
        total_pages: 2,
      })
      .mockResolvedValueOnce({
        keys: [
          { organization_id: "org-a" },
          { organization_id: "org-c" },
        ],
        total_pages: 2,
      });

    const result = await fetchTeamFilterOptions("tok-123", "team-1");

    expect(result.organizationIds).toEqual(["org-a", "org-b", "org-c"]);
  });

  it("should map user IDs with email addresses", async () => {
    mockKeyListCall.mockResolvedValue({
      keys: [
        { user_id: "u1", user: { user_email: "alice@example.com" } },
        { user_id: "u2", user: { user_email: "bob@example.com" } },
      ],
      total_pages: 1,
    });

    const result = await fetchTeamFilterOptions("tok-123", "team-1");

    expect(result.userIds).toEqual(
      expect.arrayContaining([
        { id: "u1", email: "alice@example.com" },
        { id: "u2", email: "bob@example.com" },
      ]),
    );
  });

  it("should handle API errors gracefully and return empty arrays", async () => {
    mockKeyListCall.mockRejectedValue(new Error("Network error"));

    const result = await fetchTeamFilterOptions("tok-123", "team-1");

    expect(result).toEqual({ keyAliases: [], organizationIds: [], userIds: [] });
  });
});
