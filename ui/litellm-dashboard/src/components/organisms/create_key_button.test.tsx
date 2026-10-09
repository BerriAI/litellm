import { beforeEach, describe, expect, it, vi } from "vitest";
import { modelAvailableCall } from "../networking";
import { fetchTeamModels, fetchUserModels } from "./create_key_button";

vi.mock("../networking", () => ({
  modelAvailableCall: vi.fn(),
}));

describe("fetchTeamModels", () => {
  beforeEach(() => {
    vi.mocked(modelAvailableCall).mockReset();
  });

  it("asks the proxy for the team-scoped model list and returns the ids", async () => {
    vi.mocked(modelAvailableCall).mockResolvedValue({ data: [{ id: "gpt-4" }, { id: "claude-opus-4" }] });

    await expect(fetchTeamModels("user-1", "Admin", "token-1", "team-1")).resolves.toStrictEqual([
      "gpt-4",
      "claude-opus-4",
    ]);
    expect(modelAvailableCall).toHaveBeenCalledWith("token-1", "user-1", "Admin", true, "team-1", true);
  });

  it("passes a null team through rather than dropping the argument", async () => {
    vi.mocked(modelAvailableCall).mockResolvedValue({ data: [] });

    await fetchTeamModels("user-1", "Admin", "token-1", null);

    expect(modelAvailableCall).toHaveBeenCalledWith("token-1", "user-1", "Admin", true, null, true);
  });

  it("returns an empty list and makes no call when the user id is null", async () => {
    await expect(fetchTeamModels(null as unknown as string, "Admin", "token-1", "team-1")).resolves.toStrictEqual([]);
    expect(modelAvailableCall).not.toHaveBeenCalled();
  });

  it("swallows a failed lookup and returns an empty list", async () => {
    vi.mocked(modelAvailableCall).mockRejectedValue(new Error("proxy down"));

    await expect(fetchTeamModels("user-1", "Admin", "token-1", "team-1")).resolves.toStrictEqual([]);
  });
});

describe("fetchUserModels", () => {
  beforeEach(() => {
    vi.mocked(modelAvailableCall).mockReset();
  });

  it("hands the returned ids to the setter without the team-scoped arguments", async () => {
    vi.mocked(modelAvailableCall).mockResolvedValue({ data: [{ id: "gpt-4" }] });
    const setUserModels = vi.fn();

    await fetchUserModels("user-1", "Admin", "token-1", setUserModels);

    expect(modelAvailableCall).toHaveBeenCalledWith("token-1", "user-1", "Admin");
    expect(setUserModels).toHaveBeenCalledWith(["gpt-4"]);
  });

  it("leaves the setter untouched when the lookup fails", async () => {
    vi.mocked(modelAvailableCall).mockRejectedValue(new Error("proxy down"));
    const setUserModels = vi.fn();

    await fetchUserModels("user-1", "Admin", "token-1", setUserModels);

    expect(setUserModels).not.toHaveBeenCalled();
  });

  it("makes no call when the user role is null", async () => {
    const setUserModels = vi.fn();

    await fetchUserModels("user-1", null as unknown as string, "token-1", setUserModels);

    expect(modelAvailableCall).not.toHaveBeenCalled();
    expect(setUserModels).not.toHaveBeenCalled();
  });
});
