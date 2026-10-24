import { beforeEach, describe, expect, it, vi } from "vitest";

import { createUserFromRow, type BulkInviteApi, type CreatedUser } from "./createUser";
import type { CsvRow } from "./parseCsv";

const BASE_URL = "https://ui.example.test";

const row = (overrides: Partial<CsvRow> = {}): CsvRow => ({
  rowNumber: 2,
  user_email: "new.hire@example.com",
  user_role: "internal_user",
  isValid: true,
  error: "",
  ...overrides,
});

const fakeApi = (created: CreatedUser | null = { key: "sk-new", user_id: "user-new" }) => ({
  createUser: vi.fn<BulkInviteApi["createUser"]>().mockResolvedValue(created),
  createInvitation: vi.fn<BulkInviteApi["createInvitation"]>().mockResolvedValue({ id: "inv-123" }),
});

const invite = { ssoEnabled: false, baseUrl: BASE_URL };

describe("createUserFromRow", () => {
  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  describe("create-user request", () => {
    it("sends only email and role when the optional cells are absent", async () => {
      const api = fakeApi();
      await createUserFromRow(row(), api, invite);

      expect(api.createUser).toHaveBeenCalledExactlyOnceWith({
        user_email: "new.hire@example.com",
        user_role: "internal_user",
      });
    });

    it("splits comma-separated teams and models, trimming and dropping blanks", async () => {
      const api = fakeApi();
      const cells = {
        teams: "team-a, team-b,,",
        models: " gpt-4o , ,claude-x",
        max_budget: "12.5",
        budget_duration: "30d",
      };
      const expected = {
        user_email: "new.hire@example.com",
        user_role: "internal_user",
        teams: ["team-a", "team-b"],
        models: ["gpt-4o", "claude-x"],
        max_budget: 12.5,
        budget_duration: "30d",
      };

      await createUserFromRow(row(cells), api, invite);

      expect(api.createUser.mock.calls[0][0]).toStrictEqual(expected);
    });

    it("omits teams and models that contain only separators", async () => {
      const api = fakeApi();
      await createUserFromRow(row({ teams: " , ", models: "," }), api, invite);

      expect(api.createUser.mock.calls[0][0]).toStrictEqual({
        user_email: "new.hire@example.com",
        user_role: "internal_user",
      });
    });

    it.each(["", "abc", "0", "-5"])("drops a max budget of %j instead of sending it", async (maxBudget) => {
      const api = fakeApi();
      await createUserFromRow(row({ max_budget: maxBudget }), api, invite);

      expect(api.createUser.mock.calls[0][0]).not.toHaveProperty("max_budget");
    });

    it("drops an empty budget duration", async () => {
      const api = fakeApi();
      await createUserFromRow(row({ budget_duration: "" }), api, invite);

      expect(api.createUser.mock.calls[0][0]).not.toHaveProperty("budget_duration");
    });
  });

  describe("invitation link", () => {
    it("creates an invitation for the new user and links to onboarding", async () => {
      const api = fakeApi({ key: "sk-new", user_id: "user-new" });

      const outcome = await createUserFromRow(row(), api, invite);

      expect(api.createInvitation).toHaveBeenCalledExactlyOnceWith("user-new");
      expect(outcome).toStrictEqual({
        status: "success",
        key: "sk-new",
        invitation_link: `${BASE_URL}/ui/onboarding?invitation_id=inv-123`,
      });
    });

    it("prefers the nested data.user_id when inviting", async () => {
      const api = fakeApi({ key: "sk-new", user_id: "top-level", data: { user_id: "nested" } });

      await createUserFromRow(row(), api, invite);

      expect(api.createInvitation).toHaveBeenCalledExactlyOnceWith("nested");
    });

    it("links SSO deployments to the UI root without creating an invitation", async () => {
      const api = fakeApi();

      const outcome = await createUserFromRow(row(), api, { ssoEnabled: true, baseUrl: BASE_URL });

      expect(api.createInvitation).not.toHaveBeenCalled();
      expect(outcome).toStrictEqual({ status: "success", key: "sk-new", invitation_link: `${BASE_URL}/ui` });
    });

    it("still reports success with the key when the invitation fails", async () => {
      const api = fakeApi();
      api.createInvitation.mockRejectedValue(new Error("invite down"));

      const outcome = await createUserFromRow(row(), api, invite);

      expect(outcome).toStrictEqual({
        status: "success",
        key: "sk-new",
        error: "User created but failed to generate invitation link",
      });
    });
  });

  describe("result", () => {
    it("falls back to the user id as the key when no key is returned", async () => {
      const outcome = await createUserFromRow(row(), fakeApi({ user_id: "user-only" }), invite);

      expect(outcome).toMatchObject({ status: "success", key: "user-only" });
    });

    it.each([
      [{ error: "Email already exists" }, "Email already exists"],
      [{}, "Failed to create user"],
      [null, "Failed to create user"],
    ])("fails when the response %j has neither key nor user id", async (created, message) => {
      const api = fakeApi(created);

      const outcome = await createUserFromRow(row(), api, invite);

      expect(outcome).toStrictEqual({ status: "failed", error: message });
      expect(api.createInvitation).not.toHaveBeenCalled();
    });

    it("fails with the thrown error message when the request throws", async () => {
      const api = fakeApi();
      api.createUser.mockRejectedValue(new Error('{"error":"duplicate"}'));

      const outcome = await createUserFromRow(row(), api, invite);

      expect(outcome).toStrictEqual({ status: "failed", error: '{"error":"duplicate"}' });
    });

    it("stringifies a thrown value that is not an Error", async () => {
      const api = fakeApi();
      api.createUser.mockRejectedValue("network gone");

      const outcome = await createUserFromRow(row(), api, invite);

      expect(outcome).toStrictEqual({ status: "failed", error: "network gone" });
    });
  });
});
