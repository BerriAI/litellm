import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "@/lib/http/client";
import { registerBaseUrlGetter } from "@/lib/http/runtime";
import { createCustomerHierarchyEntity } from "./createCustomerHierarchyEntity";
import type { CustomerHierarchyCreate } from "./customerHierarchyForms";

const fetchMock = vi.fn<typeof fetch>();
const team = { customer_team_id: "org-1:default", parent_org_id: "org-1" };
const member = { customer_member_id: "member-1", parent_team_ids: [team.customer_team_id] };
const create: CustomerHierarchyCreate = {
  kind: "member",
  defaultTeamOrganizationId: "org-1",
  body: { customer_member_id: "member-1", parent_org_id: "org-1", parent_team_ids: [team.customer_team_id] },
};
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
const requests = () =>
  Promise.all(
    fetchMock.mock.calls.map(async ([input]) => {
      const request = input as Request;
      return {
        method: request.method,
        path: decodeURIComponent(new URL(request.url).pathname),
        body: request.method === "POST" ? await request.clone().json() : null,
      };
    }),
  );

describe("createCustomerHierarchyEntity", () => {
  beforeEach(() => {
    registerBaseUrlGetter(() => "http://localhost");
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => vi.unstubAllGlobals());

  it("reuses the default team and strictly creates the member in the requested organization", async () => {
    fetchMock.mockResolvedValueOnce(json(team)).mockResolvedValueOnce(json(member, 201));
    expect(await createCustomerHierarchyEntity(create)).toEqual({ kind: "member", record: member });
    expect(await requests()).toEqual([
      { method: "GET", path: "/management/v1/customer_teams/org-1:default", body: null },
      { method: "POST", path: "/management/v1/customer_members", body: create.body },
    ]);
  });

  it("creates a missing default team before creating the member", async () => {
    fetchMock
      .mockResolvedValueOnce(json({ detail: "Not found" }, 404))
      .mockResolvedValueOnce(json({ detail: "Not found" }, 404))
      .mockResolvedValueOnce(json(team, 201))
      .mockResolvedValueOnce(json(member, 201));
    expect(await createCustomerHierarchyEntity(create)).toEqual({ kind: "member", record: member });
    expect(await requests()).toEqual([
      { method: "GET", path: "/management/v1/customer_teams/org-1:default", body: null },
      { method: "GET", path: "/management/v1/customer_members/member-1", body: null },
      { method: "POST", path: "/management/v1/customer_teams", body: { ...team, alias: "default" } },
      { method: "POST", path: "/management/v1/customer_members", body: create.body },
    ]);
  });

  it("reuses a default team created concurrently", async () => {
    fetchMock
      .mockResolvedValueOnce(json({ detail: "Not found" }, 404))
      .mockResolvedValueOnce(json({ detail: "Not found" }, 404))
      .mockResolvedValueOnce(json({ detail: "Already exists" }, 409))
      .mockResolvedValueOnce(json(team))
      .mockResolvedValueOnce(json(member, 201));
    expect(await createCustomerHierarchyEntity(create)).toEqual({ kind: "member", record: member });
    expect((await requests()).map(({ method }) => method)).toEqual(["GET", "GET", "POST", "GET", "POST"]);
  });

  it("does not use a canonical team ID belonging to another organization", async () => {
    fetchMock.mockResolvedValueOnce(json({ ...team, parent_org_id: "another-org" }));
    expect(await createCustomerHierarchyEntity(create)).toBeNull();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it.each([403, 500])("does not create anything after a default team lookup fails with %s", async (status) => {
    fetchMock.mockResolvedValueOnce(json({ detail: "Unavailable" }, status));
    await expect(createCustomerHierarchyEntity(create)).rejects.toMatchObject({ status });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("propagates default team creation failures without attempting member creation", async () => {
    fetchMock
      .mockResolvedValueOnce(json({ detail: "Not found" }, 404))
      .mockResolvedValueOnce(json({ detail: "Not found" }, 404))
      .mockResolvedValueOnce(json({ detail: "Team limit reached" }, 403));
    await expect(createCustomerHierarchyEntity(create)).rejects.toMatchObject({ status: 403 });
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("preserves the selected team and rejects duplicate members without an upsert", async () => {
    fetchMock.mockResolvedValueOnce(json({ detail: "Customer entity ID already exists." }, 409));
    const scoped: CustomerHierarchyCreate = {
      kind: "member",
      body: { ...create.body, parent_team_ids: ["support"] },
    };
    await expect(createCustomerHierarchyEntity(scoped)).rejects.toBeInstanceOf(ApiError);
    expect(await requests()).toEqual([{ method: "POST", path: "/management/v1/customer_members", body: scoped.body }]);
  });

  it("rejects an existing member ID before creating a missing default team", async () => {
    fetchMock.mockImplementation(async (input) => {
      const request = input as Request;
      const path = new URL(request.url).pathname;
      if (request.method === "GET" && path.endsWith("/customer_members/member-1")) return json(member);
      if (request.method === "GET") return json({ detail: "Not found" }, 404);
      if (path.endsWith("/customer_teams")) return json(team, 201);
      return json({ detail: "Customer entity ID already exists." }, 409);
    });
    await expect(createCustomerHierarchyEntity(create)).rejects.toMatchObject({ status: 409 });
    expect((await requests()).filter(({ method }) => method !== "GET")).toEqual([]);
  });

  it.each([403, 500])("does not create a default team when member lookup fails with %s", async (status) => {
    fetchMock
      .mockResolvedValueOnce(json({ detail: "Not found" }, 404))
      .mockResolvedValueOnce(json({ detail: "Unavailable" }, status));
    await expect(createCustomerHierarchyEntity(create)).rejects.toMatchObject({ status });
    expect((await requests()).filter(({ method }) => method !== "GET")).toEqual([]);
  });

  it("creates a default team without a member lookup for a generated member ID", async () => {
    const generated: CustomerHierarchyCreate = { ...create, body: { ...create.body, customer_member_id: undefined } };
    fetchMock
      .mockResolvedValueOnce(json({ detail: "Not found" }, 404))
      .mockResolvedValueOnce(json(team, 201))
      .mockResolvedValueOnce(json(member, 201));
    expect(await createCustomerHierarchyEntity(generated)).toEqual({ kind: "member", record: member });
    expect((await requests()).map(({ method }) => method)).toEqual(["GET", "POST", "POST"]);
  });
});
