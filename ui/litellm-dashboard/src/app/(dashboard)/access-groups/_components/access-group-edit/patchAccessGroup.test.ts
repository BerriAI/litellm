import { describe, expect, it, vi } from "vitest";

const patchMock = vi.fn();
vi.mock("@/lib/http/api", () => ({ fetchClient: { PATCH: (...args: unknown[]) => patchMock(...args) } }));

import { patchAccessGroup } from "./patchAccessGroup";

const record = { access_group_id: "ag-1", access_group_name: "prod", access_model_names: ["gpt-5.2"] };

describe("patchAccessGroup", () => {
  it("PATCHes the control-plane route with the id in the path and unwraps the data envelope", async () => {
    patchMock.mockResolvedValueOnce({ data: { data: record } });

    const result = await patchAccessGroup("ag-1", { description: null });

    expect(patchMock).toHaveBeenCalledWith("/management/v1/access-groups/{access_group_id}", {
      params: { path: { access_group_id: "ag-1" } },
      body: { description: null },
    });
    expect(result).toBe(record);
  });

  it("returns undefined when the response has no body", async () => {
    patchMock.mockResolvedValueOnce({ data: undefined });

    await expect(patchAccessGroup("ag-1", { description: "x" })).resolves.toBeUndefined();
  });
});
