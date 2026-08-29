import { describe, expect, it } from "vitest";
import type { DeploymentInfoRow } from "@/components/networking";
import {
  aliasAdditionsFromRows,
  buildDiscoveredRows,
  buildManualRow,
  buildModelCreationPayload,
  litellmModelForUpstreamId,
  mergeModelGroupAliases,
  rowsPendingCreation,
  type DiscoveredModelRow,
} from "./wizardLogic";

describe("litellmModelForUpstreamId", () => {
  it("prefixes a bare upstream id with the provider", () => {
    expect(litellmModelForUpstreamId("anthropic", "claude-3-opus")).toBe("anthropic/claude-3-opus");
  });

  it("does not double-prefix an id that already carries the provider prefix", () => {
    expect(litellmModelForUpstreamId("anthropic", "anthropic/claude-3-opus")).toBe("anthropic/claude-3-opus");
  });
});

describe("buildDiscoveredRows", () => {
  it("defaults every row to enabled, with model_name equal to the discovered id", () => {
    const rows = buildDiscoveredRows(["claude-3-opus", "claude-3-haiku"]);
    expect(rows).toHaveLength(2);
    const defaultedOpusRow = {
      upstreamId: "claude-3-opus",
      modelName: "claude-3-opus",
      enabled: true,
      alternateNames: [],
      manual: false,
    };
    expect(rows[0]).toMatchObject(defaultedOpusRow);
  });

  it("gives every row a unique id even for duplicate upstream ids", () => {
    const rows = buildDiscoveredRows(["same-id", "same-id"]);
    expect(rows[0].id).not.toBe(rows[1].id);
  });
});

describe("buildManualRow", () => {
  it("marks the row manual and enabled by default", () => {
    const row = buildManualRow("hidden-model");
    const manualEnabledRow = { upstreamId: "hidden-model", modelName: "hidden-model", enabled: true, manual: true };
    expect(row).toMatchObject(manualEnabledRow);
  });
});

describe("buildModelCreationPayload", () => {
  const baseRow: DiscoveredModelRow = {
    id: "row-1",
    upstreamId: "claude-3-opus",
    modelName: "my-claude",
    enabled: true,
    alternateNames: [],
    manual: false,
  };

  it("maps enabled=true to blocked=false", () => {
    const payload = buildModelCreationPayload("anthropic", "my-cred", baseRow);
    const unblockedCreation = {
      model_name: "my-claude",
      litellm_params: { model: "anthropic/claude-3-opus", litellm_credential_name: "my-cred" },
      model_info: {},
      blocked: false,
    };
    expect(payload).toEqual(unblockedCreation);
  });

  it("maps enabled=false to blocked=true", () => {
    const payload = buildModelCreationPayload("anthropic", "my-cred", { ...baseRow, enabled: false });
    expect(payload.blocked).toBe(true);
  });
});

describe("rowsPendingCreation", () => {
  const rows: DiscoveredModelRow[] = [
    { id: "1", upstreamId: "claude-3-opus", modelName: "opus", enabled: true, alternateNames: [], manual: false },
    { id: "2", upstreamId: "claude-3-haiku", modelName: "haiku", enabled: true, alternateNames: [], manual: false },
  ];

  it("returns every row when nothing exists yet", () => {
    expect(rowsPendingCreation(rows, "anthropic", "my-cred", [])).toHaveLength(2);
  });

  it("skips a row already created under the same credential", () => {
    const existing: DeploymentInfoRow[] = [
      {
        model_name: "opus",
        litellm_params: { model: "anthropic/claude-3-opus", litellm_credential_name: "my-cred" },
        model_info: { id: "abc" },
      },
    ];
    const pending = rowsPendingCreation(rows, "anthropic", "my-cred", existing);
    expect(pending.map((r) => r.upstreamId)).toEqual(["claude-3-haiku"]);
  });

  it("does not skip a same-model row that belongs to a different credential", () => {
    const existing: DeploymentInfoRow[] = [
      {
        model_name: "opus",
        litellm_params: { model: "anthropic/claude-3-opus", litellm_credential_name: "someone-elses-cred" },
        model_info: { id: "abc" },
      },
    ];
    expect(rowsPendingCreation(rows, "anthropic", "my-cred", existing)).toHaveLength(2);
  });
});

describe("mergeModelGroupAliases", () => {
  it("adds new aliases to an empty map", () => {
    const { merged, collisions } = mergeModelGroupAliases({}, [{ alias: "gpt-4o", targetModelGroup: "opus" }]);
    expect(merged).toEqual({ "gpt-4o": "opus" });
    expect(collisions).toEqual([]);
  });

  it("preserves an existing object-valued entry untouched", () => {
    const existing = { "hidden-alias": { model: "some-model", hidden: true } };
    const { merged } = mergeModelGroupAliases(existing, [{ alias: "new-alias", targetModelGroup: "opus" }]);
    expect(merged["hidden-alias"]).toEqual({ model: "some-model", hidden: true });
    expect(merged["new-alias"]).toBe("opus");
  });

  it("rejects a collision with an existing alias rather than overwriting it", () => {
    const existing = { "gpt-4o": "some-other-model" };
    const { merged, collisions } = mergeModelGroupAliases(existing, [{ alias: "gpt-4o", targetModelGroup: "opus" }]);
    expect(merged["gpt-4o"]).toBe("some-other-model");
    expect(collisions).toEqual(["gpt-4o"]);
  });

  it("rejects a collision between two additions in the same batch", () => {
    const { merged, collisions } = mergeModelGroupAliases({}, [
      { alias: "dup", targetModelGroup: "opus" },
      { alias: "dup", targetModelGroup: "haiku" },
    ]);
    expect(merged.dup).toBe("opus");
    expect(collisions).toEqual(["dup"]);
  });
});

describe("aliasAdditionsFromRows", () => {
  it("flattens each row's alternate names against its model_name", () => {
    const rows: DiscoveredModelRow[] = [
      { id: "1", upstreamId: "opus", modelName: "my-opus", enabled: true, alternateNames: ["gpt-4o"], manual: false },
      { id: "2", upstreamId: "haiku", modelName: "my-haiku", enabled: true, alternateNames: [], manual: false },
    ];
    expect(aliasAdditionsFromRows(rows)).toEqual([{ alias: "gpt-4o", targetModelGroup: "my-opus" }]);
  });
});
