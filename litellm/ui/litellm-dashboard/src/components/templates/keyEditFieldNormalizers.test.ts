import { describe, expect, it } from "vitest";

import { moveBudgetAlertEmailsOutOfMetadataJson, moveTagsOutOfMetadataJson } from "./keyEditFieldNormalizers";

describe("moveTagsOutOfMetadataJson", () => {
  it("moves a tags array out of the JSON and appends it to the current tags", () => {
    expect(moveTagsOutOfMetadataJson('{"tags": ["pilot-tag"], "env": "non-prod"}', ["ui-tag"])).toEqual({
      metadata: '{\n  "env": "non-prod"\n}',
      tags: ["ui-tag", "pilot-tag"],
      movedTags: ["pilot-tag"],
    });
  });

  it("drops duplicates and non-string entries without reporting them as moved", () => {
    expect(moveTagsOutOfMetadataJson('{"tags": ["a", "a", "ui-tag", 7, null]}', ["ui-tag"])).toEqual({
      metadata: "{}",
      tags: ["ui-tag", "a"],
      movedTags: ["a"],
    });
  });

  it("trims whitespace and drops blank entries the same way the Tags control does", () => {
    expect(moveTagsOutOfMetadataJson('{"tags": [" a ", "a", "   ", "", " ui-tag"]}', ["ui-tag"])).toEqual({
      metadata: "{}",
      tags: ["ui-tag", "a"],
      movedTags: ["a"],
    });
  });

  it("strips an empty tags array while moving nothing", () => {
    expect(moveTagsOutOfMetadataJson('{"tags": []}', undefined)).toEqual({
      metadata: "{}",
      tags: [],
      movedTags: [],
    });
  });

  it("is a no-op when there is no tags array to move", () => {
    expect(moveTagsOutOfMetadataJson('{"env": "prod"}', ["a"])).toBeNull();
    expect(moveTagsOutOfMetadataJson('{"tags": "not-an-array"}', ["a"])).toBeNull();
    expect(moveTagsOutOfMetadataJson("", ["a"])).toBeNull();
    expect(moveTagsOutOfMetadataJson(undefined, ["a"])).toBeNull();
  });

  it("leaves invalid or non-object JSON alone so the save path can report it", () => {
    expect(moveTagsOutOfMetadataJson('{"tags": [', ["a"])).toBeNull();
    expect(moveTagsOutOfMetadataJson('["tags"]', ["a"])).toBeNull();
    expect(moveTagsOutOfMetadataJson("null", ["a"])).toBeNull();
  });
});

describe("moveBudgetAlertEmailsOutOfMetadataJson", () => {
  it("moves the typed thresholds into the rows, letting a typed entry replace the row with its threshold", () => {
    const typed = JSON.stringify({
      env: "prod",
      max_budget_alert_emails: { "80": "typed@example.com, dl@example.com", "90": null },
    });
    const rows = [
      { threshold: 50, emails: "kept@example.com" },
      { threshold: 80, emails: "replaced@example.com" },
    ];

    expect(moveBudgetAlertEmailsOutOfMetadataJson(typed, rows)).toEqual({
      metadata: '{\n  "env": "prod"\n}',
      rows: [
        { threshold: 50, emails: "kept@example.com" },
        { threshold: 80, emails: "typed@example.com, dl@example.com" },
        { threshold: 90, emails: "" },
      ],
      movedThresholds: [80, 90],
    });
  });

  it("leaves the typed entries the rows cannot show in the JSON", () => {
    const typed = JSON.stringify({ max_budget_alert_emails: { "80": ["a@example.com"], "150": ["b@example.com"] } });

    expect(moveBudgetAlertEmailsOutOfMetadataJson(typed, [])).toEqual({
      metadata: JSON.stringify({ max_budget_alert_emails: { "150": ["b@example.com"] } }, null, 2),
      rows: [{ threshold: 80, emails: "a@example.com" }],
      movedThresholds: [80],
    });
  });

  it("returns null when nothing typed can become a row", () => {
    expect(
      moveBudgetAlertEmailsOutOfMetadataJson('{"max_budget_alert_emails": {"150": ["b@example.com"]}}', []),
    ).toBeNull();
    expect(moveBudgetAlertEmailsOutOfMetadataJson('{"env": "prod"}', [])).toBeNull();
    expect(moveBudgetAlertEmailsOutOfMetadataJson('{"max_budget_alert_emails": {', [])).toBeNull();
    expect(moveBudgetAlertEmailsOutOfMetadataJson(undefined, [])).toBeNull();
  });
});
