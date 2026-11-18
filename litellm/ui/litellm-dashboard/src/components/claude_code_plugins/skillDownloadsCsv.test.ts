import Papa from "papaparse";
import { describe, expect, it } from "vitest";
import { buildSkillDownloadsCsv } from "./skillDownloadsCsv";
import type { PluginDownloadEvent } from "./types";

const event: PluginDownloadEvent = {
  id: "123",
  downloaded_at: "2026-10-01T12:00:00+02:00",
  authenticated: true,
  user_id: "reader",
  user_email: "reader@example.test",
  key_alias: "key",
  team_id: "team-id",
  team_alias: "Team",
  version: "1.0",
  digest: "abc",
};

describe("buildSkillDownloadsCsv", () => {
  it("exports ISO timestamps and all attribution fields", () => {
    const result = Papa.parse(buildSkillDownloadsCsv([event]), { header: true });
    expect(result.errors).toEqual([]);
    expect(result.data).toEqual([
      {
        downloaded_at: "2026-10-01T10:00:00.000Z",
        downloader: "reader@example.test",
        key: "key",
        team: "Team",
        version: "1.0",
        authenticated: "true",
        user_id: "reader",
        user_email: "reader@example.test",
        team_id: "team-id",
        id: "123",
        digest: "abc",
      },
    ]);
  });

  it("exports headers for empty history", () => {
    expect(buildSkillDownloadsCsv([])).toBe(
      "downloaded_at,downloader,key,team,version,authenticated,user_id,user_email,team_id,id,digest\r\n",
    );
  });

  it.each(["=SUM(1,2)", "+123", "-123", "@name", "\tformula", "\rformula"])("neutralizes formula %s", (key_alias) => {
    const result = Papa.parse<{ key: string }>(buildSkillDownloadsCsv([{ ...event, key_alias }]), { header: true });
    expect(result.data[0].key).toBe(`'${key_alias}`);
  });

  it("preserves quotes, commas, newlines and Unicode with null attribution", () => {
    const result = Papa.parse(
      buildSkillDownloadsCsv([
        {
          ...event,
          user_email: null,
          key_alias: 'clé, "quoted"\nnext',
          team_alias: null,
          version: null,
        },
      ]),
      { header: true },
    );
    expect(result.errors).toEqual([]);
    const expected = {
      downloader: "reader",
      key: 'clé, "quoted"\nnext',
      team: "team-id",
      version: "",
    };
    expect(result.data[0]).toMatchObject(expected);
  });
});
