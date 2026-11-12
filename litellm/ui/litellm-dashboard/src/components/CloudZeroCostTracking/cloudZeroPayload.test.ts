import { describe, expect, it } from "vitest";

import { buildCloudZeroPayload } from "./cloudZeroPayload";

describe("buildCloudZeroPayload", () => {
  it("passes a filled form through verbatim", () => {
    expect(
      buildCloudZeroPayload({ api_key: "cz-key", connection_id: "conn-1", timezone: "America/New_York" }),
    ).toStrictEqual({
      connection_id: "conn-1",
      timezone: "America/New_York",
      api_key: "cz-key",
    });
  });

  it("omits the api_key key entirely when the field is blank, so a stored secret survives an untouched save", () => {
    const payload = buildCloudZeroPayload({ api_key: "", connection_id: "conn-1", timezone: "UTC" });

    expect("api_key" in payload).toBe(false);
    expect(payload).toStrictEqual({ connection_id: "conn-1", timezone: "UTC" });
  });

  it.each([
    ["blank", ""],
    ["absent", undefined],
  ])("falls back to UTC when the timezone is %s", (_label, timezone) => {
    const payload = buildCloudZeroPayload({
      api_key: "cz-key",
      connection_id: "conn-1",
      timezone: timezone as string,
    });

    expect(payload.timezone).toBe("UTC");
  });

  it("never invents a timezone default over a real value", () => {
    expect(buildCloudZeroPayload({ api_key: "", connection_id: "conn-1", timezone: "UTC+2" }).timezone).toBe("UTC+2");
  });

  it("keeps a whitespace-only api_key, matching the truthiness check the antd modals used", () => {
    expect(buildCloudZeroPayload({ api_key: " ", connection_id: "conn-1", timezone: "UTC" }).api_key).toBe(" ");
  });
});
