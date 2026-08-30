import { describe, expect, it, vi } from "vitest";

vi.mock("@/components/networking", () => ({ serverRootPath: "" }));

import { modelGroupHref } from "./entityLinks";

describe("modelGroupHref", () => {
  it("targets the models page filtered to the encoded model group", () => {
    expect(modelGroupHref("gpt-4.1")).toMatch(/\/models-and-endpoints\?model_group=gpt-4\.1$/);
    expect(modelGroupHref("openai/*")).toMatch(/\?model_group=openai%2F\*$/);
  });

  it.each(["all-proxy-models", "all-team-models", "no-default-models"])(
    "returns no href for the %s grant sentinel",
    (sentinel) => {
      expect(modelGroupHref(sentinel)).toBeUndefined();
    },
  );
});
