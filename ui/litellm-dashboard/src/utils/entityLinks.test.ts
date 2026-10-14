import { describe, expect, it, vi } from "vitest";

vi.mock("@/components/networking", () => ({ serverRootPath: "" }));

import { accessGroupHref, modelGroupHref, modelOrAccessGroupHref, teamDetailHref, userDetailHref } from "./entityLinks";

describe("userDetailHref", () => {
  it("targets the users page filtered to the encoded user id", () => {
    expect(userDetailHref("user-1")).toMatch(/\/users\?user=user-1$/);
    expect(userDetailHref("a b/c")).toMatch(/\?user=a%20b%2Fc$/);
  });

  it("returns no href for the proxy admin placeholder, which has no user page", () => {
    expect(userDetailHref("default_user_id")).toBeUndefined();
  });
});

describe("teamDetailHref", () => {
  it("targets the teams page filtered to the encoded team id", () => {
    expect(teamDetailHref("team-1")).toMatch(/\/teams\?team=team-1$/);
    expect(teamDetailHref("a b/c")).toMatch(/\?team=a%20b%2Fc$/);
  });

  it("returns no href for the Admin UI session team, which has no team page", () => {
    expect(teamDetailHref("litellm-dashboard")).toBeUndefined();
  });
});

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

describe("accessGroupHref", () => {
  it("targets the models page filtered to the encoded access group", () => {
    expect(accessGroupHref("a b/c")).toMatch(/\/models-and-endpoints\?access_group=a%20b%2Fc$/);
  });
});

describe("modelOrAccessGroupHref", () => {
  it("uses the access-group filter for a known access group", () => {
    expect(modelOrAccessGroupHref("repro-access-group", new Set(["repro-access-group"]))).toMatch(
      /\?access_group=repro-access-group$/,
    );
  });

  it("uses the model-group filter for a name outside the access-group set", () => {
    expect(modelOrAccessGroupHref("gpt-4.1", new Set(["repro-access-group"]))).toMatch(/\?model_group=gpt-4\.1$/);
  });

  it("keeps grant sentinels without a link unless they are access groups", () => {
    expect(modelOrAccessGroupHref("all-team-models", new Set())).toBeUndefined();
  });

  it("keeps model names unlinked until access group names are available", () => {
    expect(modelOrAccessGroupHref("repro-access-group", undefined)).toBeUndefined();
    expect(modelOrAccessGroupHref("gpt-4.1", undefined)).toBeUndefined();
  });
});
