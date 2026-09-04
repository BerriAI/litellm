import { describe, expect, it } from "vitest";
import {
  federationIdsUpdate,
  missingFederationFields,
  readFederationIds,
  withFederationIds,
} from "./anthropicFederation";

const ALL_IDS = {
  anthropic_organization_id: "org-1",
  anthropic_federation_rule_id: "fdrl_1",
  anthropic_service_account_id: "svac_1",
  anthropic_workspace_id: "wrkspc_1",
};

const ids = (overrides: Partial<Record<string, string>> = {}) => readFederationIds({ ...ALL_IDS, ...overrides });

describe("readFederationIds", () => {
  it("reads the four ids, trimming pasted whitespace and treating anything else as blank", () => {
    const formValues = {
      anthropic_organization_id: "  org-1 ",
      anthropic_federation_rule_id: undefined,
      anthropic_service_account_id: 42,
      api_key: "unrelated",
    };
    const onlyOrganization = {
      anthropic_organization_id: "org-1",
      anthropic_federation_rule_id: "",
      anthropic_service_account_id: "",
      anthropic_workspace_id: "",
    };
    expect(readFederationIds(formValues)).toEqual(onlyOrganization);
  });
});

describe("missingFederationFields", () => {
  it("names only the organization and federation rule when everything is blank", () => {
    expect(missingFederationFields(readFederationIds({}))).toEqual(["Organization ID", "Federation Rule ID"]);
  });

  it("does not require the service account or workspace ids", () => {
    expect(missingFederationFields(ids({ anthropic_service_account_id: "", anthropic_workspace_id: "" }))).toEqual([]);
  });

  it("treats whitespace as blank", () => {
    expect(missingFederationFields(ids({ anthropic_federation_rule_id: "   " }))).toEqual(["Federation Rule ID"]);
  });
});

describe("federationIdsUpdate", () => {
  it("is null when the entered ids match the saved credential", () => {
    expect(
      federationIdsUpdate(
        { anthropic_organization_id: "org-1", anthropic_issuer_url: "https://proxy.example.com" },
        ids({ anthropic_federation_rule_id: "", anthropic_service_account_id: "", anthropic_workspace_id: "" }),
      ),
    ).toBeNull();
  });

  it("sends only the ids that changed and keeps the untouched saved ones out of the payload", () => {
    expect(
      federationIdsUpdate(
        { anthropic_organization_id: "org-1" },
        ids({ anthropic_service_account_id: "", anthropic_workspace_id: "" }),
      ),
    ).toEqual({ credential_values: { anthropic_federation_rule_id: "fdrl_1" }, credential_values_to_delete: [] });
  });

  it("deletes an id the operator cleared after it was saved instead of merging over it", () => {
    expect(
      federationIdsUpdate(
        { anthropic_organization_id: "org-1", anthropic_federation_rule_id: "fdrl_1", anthropic_workspace_id: "w" },
        ids({ anthropic_service_account_id: "", anthropic_workspace_id: "" }),
      ),
    ).toEqual({ credential_values: {}, credential_values_to_delete: ["anthropic_workspace_id"] });
  });

  it("ignores whitespace-only differences against the saved value", () => {
    expect(federationIdsUpdate({ anthropic_organization_id: " org-1" }, ids())).toEqual({
      credential_values: {
        anthropic_federation_rule_id: "fdrl_1",
        anthropic_service_account_id: "svac_1",
        anthropic_workspace_id: "wrkspc_1",
      },
      credential_values_to_delete: [],
    });
  });
});

describe("withFederationIds", () => {
  it("replaces the saved ids with the entered ones, dropping cleared ids and keeping other fields", () => {
    const saved = { anthropic_organization_id: "old", anthropic_workspace_id: "w", anthropic_issuer_url: "u" };
    const merged = {
      anthropic_issuer_url: "u",
      anthropic_organization_id: "org-1",
      anthropic_federation_rule_id: "fdrl_1",
      anthropic_service_account_id: "svac_1",
    };
    expect(withFederationIds(saved, ids({ anthropic_workspace_id: "" }))).toEqual(merged);
  });
});
