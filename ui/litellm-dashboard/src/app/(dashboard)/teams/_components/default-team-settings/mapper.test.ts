import { describe, expect, it } from "vitest";

import { buildBody, settingsToForm } from "./mapper";
import type { DefaultTeamSettingsFormValues } from "./schema";

const CONFIGURED_SETTINGS = {
  max_budget: 100.5,
  budget_duration: "30d",
  tpm_limit: 1000,
  rpm_limit: 50,
  models: ["gpt-5.2"],
  team_member_permissions: ["/key/generate", "/key/update"],
};

const UNCONFIGURED_SETTINGS = {
  max_budget: null,
  budget_duration: null,
  tpm_limit: null,
  rpm_limit: null,
  models: null,
  team_member_permissions: null,
};

const CONFIGURED_FORM = {
  max_budget: "100.5",
  budget_duration: "30d",
  tpm_limit: "1000",
  rpm_limit: "50",
  models: ["gpt-5.2"],
  team_member_permissions: ["/key/generate", "/key/update"],
};

const UNCONFIGURED_FORM = {
  max_budget: "",
  budget_duration: "",
  tpm_limit: "",
  rpm_limit: "",
  models: [],
  team_member_permissions: [],
};

describe("settingsToForm", () => {
  it("maps a fully populated settings blob onto widget-space strings", () => {
    expect(settingsToForm(CONFIGURED_SETTINGS)).toStrictEqual(CONFIGURED_FORM);
  });

  it("maps an unconfigured settings blob onto empty widget state", () => {
    expect(settingsToForm(UNCONFIGURED_SETTINGS)).toStrictEqual(UNCONFIGURED_FORM);
  });

  it("maps an empty settings blob onto empty widget state", () => {
    expect(settingsToForm({})).toStrictEqual(UNCONFIGURED_FORM);
  });

  it("drops stored permissions the form does not offer instead of breaking the save", () => {
    const hydrated = settingsToForm({
      team_member_permissions: ["/key/generate", "/key/health", "/made/up/route"],
    });

    expect(hydrated.team_member_permissions).toStrictEqual(["/key/generate"]);
  });

  it("degrades unrecognisable value types to empty widget state instead of throwing", () => {
    expect(
      settingsToForm({ max_budget: "lots", tpm_limit: [], models: "gpt-5.2", team_member_permissions: 7 }),
    ).toStrictEqual(UNCONFIGURED_FORM);
  });
});

const formValues = (overrides: Partial<DefaultTeamSettingsFormValues> = {}): DefaultTeamSettingsFormValues => ({
  max_budget: "100",
  budget_duration: "30d",
  tpm_limit: "1000",
  rpm_limit: "50",
  models: ["gpt-5.2"],
  team_member_permissions: ["/key/generate"],
  ...overrides,
});

const SAVED_BODY = {
  max_budget: 100,
  budget_duration: "30d",
  tpm_limit: 1000,
  rpm_limit: 50,
  models: ["gpt-5.2"],
  team_member_permissions: ["/key/generate"],
};

describe("buildBody", () => {
  it("sends every field, because the endpoint replaces the whole settings object", () => {
    expect(buildBody(formValues())).toStrictEqual(SAVED_BODY);
  });

  it("clears emptied fields with null but keeps models a list, because that backend field is non-nullable", () => {
    expect(
      buildBody(
        formValues({
          max_budget: "",
          budget_duration: "",
          tpm_limit: "",
          rpm_limit: "",
          models: [],
          team_member_permissions: [],
        }),
      ),
    ).toStrictEqual({
      max_budget: null,
      budget_duration: null,
      tpm_limit: null,
      rpm_limit: null,
      models: [],
      team_member_permissions: null,
    });
  });
});
