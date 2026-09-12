import { describe, expect, it } from "vitest";

import { buildBody, settingsToForm } from "./mapper";
import type { DefaultUserSettingsFormValues } from "./schema";

const CONFIGURED_SETTINGS = {
  user_role: "internal_user",
  max_budget: 100.5,
  budget_duration: "30d",
  models: ["gpt-5.2"],
  teams: [{ team_id: "team-1", max_budget_in_team: 25, user_role: "admin" }],
};

const UNCONFIGURED_SETTINGS = {
  user_role: "internal_user_viewer",
  max_budget: null,
  budget_duration: null,
  models: null,
  teams: null,
};

const CONFIGURED_FORM = {
  user_role: "internal_user",
  max_budget: "100.5",
  budget_duration: "30d",
  models: ["gpt-5.2"],
  teams: [{ team_id: "team-1", max_budget_in_team: "25", user_role: "admin" }],
};

const UNCONFIGURED_FORM = {
  user_role: "internal_user_viewer",
  max_budget: "",
  budget_duration: "",
  models: [],
  teams: [],
};

describe("settingsToForm", () => {
  it("maps a fully populated settings blob onto widget-space strings", () => {
    expect(settingsToForm(CONFIGURED_SETTINGS)).toStrictEqual(CONFIGURED_FORM);
  });

  it("maps an unconfigured settings blob onto empty widget state", () => {
    expect(settingsToForm(UNCONFIGURED_SETTINGS)).toStrictEqual(UNCONFIGURED_FORM);
  });

  it("hydrates the legacy list-of-team-ids shape as full team rows", () => {
    expect(settingsToForm({ teams: ["team-1", "team-2"] }).teams).toStrictEqual([
      { team_id: "team-1", max_budget_in_team: "", user_role: "user" },
      { team_id: "team-2", max_budget_in_team: "", user_role: "user" },
    ]);
  });

  it("defaults a team row's role to user and leaves an absent in-team budget blank", () => {
    expect(settingsToForm({ teams: [{ team_id: "team-1" }] }).teams).toStrictEqual([
      { team_id: "team-1", max_budget_in_team: "", user_role: "user" },
    ]);
  });

  it("degrades an unrecognisable team entry to a blank row instead of throwing", () => {
    expect(settingsToForm({ teams: [{ max_budget_in_team: 5 }, 7] }).teams).toStrictEqual([
      { team_id: "", max_budget_in_team: "", user_role: "user" },
      { team_id: "", max_budget_in_team: "", user_role: "user" },
    ]);
  });
});

const formValues = (overrides: Partial<DefaultUserSettingsFormValues> = {}): DefaultUserSettingsFormValues => ({
  user_role: "internal_user",
  max_budget: "100",
  budget_duration: "30d",
  models: ["gpt-5.2"],
  teams: [{ team_id: "team-1", max_budget_in_team: "25", user_role: "admin" }],
  ...overrides,
});

const SAVED_BODY = {
  user_role: "internal_user",
  max_budget: 100,
  budget_duration: "30d",
  models: ["gpt-5.2"],
  teams: [{ team_id: "team-1", max_budget_in_team: 25, user_role: "admin" }],
};

const CLEARED_FORM = { user_role: "", max_budget: "", budget_duration: "", models: [], teams: [] };

const CLEARED_BODY = {
  user_role: null,
  max_budget: null,
  budget_duration: null,
  models: null,
  teams: null,
};

describe("buildBody", () => {
  it("sends every field, because the endpoint replaces the whole settings object", () => {
    expect(buildBody(formValues())).toStrictEqual(SAVED_BODY);
  });

  it("clears emptied fields with null so the backend drops them", () => {
    expect(buildBody(formValues(CLEARED_FORM))).toStrictEqual(CLEARED_BODY);
  });

  it("sends teams as objects and nulls an in-team budget that was left blank", () => {
    expect(
      buildBody(formValues({ teams: [{ team_id: "team-9", max_budget_in_team: "", user_role: "user" }] })),
    ).toStrictEqual({
      ...SAVED_BODY,
      teams: [{ team_id: "team-9", max_budget_in_team: null, user_role: "user" }],
    });
  });

  it("refuses to send a role the backend does not accept", () => {
    expect(buildBody(formValues({ user_role: "made_up_role" })).user_role).toBeNull();
  });
});
