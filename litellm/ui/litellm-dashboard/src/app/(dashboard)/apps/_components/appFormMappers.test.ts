import { describe, expect, it } from "vitest";
import { appDefinitionTypeBody, appDefinitionWithoutAppBody } from "./appDefinitionMapper";
import type { App, AppMember, AppOrganization } from "./appForms";
import { appResourceBody, appSettingsBody, appSettingsDraft, createAppSettingsBody } from "./appSettingsMapper";
import { granteeAppsWrite } from "./internalAccessMapper";
import { memberAppBody, memberAppDraft } from "./memberAppMapper";
import { organizationAppBody, organizationAppDraft } from "./organizationAppMapper";

const resource = { resource_id: "team-chat", resource_type: "team" } as const;

describe("app form request payloads", () => {
  it.each([
    ["team", { team_id: "grantee" }],
    ["user", { user_id: "grantee" }],
    ["organization", { organization_id: "grantee" }],
  ] as const)("sends an empty app list to clear grants for a %s", (kind, grantee) => {
    expect(granteeAppsWrite(kind, "grantee", []).body).toStrictEqual({
      ...grantee,
      object_permission: { apps: [] },
    });
  });

  it("keeps the stored app definition when changing only customer creation", () => {
    expect(appDefinitionWithoutAppBody(false, undefined)).toStrictEqual({
      resource_type: null,
      auto_create_customers_without_app: false,
    });
    expect(
      appDefinitionTypeBody("project", { resource_type: "team", auto_create_customers_without_app: false }),
    ).toStrictEqual({ resource_type: "project", auto_create_customers_without_app: false });
  });

  it("registers an app with its resource and the entered settings", () => {
    const draft = { ...appSettingsDraft(undefined, []), surcharge: "12.5", autoCreate: false };
    const parsed = appSettingsBody(draft);
    if (!parsed.ok) throw new Error(parsed.error);

    const expected = {
      budget_id: null,
      auto_create_customers: false,
      customer_margin: 0.125,
      resource_id: "team-chat",
      resource_type: "team",
    };
    expect(createAppSettingsBody(parsed.value, resource)).toStrictEqual(expected);
  });

  it("omits unchanged app group membership but sends null to clear it", () => {
    const app: App = {
      ...resource,
      app_id: "app-chat",
      alias: "Chat",
      app_group_id: "group-chat",
      auto_create_customers: true,
      created_at: "2026-01-01T00:00:00Z",
      created_by: "admin",
      spend: 0,
      updated_at: "2026-01-01T00:00:00Z",
      updated_by: "admin",
    };
    const draft = appSettingsDraft(app, []);
    const retained = { budget_id: null, auto_create_customers: true, customer_margin: null };

    expect(appSettingsBody(draft, app)).toStrictEqual({ ok: true, value: retained });
    expect(appSettingsBody({ ...draft, appGroupId: null }, app)).toStrictEqual({
      ok: true,
      value: { ...retained, app_group_id: null },
    });
    expect(appSettingsBody({ ...draft, surcharge: "-1" }, app)).toStrictEqual({
      ok: false,
      error: "apps.surchargeInvalid",
    });
  });

  it("registers only the resource before assigning its organization budget", () => {
    expect(appResourceBody(resource)).toStrictEqual({ resource_id: "team-chat", resource_type: "team" });
    const draft = { ...organizationAppDraft(null, null), source: "profile:standard" };
    expect(organizationAppBody(draft, null, null)).toStrictEqual({
      ok: true,
      value: { budget_id: "standard" },
    });
  });

  it("clears a legacy organization override but omits a temporary override when restoring inheritance", () => {
    const row: AppOrganization = {
      app_id: "app-chat",
      customer_org_id: "org-chat",
      budget_override: { max_budget: 10 },
      spend: 0,
    };
    const draft = organizationAppDraft(null, null);
    expect(organizationAppBody(draft, null, row)).toStrictEqual({
      ok: true,
      value: { budget_id: null, budget_override: null },
    });
    expect(
      organizationAppBody(draft, null, { ...row, budget_override_expires_at: "2027-01-01T00:00:00Z" }),
    ).toStrictEqual({ ok: true, value: { budget_id: null } });
  });

  it("clears a member's permanent budget without changing their temporary override", () => {
    const row: AppMember = {
      app_id: "app-chat",
      customer_member_id: "member-chat",
      budget: { max_budget: 10 },
      budget_override: { max_budget: 20 },
      budget_override_expires_at: "2027-01-01T00:00:00Z",
      budget_source: "member_override",
      spend: 0,
    };
    const draft = { ...memberAppDraft(row, row.budget ?? null), budgetEnabled: false };
    expect(memberAppBody(draft, row.budget ?? null, row)).toStrictEqual({ ok: true, value: { budget: null } });
  });
});
