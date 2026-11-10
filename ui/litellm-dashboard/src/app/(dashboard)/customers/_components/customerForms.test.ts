import { describe, expect, test } from "vitest";
import { buildCreateCustomerBody, buildUpdateCustomerBody } from "./customerForms";

describe("buildCreateCustomerBody", () => {
  const fields = { userId: " acme ", alias: " Acme Corp ", budget: "25", tpm: "1000", rpm: "60" };

  test("trims ids and aliases and sends numeric limits with a monthly budget reset", () => {
    expect(buildCreateCustomerBody(fields)).toEqual({
      user_id: "acme",
      alias: "Acme Corp",
      blocked: false,
      max_budget: 25,
      budget_duration: "1mo",
      tpm_limit: 1000,
      rpm_limit: 60,
    });
  });

  test("empty optional fields become null and no budget means no budget_duration", () => {
    expect(buildCreateCustomerBody({ userId: "acme", alias: "", budget: "", tpm: "", rpm: "" })).toEqual({
      user_id: "acme",
      alias: null,
      blocked: false,
      max_budget: null,
      budget_duration: null,
      tpm_limit: null,
      rpm_limit: null,
    });
  });

  test("a blank customer id produces no request", () => {
    expect(buildCreateCustomerBody({ ...fields, userId: "   " })).toBeNull();
  });
});

describe("buildUpdateCustomerBody", () => {
  test("preserves the blocked flag and clears the budget when emptied", () => {
    expect(buildUpdateCustomerBody({ userId: "acme", alias: "", budget: "", blocked: true })).toEqual({
      user_id: "acme",
      alias: null,
      blocked: true,
      max_budget: null,
    });
  });
  test("parses a decimal budget", () => {
    expect(buildUpdateCustomerBody({ userId: "acme", alias: "Acme", budget: "12.50", blocked: false }).max_budget).toBe(
      12.5,
    );
  });
});
