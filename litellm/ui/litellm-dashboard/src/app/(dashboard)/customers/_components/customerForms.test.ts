import { describe, expect, test } from "vitest";
import { buildCreateCustomerBody, buildUpdateCustomerBody } from "./customerForms";

describe("buildCreateCustomerBody", () => {
  const fields = { userId: " acme ", alias: " Acme Corp ", budget: "25", tpm: "1000", rpm: "60" };
  const emptyFields = { userId: "acme", alias: "", budget: "", tpm: "", rpm: "" };

  test("trims ids and aliases and sends numeric limits with a monthly budget reset", () => {
    const expected = {
      user_id: "acme",
      alias: "Acme Corp",
      blocked: false,
      max_budget: 25,
      budget_duration: "1mo",
      tpm_limit: 1000,
      rpm_limit: 60,
    };
    expect(buildCreateCustomerBody(fields)).toEqual(expected);
  });

  test("empty optional fields become null and no budget means no budget_duration", () => {
    const expected = {
      user_id: "acme",
      alias: null,
      blocked: false,
      max_budget: null,
      budget_duration: null,
      tpm_limit: null,
      rpm_limit: null,
    };
    expect(buildCreateCustomerBody(emptyFields)).toEqual(expected);
  });

  test("a blank customer id produces no request", () => {
    expect(buildCreateCustomerBody({ ...fields, userId: "   " })).toBeNull();
  });
});

describe("buildUpdateCustomerBody", () => {
  test("preserves the blocked flag and clears the budget when emptied", () => {
    const fields = { userId: "acme", alias: "", budget: "", blocked: true };
    const expected = { user_id: "acme", alias: null, blocked: true, max_budget: null };
    expect(buildUpdateCustomerBody(fields)).toEqual(expected);
  });
  test("parses a decimal budget", () => {
    const fields = { userId: "acme", alias: "Acme", budget: "12.50", blocked: false };
    expect(buildUpdateCustomerBody(fields).max_budget).toBe(12.5);
  });
});
