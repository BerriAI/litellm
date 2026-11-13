import { describe, expect, it } from "vitest";
import { ApiError } from "@/lib/http/client";
import {
  CUSTOMER_CONTEXT_PARSERS,
  customerPlaygroundHref,
  customerRequestFailure,
  remainingCustomerBudget,
} from "./customerContext";

describe("customer deep links", () => {
  it("encodes IDs without making them URL paths or executable URLs", () => {
    const url = new URL(customerPlaygroundHref("customer/org & one", "customer/member?x"), "https://dashboard.example");
    expect(url.pathname).toMatch(/\/playground$/);
    expect(url.searchParams.get("customerOrg")).toBe("customer/org & one");
    expect(url.searchParams.get("customerMember")).toBe("customer/member?x");
    expect(CUSTOMER_CONTEXT_PARSERS.customerOrg.parse("customer/org")).toBe("customer/org");
    expect(CUSTOMER_CONTEXT_PARSERS.customerMember.parse("customer/member")).toBe("customer/member");
  });
  it.each(["", " ", "bad\r\nheader", "bad\u0000id", "x".repeat(1025)])("rejects malformed context %j", (value) => {
    expect(CUSTOMER_CONTEXT_PARSERS.customerOrg.parse(value)).toBeNull();
  });
});

it("keeps a budget denial's status and useful message", () => {
  expect(customerRequestFailure(new ApiError("Customer member budget_exceeded", 422, {}))).toEqual({
    status: 422,
    message: "Customer member budget_exceeded",
  });
  expect(customerRequestFailure(null)).toBeNull();
  expect(
    customerRequestFailure(
      new ApiError("Request failed", 422, {
        error: { type: "budget_exceeded", message: "Customer member limit reached" },
      }),
    ),
  ).toEqual({ status: 422, message: "budget_exceeded: Customer member limit reached" });
  expect(
    customerRequestFailure(
      new ApiError("Request failed", 422, {
        error: {
          type: "invalid_request_error",
          message: "ExceededBudget: customer member is over budget",
          litellm_error_type: "budget_exceeded",
        },
      }),
    ),
  ).toEqual({ status: 422, message: "budget_exceeded: ExceededBudget: customer member is over budget" });
  expect(remainingCustomerBudget(3, 10)).toBe(7);
  expect(remainingCustomerBudget(12, 10)).toBe(0);
  expect(remainingCustomerBudget(12, null)).toBeNull();
});
