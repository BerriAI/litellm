import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import PassThroughGuardrailsSection from "./PassThroughGuardrailsSection";

vi.mock("../networking", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../networking")>()),
  getGuardrailsList: vi.fn(async () => ({ guardrails: [{ guardrail_name: "pii-guard" }] })),
}));

const FIELDS = [
  { label: "Request Fields", matcher: /Request Fields/, payloadKey: "request_fields", typed: "query" },
  { label: "Response Fields", matcher: /Response Fields/, payloadKey: "response_fields", typed: "choices.content" },
] as const;

const renderSection = (disabled: boolean) => {
  const onChange = vi.fn();
  render(
    <PassThroughGuardrailsSection
      accessToken="test-token"
      value={{ "pii-guard": null }}
      onChange={onChange}
      disabled={disabled}
    />,
  );
  return onChange;
};

const fieldInput = (matcher: RegExp) => screen.getByLabelText(matcher) as HTMLInputElement;

describe("PassThroughGuardrailsSection field targeting", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it.each(FIELDS)(
    "commits a typed $label entry while the section is enabled",
    async ({ matcher, payloadKey, typed }) => {
      const user = userEvent.setup();
      const onChange = renderSection(false);

      await user.type(fieldInput(matcher), `${typed},`);

      expect(onChange.mock.calls.at(-1)?.[0]).toStrictEqual({ "pii-guard": { [payloadKey]: [typed] } });
    },
  );

  it.each(FIELDS)("refuses typed $label input while the section is disabled", async ({ matcher, typed }) => {
    const user = userEvent.setup();
    const onChange = renderSection(true);
    const input = fieldInput(matcher);

    await user.type(input, `${typed},`);
    await user.type(input, "sneaked-in{Enter}");
    await user.tab();

    expect(input.value).toBe("");
    expect(onChange).not.toHaveBeenCalled();
  });
});
