import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/../tests/test-utils";
import { listGuardrailSubmissions } from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

import { TeamGuardrailsTab } from "./TeamGuardrailsTab";

const mutateAsync = vi.fn();

vi.mock("@/components/networking", () => ({
  listGuardrailSubmissions: vi.fn(),
  approveGuardrailSubmission: vi.fn(),
  rejectGuardrailSubmission: vi.fn(),
  updateGuardrailCall: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/guardrails/useRegisterGuardrail", () => ({
  useRegisterGuardrail: () => ({ mutateAsync, isPending: false }),
}));

vi.mock("@/components/common_components/team_dropdown", () => ({
  default: ({ value, onChange }: { value?: string; onChange?: (value: string) => void }) => (
    <input aria-label="team" value={value ?? ""} onChange={(event) => onChange?.(event.target.value)} />
  ),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: vi.fn() }));

const authorized = {
  isLoading: false,
  isAuthorized: true,
  token: "sk-test",
  accessToken: "sk-test",
  userId: "user-1",
  userEmail: "user@example.com",
  userRole: "Admin",
  userRoleLabel: "Admin",
  isViewOnly: false,
  premiumUser: false,
  disabledPersonalKeyCreation: null,
  showSSOBanner: false,
};

const openSubmitModal = async (user: ReturnType<typeof userEvent.setup>) => {
  renderWithProviders(<TeamGuardrailsTab accessToken="sk-test" />);
  await user.click(await screen.findByRole("button", { name: /Add Guardrail/ }));
  await screen.findByText("Submit Guardrail for Review");
};

const fillRequiredFields = async (user: ReturnType<typeof userEvent.setup>, apiBase: string) => {
  await user.type(screen.getByLabelText("team"), "team-1");
  await user.type(screen.getByPlaceholderText("e.g. pii-detection"), "pii-detection");
  await user.type(screen.getByPlaceholderText("https://your-guardrail-api.com/v1/check"), apiBase);
};

const submit = (user: ReturnType<typeof userEvent.setup>) =>
  user.click(screen.getByRole("button", { name: "Submit for Review" }));

const registeredPayload = () => mutateAsync.mock.calls[0][0];

const VALIDATION_MESSAGES = [
  "Select a team",
  "Enter a guardrail name",
  "Enter the API base URL",
  "Must be a valid URL",
  "Must be a JSON object",
  "Invalid JSON",
];

const visibleErrors = () => VALIDATION_MESSAGES.filter((message) => screen.queryByText(message) !== null);

describe("TeamGuardrailsTab submit payload", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useAuthorized).mockReturnValue(authorized);
    vi.mocked(listGuardrailSubmissions).mockResolvedValue({
      submissions: [],
      summary: { total: 0, pending_review: 0, active: 0, rejected: 0 },
    });
  });

  it("sends the guardrail defaults and leaves guardrail_info undefined when the optional fields are blank", async () => {
    const user = userEvent.setup();
    await openSubmitModal(user);

    await fillRequiredFields(user, "https://guard.example.com/v1/check");
    await submit(user);

    await vi.waitFor(() => expect(mutateAsync).toHaveBeenCalledTimes(1));
    expect(registeredPayload()).toStrictEqual({
      team_id: "team-1",
      guardrail_name: "pii-detection",
      litellm_params: {
        guardrail: "generic_guardrail_api",
        mode: "pre_call",
        api_base: "https://guard.example.com/v1/check",
      },
      guardrail_info: undefined,
    });
  });

  it("merges the extra params underneath the form fields so the form's api_base and mode win", async () => {
    const user = userEvent.setup();
    await openSubmitModal(user);

    await fillRequiredFields(user, "https://guard.example.com/v1/check");
    await user.type(
      screen.getByPlaceholderText('{"forward_api_key": true, "headers": {"X-Custom": "value"}}'),
      '{{"forward_api_key": true, "api_base": "https://ignored.example.com", "mode": "post_call"}',
    );
    await user.type(
      screen.getByPlaceholderText('{"description": "Detects PII in requests"}'),
      '{{"description": "Detects PII"}',
    );
    await submit(user);

    await vi.waitFor(() => expect(mutateAsync).toHaveBeenCalledTimes(1));
    expect(registeredPayload()).toStrictEqual({
      team_id: "team-1",
      guardrail_name: "pii-detection",
      litellm_params: {
        forward_api_key: true,
        api_base: "https://guard.example.com/v1/check",
        mode: "pre_call",
        guardrail: "generic_guardrail_api",
      },
      guardrail_info: { description: "Detects PII" },
    });
  });

  it("sends the mode the user picked", async () => {
    const user = userEvent.setup();
    await openSubmitModal(user);

    await fillRequiredFields(user, "https://guard.example.com/v1/check");
    await user.click(screen.getByRole("combobox", { name: "Mode" }));
    const options = await screen.findAllByText("During Call");
    await user.click(options[options.length - 1]);
    await submit(user);

    await vi.waitFor(() => expect(mutateAsync).toHaveBeenCalledTimes(1));
    expect(registeredPayload().litellm_params.mode).toBe("during_call");
  });

  it("shows the mode by its human label on the trigger", async () => {
    const user = userEvent.setup();
    await openSubmitModal(user);

    const mode = screen.getByRole("combobox", { name: "Mode" });
    expect(mode).toHaveTextContent("Pre Call");

    await user.click(mode);
    await user.click(await screen.findByRole("option", { name: "During Call" }));

    expect(screen.getByRole("combobox", { name: "Mode" })).toHaveTextContent("During Call");
  });

  it("blocks an empty submit and reports every required field", async () => {
    const user = userEvent.setup();
    await openSubmitModal(user);

    await submit(user);

    expect(await screen.findByText("Enter a guardrail name")).toBeInTheDocument();
    expect(visibleErrors()).toStrictEqual(["Select a team", "Enter a guardrail name", "Enter the API base URL"]);
    expect(mutateAsync).not.toHaveBeenCalled();
  });

  it("accepts a protocol-less www host but rejects a bare domain", async () => {
    const user = userEvent.setup();
    await openSubmitModal(user);

    await user.type(screen.getByPlaceholderText("https://your-guardrail-api.com/v1/check"), "www.example.com");
    await submit(user);
    await screen.findByText("Enter a guardrail name");
    expect(visibleErrors()).toStrictEqual(["Select a team", "Enter a guardrail name"]);

    await user.clear(screen.getByPlaceholderText("https://your-guardrail-api.com/v1/check"));
    await user.type(screen.getByPlaceholderText("https://your-guardrail-api.com/v1/check"), "example.com");
    await submit(user);

    await vi.waitFor(() =>
      expect(visibleErrors()).toStrictEqual(["Select a team", "Enter a guardrail name", "Must be a valid URL"]),
    );
    expect(mutateAsync).not.toHaveBeenCalled();
  });

  it("rejects a non-object extra params value and unparsable guardrail info", async () => {
    const user = userEvent.setup();
    await openSubmitModal(user);

    await fillRequiredFields(user, "https://guard.example.com/v1/check");
    await user.type(
      screen.getByPlaceholderText('{"forward_api_key": true, "headers": {"X-Custom": "value"}}'),
      '"a plain string"',
    );
    await user.type(screen.getByPlaceholderText('{"description": "Detects PII in requests"}'), "nope");
    await submit(user);

    await vi.waitFor(() => expect(visibleErrors()).toStrictEqual(["Must be a JSON object", "Invalid JSON"]));
    expect(mutateAsync).not.toHaveBeenCalled();
  });
});
