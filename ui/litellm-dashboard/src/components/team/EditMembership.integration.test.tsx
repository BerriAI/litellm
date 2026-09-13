import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import EditMembership from "./EditMembership";

const onSubmit = vi.fn();

const roleOptions = [
  { label: "Admin", value: "admin" },
  { label: "User", value: "user" },
];

const additionalFields = [
  { name: "max_budget_in_team", label: "Team Member Budget (USD)", type: "numerical" as const, step: 0.01, min: 0 },
  { name: "budget_duration", label: "Budget Reset Period", type: "budget-duration" as const },
  { name: "tpm_limit", label: "Team Member TPM Limit", type: "numerical" as const, step: 1, min: 0 },
  { name: "rpm_limit", label: "Team Member RPM Limit", type: "numerical" as const, step: 1, min: 0 },
  {
    name: "allowed_models",
    label: "Allowed Models",
    type: "multi-select" as const,
    options: [
      { label: "gpt-4o", value: "gpt-4o" },
      { label: "claude", value: "claude" },
    ],
  },
];

const teamMemberConfig = { title: "Edit Member", showEmail: true, showUserId: true, roleOptions, additionalFields };

const orgMemberConfig = { title: "Edit Member", showEmail: true, showUserId: true, roleOptions };

type Member = Record<string, unknown>;

const renderEdit = (config: object, initialData: Member) =>
  renderWithProviders(
    <EditMembership
      visible
      onCancel={vi.fn()}
      onSubmit={onSubmit}
      mode="edit"
      config={config as never}
      initialData={initialData as never}
    />,
  );

const save = () => fireEvent.click(screen.getByRole("button", { name: "Save Changes" }));

const submitted = (): Record<string, unknown> => onSubmit.mock.calls[0][0] as Record<string, unknown>;

describe("EditMembership submit payload", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("submits only the fields the config renders, never the rest of the member record", async () => {
    renderEdit(teamMemberConfig, {
      user_id: "u1",
      user_email: "a@b.com",
      role: "user",
      max_budget_in_team: 12.5,
      budget_duration: "24h",
      tpm_limit: 100,
      rpm_limit: 20,
      allowed_models: ["gpt-4o"],
      spend: 3.21,
      team_id: "t1",
      created_at: "2026-01-01T00:00:00Z",
    });

    save();

    await waitFor(() => expect(onSubmit).toHaveBeenCalledOnce());
    expect(submitted()).toStrictEqual({
      user_email: "a@b.com",
      user_id: "u1",
      role: "user",
      max_budget_in_team: 12.5,
      budget_duration: "24h",
      tpm_limit: 100,
      rpm_limit: 20,
      allowed_models: ["gpt-4o"],
    });
  });

  it("omits a field the config hides even when the member record carries it", async () => {
    renderEdit({ ...orgMemberConfig, showUserId: false }, { user_id: "u1", user_email: "a@b.com", role: "admin" });

    save();

    await waitFor(() => expect(onSubmit).toHaveBeenCalledOnce());
    expect(submitted()).toStrictEqual({ user_email: "a@b.com", role: "admin" });
  });

  it.each([
    [
      { user_id: "u1", user_email: null, role: "user" },
      { user_email: null, user_id: "u1", role: "user" },
    ],
    [
      { user_id: null, user_email: "a@b.com", role: "user" },
      { user_email: "a@b.com", user_id: null, role: "user" },
    ],
  ])("submits a member whose identity the API returned as null", async (initialData, expected) => {
    renderEdit(orgMemberConfig, initialData as Member);

    save();

    await waitFor(() => expect(onSubmit).toHaveBeenCalledOnce());
    expect(submitted()).toStrictEqual(expected);
  });

  it("keeps stored 0 budget and limits as 0 on an untouched save, collapsing only empty strings and a missing model list", async () => {
    renderEdit(teamMemberConfig, {
      user_id: "u1",
      user_email: "a@b.com",
      role: "user",
      max_budget_in_team: 0,
      tpm_limit: 0,
      rpm_limit: 0,
      budget_duration: "",
    });

    save();

    await waitFor(() => expect(onSubmit).toHaveBeenCalledOnce());
    expect(submitted()).toStrictEqual({
      user_email: "a@b.com",
      user_id: "u1",
      role: "user",
      max_budget_in_team: 0,
      budget_duration: null,
      tpm_limit: 0,
      rpm_limit: 0,
      allowed_models: [],
    });
  });

  it("submits a typed numeric field as the raw string and a cleared one as null", async () => {
    renderEdit(teamMemberConfig, { user_id: "u1", user_email: "a@b.com", role: "user", tpm_limit: 7 });

    fireEvent.change(screen.getByLabelText("Team Member Budget (USD)"), { target: { value: "42.56" } });
    fireEvent.change(screen.getByLabelText("Team Member TPM Limit"), { target: { value: "" } });

    save();

    await waitFor(() => expect(onSubmit).toHaveBeenCalledOnce());
    expect(submitted().max_budget_in_team).toBe("42.56");
    expect(submitted().tpm_limit).toBeNull();
  });

  it("trims surrounding whitespace off text fields", async () => {
    renderEdit(orgMemberConfig, { user_id: "u1", user_email: "a@b.com", role: "user" });

    fireEvent.change(screen.getByLabelText("User ID"), { target: { value: "  padded-id  " } });

    save();

    await waitFor(() => expect(onSubmit).toHaveBeenCalledOnce());
    expect(submitted().user_id).toBe("padded-id");
  });

  it("keeps a blanked text field as an empty string rather than null", async () => {
    renderEdit(orgMemberConfig, { user_id: "u1", user_email: "a@b.com", role: "user" });

    fireEvent.change(screen.getByLabelText("User ID"), { target: { value: "   " } });

    save();

    await waitFor(() => expect(onSubmit).toHaveBeenCalledOnce());
    expect(submitted().user_id).toBe("");
  });

  it("registers the hidden fields in add mode and sends only the default role", async () => {
    renderWithProviders(
      <EditMembership
        visible
        onCancel={vi.fn()}
        onSubmit={onSubmit}
        mode="add"
        config={{ ...orgMemberConfig, title: "Add Member", defaultRole: "user" } as never}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Add Member" }));

    await waitFor(() => expect(onSubmit).toHaveBeenCalledOnce());
    expect(submitted()).toStrictEqual({ user_email: undefined, user_id: undefined, role: "user" });
    expect(JSON.stringify(submitted())).toBe('{"role":"user"}');
  });

  it("falls back to the first role option when the config names no default", async () => {
    renderWithProviders(
      <EditMembership
        visible
        onCancel={vi.fn()}
        onSubmit={onSubmit}
        mode="add"
        config={{ ...orgMemberConfig, title: "Add Member" } as never}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Add Member" }));

    await waitFor(() => expect(onSubmit).toHaveBeenCalledOnce());
    expect(submitted().role).toBe("admin");
  });

  it("blocks submission when the email is not an address", async () => {
    renderEdit(orgMemberConfig, { user_id: "u1", user_email: "a@b.com", role: "user" });

    fireEvent.change(screen.getByLabelText("Email"), { target: { value: "not-an-email" } });
    save();

    expect(await screen.findByText("Please enter a valid email!")).toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("blocks submission when no role is selected", async () => {
    renderEdit(orgMemberConfig, { user_id: "u1", user_email: "a@b.com", role: "" });

    save();

    expect(await screen.findByText("Please select a role!")).toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it.each([
    ["Team Member Budget (USD)", "42.567", "stepMismatch"],
    ["Team Member TPM Limit", "12.7", "stepMismatch"],
    ["Team Member Budget (USD)", "-5", "rangeUnderflow"],
  ])("blocks submission when %s holds %s", async (label, value, violation) => {
    renderEdit(teamMemberConfig, { user_id: "u1", user_email: "a@b.com", role: "user" });

    const input = screen.getByLabelText(label) as HTMLInputElement;
    fireEvent.change(input, { target: { value } });

    expect(input.validity[violation as "stepMismatch" | "rangeUnderflow"]).toBe(true);

    save();

    await waitFor(() => expect(onSubmit).not.toHaveBeenCalled());
  });

  it("clears the fields once the submit handler resolves", async () => {
    renderEdit(orgMemberConfig, { user_id: "u1", user_email: "a@b.com", role: "user" });

    save();

    await waitFor(() => expect(onSubmit).toHaveBeenCalledOnce());
    await waitFor(() => expect(screen.getByLabelText("User ID")).toHaveValue(""));
    expect(screen.getByLabelText("Email")).toHaveValue("");
  });
});
