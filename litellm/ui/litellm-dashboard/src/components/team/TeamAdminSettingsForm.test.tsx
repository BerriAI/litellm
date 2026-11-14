import { describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";

import { chooseSelectOption, fireEvent, renderWithProviders, screen, waitFor } from "@/../tests/test-utils";

import TeamAdminSettingsForm, { type TeamAdminMetadataSettings } from "./TeamAdminSettingsForm";

const renderForm = (
  editableFields: ReadonlySet<string>,
  overrides: { isSaving?: boolean; mayRaiseMaxBudget?: boolean; metadata?: TeamAdminMetadataSettings } = {},
) => {
  const onSave = vi.fn().mockResolvedValue(undefined);
  const onCancel = vi.fn();
  renderWithProviders(
    <TeamAdminSettingsForm
      initialValues={{
        tpm_limit: 1000,
        rpm_limit: 50,
        max_budget: 20,
        soft_budget: 15,
        team_member_budget: 10,
        team_member_budget_duration: "30d",
        team_member_tpm_limit: 100,
        team_member_rpm_limit: 5,
      }}
      metadata={overrides.metadata}
      editableFields={editableFields}
      mayRaiseMaxBudget={overrides.mayRaiseMaxBudget ?? false}
      isSaving={overrides.isSaving ?? false}
      onCancel={onCancel}
      onSave={onSave}
    />,
  );
  return { onSave, onCancel };
};

describe("TeamAdminSettingsForm", () => {
  it("shows the team's current values for every field the proxy lets team admins edit", () => {
    renderForm(new Set(["tpm_limit", "rpm_limit", "max_budget"]));

    expect(screen.getByLabelText("Tokens per minute Limit (TPM)")).toHaveValue(1000);
    expect(screen.getByLabelText("Requests per minute Limit (RPM)")).toHaveValue(50);
    expect(screen.getByLabelText("Max Budget (USD)")).toHaveValue(20);
  });

  it("hides the fields the proxy has not enabled for team admins", () => {
    renderForm(new Set(["rpm_limit"]));

    expect(screen.getByLabelText("Requests per minute Limit (RPM)")).toBeInTheDocument();
    expect(screen.queryByLabelText("Tokens per minute Limit (TPM)")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Max Budget (USD)")).not.toBeInTheDocument();
  });

  it("saves the new TPM limit and nothing else", async () => {
    const user = userEvent.setup();
    const { onSave } = renderForm(new Set(["tpm_limit"]));

    fireEvent.change(screen.getByLabelText("Tokens per minute Limit (TPM)"), { target: { value: "5000" } });
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(onSave).toHaveBeenCalledWith({ tpm_limit: 5000 }));
  });

  it("saves a lowered budget and a new RPM limit without resending the unchanged TPM limit", async () => {
    const user = userEvent.setup();
    const { onSave } = renderForm(new Set(["tpm_limit", "rpm_limit", "max_budget"]));

    fireEvent.change(screen.getByLabelText("Requests per minute Limit (RPM)"), { target: { value: "80" } });
    fireEvent.change(screen.getByLabelText("Max Budget (USD)"), { target: { value: "12.5" } });
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(onSave).toHaveBeenCalledWith({ rpm_limit: 80, max_budget: 12.5 }));
  });

  it("saves a cleared TPM limit as no limit", async () => {
    const user = userEvent.setup();
    const { onSave } = renderForm(new Set(["tpm_limit"]));

    fireEvent.change(screen.getByLabelText("Tokens per minute Limit (TPM)"), { target: { value: "" } });
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(onSave).toHaveBeenCalledWith({ tpm_limit: null }));
  });

  it("keeps Save disabled until the TPM limit differs from the team's", () => {
    renderForm(new Set(["tpm_limit"]));
    const tpmInput = screen.getByLabelText("Tokens per minute Limit (TPM)");
    const save = screen.getByRole("button", { name: /save changes/i });

    expect(save).toBeDisabled();
    fireEvent.change(tpmInput, { target: { value: "5000" } });
    expect(save).toBeEnabled();
    fireEvent.change(tpmInput, { target: { value: "1000" } });
    expect(save).toBeDisabled();
  });

  it("shows the team's member defaults when the proxy lets team admins edit them", () => {
    renderForm(
      new Set(["team_member_budget", "team_member_budget_duration", "team_member_tpm_limit", "team_member_rpm_limit"]),
    );

    expect(screen.getByLabelText("Member Default Budget (USD)")).toHaveValue(10);
    expect(screen.getByLabelText("Member Default Budget Duration")).toHaveTextContent("monthly");
    expect(screen.getByLabelText("Member Default TPM Limit")).toHaveValue(100);
    expect(screen.getByLabelText("Member Default RPM Limit")).toHaveValue(5);
  });

  it("saves a new member default budget and a weekly reset without resending the unchanged limits", async () => {
    const user = userEvent.setup();
    const { onSave } = renderForm(
      new Set(["team_member_budget", "team_member_budget_duration", "team_member_tpm_limit", "team_member_rpm_limit"]),
    );

    fireEvent.change(screen.getByLabelText("Member Default Budget (USD)"), { target: { value: "25" } });
    await chooseSelectOption(user, screen.getByLabelText("Member Default Budget Duration"), "weekly");
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() =>
      expect(onSave).toHaveBeenCalledWith({ team_member_budget: 25, team_member_budget_duration: "7d" }),
    );
  });

  it("saves a member budget that never resets as a null duration", async () => {
    const user = userEvent.setup();
    const { onSave } = renderForm(new Set(["team_member_budget_duration"]));

    await chooseSelectOption(user, screen.getByLabelText("Member Default Budget Duration"), "Never resets");
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(onSave).toHaveBeenCalledWith({ team_member_budget_duration: null }));
  });

  it("shows a stored member budget that never resets as Never resets", () => {
    renderWithProviders(
      <TeamAdminSettingsForm
        initialValues={{ team_member_budget: 10, team_member_budget_duration: null }}
        editableFields={new Set(["team_member_budget_duration"])}
        mayRaiseMaxBudget={false}
        isSaving={false}
        onCancel={vi.fn()}
        onSave={vi.fn()}
      />,
    );

    expect(screen.getByLabelText("Member Default Budget Duration")).toHaveTextContent("Never resets");
  });

  it("shows the soft budget when the proxy lets team admins edit it", () => {
    renderForm(new Set(["soft_budget"]));

    expect(screen.getByLabelText("Soft Budget (USD)")).toHaveValue(15);
  });

  it("saves a lowered soft budget and nothing else", async () => {
    const user = userEvent.setup();
    const { onSave } = renderForm(new Set(["soft_budget", "max_budget"]));

    fireEvent.change(screen.getByLabelText("Soft Budget (USD)"), { target: { value: "10.5" } });
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(onSave).toHaveBeenCalledWith({ soft_budget: 10.5 }));
  });

  it("closes without saving on cancel", async () => {
    const user = userEvent.setup();
    const { onSave, onCancel } = renderForm(new Set(["tpm_limit"]));

    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(onSave).not.toHaveBeenCalled();
  });

  it("locks both buttons while a save is in flight", () => {
    renderForm(new Set(["tpm_limit"]), { isSaving: true });
    fireEvent.change(screen.getByLabelText("Tokens per minute Limit (TPM)"), { target: { value: "5000" } });

    expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled();
    expect(screen.getByRole("button", { name: /save changes/i })).toBeDisabled();
  });

  it.each([
    [false, "You can keep or lower this budget. Ask a proxy admin to raise it."],
    [true, "You can raise this budget. Raises are capped by your organization's budget when the team belongs to one."],
  ])("when mayRaiseMaxBudget is %s, describes only the Max Budget input with: %s", (mayRaiseMaxBudget, hint) => {
    renderForm(new Set(["tpm_limit", "max_budget"]), { mayRaiseMaxBudget });

    expect(screen.getByLabelText("Max Budget (USD)")).toHaveAccessibleDescription(hint);
    expect(screen.getByLabelText("Tokens per minute Limit (TPM)")).not.toHaveAccessibleDescription();
  });

  it("shows no budget hint when the proxy has not enabled Max Budget for team admins", () => {
    renderForm(new Set(["tpm_limit"]), { mayRaiseMaxBudget: true });

    expect(screen.queryByText(/this budget/)).not.toBeInTheDocument();
  });

  describe("team metadata", () => {
    const metadata: TeamAdminMetadataSettings = {
      pairs: [
        { key: "cost_center", value: "eng" },
        { key: "region", value: "eu" },
      ],
      context: {
        storedMetadata: { cost_center: "eng", region: "eu", priority: "high" },
        proxyControlledKeys: new Set(["priority"]),
        declaredKeys: new Set(["cost_center"]),
      },
      schemaFields: [{ key: "cost_center", label: "Cost Center" }],
      schemaLoading: false,
    };

    it("shows the declared field without a remove control and the team's own keys with one", () => {
      renderForm(new Set(["metadata"]), { metadata });

      expect(screen.getByLabelText("Cost Center")).toHaveValue("eng");
      expect(screen.getByDisplayValue("region")).toBeInTheDocument();
      expect(screen.queryByDisplayValue("priority")).not.toBeInTheDocument();
      expect(screen.getAllByRole("button", { name: "Remove key-value pair" })).toHaveLength(1);
    });

    it("saves a new key with the proxy's own keys carried over", async () => {
      const user = userEvent.setup();
      const { onSave } = renderForm(new Set(["metadata"]), { metadata });

      await user.click(screen.getByRole("button", { name: /add key-value pair/i }));
      const keys = screen.getAllByPlaceholderText("Key");
      const values = screen.getAllByPlaceholderText("Value");
      await user.type(keys[keys.length - 1], "project");
      await user.type(values[values.length - 1], "atlas");
      await user.click(screen.getByRole("button", { name: /save changes/i }));

      await waitFor(() =>
        expect(onSave).toHaveBeenCalledWith({
          metadata: { cost_center: "eng", region: "eu", project: "atlas", priority: "high" },
        }),
      );
    });

    it("refuses to save a key the proxy keeps for proxy admins", async () => {
      const user = userEvent.setup();
      const { onSave } = renderForm(new Set(["metadata"]), { metadata });

      await user.click(screen.getByRole("button", { name: /add key-value pair/i }));
      const keys = screen.getAllByPlaceholderText("Key");
      await user.type(keys[keys.length - 1], "priority");
      await user.click(screen.getByRole("button", { name: /save changes/i }));

      expect(await screen.findByText("Only proxy admins can set this key")).toBeInTheDocument();
      expect(onSave).not.toHaveBeenCalled();
    });

    it("refuses to save an emptied declared field", async () => {
      const user = userEvent.setup();
      const { onSave } = renderForm(new Set(["metadata"]), { metadata });

      await user.clear(screen.getByLabelText("Cost Center"));
      await user.click(screen.getByRole("button", { name: /save changes/i }));

      expect(await screen.findByText("This field cannot be left empty")).toBeInTheDocument();
      expect(onSave).not.toHaveBeenCalled();
    });

    it("waits for the metadata schema before showing the form, then enforces the declared field", async () => {
      const user = userEvent.setup();
      const onSave = vi.fn().mockResolvedValue(undefined);
      const loading: TeamAdminMetadataSettings = {
        ...metadata,
        context: { ...metadata.context, declaredKeys: new Set() },
        schemaFields: [],
        schemaLoading: true,
      };
      const form = (settings: TeamAdminMetadataSettings) => (
        <TeamAdminSettingsForm
          initialValues={{}}
          metadata={settings}
          editableFields={new Set(["metadata"])}
          mayRaiseMaxBudget={false}
          isSaving={false}
          onCancel={vi.fn()}
          onSave={onSave}
        />
      );
      const { rerender } = renderWithProviders(form(loading));

      expect(screen.queryByRole("button", { name: /save changes/i })).not.toBeInTheDocument();

      rerender(form(metadata));
      await user.clear(await screen.findByLabelText("Cost Center"));
      await user.click(screen.getByRole("button", { name: /save changes/i }));

      expect(await screen.findByText("This field cannot be left empty")).toBeInTheDocument();
      expect(onSave).not.toHaveBeenCalled();
    });

    it("hides the metadata section when the proxy has not enabled it", () => {
      renderForm(new Set(["tpm_limit"]), { metadata });

      expect(screen.queryByLabelText("Cost Center")).not.toBeInTheDocument();
      expect(screen.getByLabelText("Tokens per minute Limit (TPM)")).toBeInTheDocument();
    });
  });
});
