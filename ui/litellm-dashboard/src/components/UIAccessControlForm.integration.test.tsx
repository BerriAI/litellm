import { render, screen, waitFor } from "@testing-library/react";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { getSSOSettings, updateSSOSettings } from "@/components/networking";

import UIAccessControlForm from "./UIAccessControlForm";

vi.mock("@/components/networking", () => ({
  getSSOSettings: vi.fn(),
  updateSSOSettings: vi.fn(),
}));

const mockGetSSOSettings = vi.mocked(getSSOSettings);
const mockUpdateSSOSettings = vi.mocked(updateSSOSettings);

const RESTRICTED_GROUP_PLACEHOLDER = "ui-access-group";
const JWT_FIELD_PLACEHOLDER = "groups";
const SUBMIT_LABEL = "Update UI Access Control";

// jsdom has no layout, so a Base UI popup can never leave its unpositioned `pointer-events: none` state.
const renderForm = (accessToken: string | null = "sk-test") => {
  const onSuccess = vi.fn();
  render(<UIAccessControlForm accessToken={accessToken} onSuccess={onSuccess} />);
  return { onSuccess, user: userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never }) };
};

const chooseAccessMode = async (user: ReturnType<typeof userEvent.setup>, optionLabel: string) => {
  await user.click(screen.getByRole("combobox"));
  const options = await screen.findAllByText(optionLabel);
  await user.click(options[options.length - 1]);
};

const typeInto = async (user: ReturnType<typeof userEvent.setup>, placeholder: string, value: string) => {
  await user.type(screen.getByPlaceholderText(placeholder), value);
};

const submit = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(screen.getByRole("button", { name: SUBMIT_LABEL }));
};

const submittedPayload = () => mockUpdateSSOSettings.mock.calls[0];

describe("UIAccessControlForm", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetSSOSettings.mockResolvedValue({ values: {} });
    mockUpdateSSOSettings.mockResolvedValue({});
  });

  it("sends the nested ui_access_mode payload with every field the user filled in", async () => {
    const { onSuccess, user } = renderForm();

    await chooseAccessMode(user, "Restricted SSO Group");
    await typeInto(user, RESTRICTED_GROUP_PLACEHOLDER, "admin-team");
    await typeInto(user, JWT_FIELD_PLACEHOLDER, "team_groups");
    await submit(user);

    await waitFor(() => expect(mockUpdateSSOSettings).toHaveBeenCalledTimes(1));
    expect(submittedPayload()).toStrictEqual([
      "sk-test",
      {
        ui_access_mode: {
          type: "restricted_sso_group",
          restricted_sso_group: "admin-team",
          sso_group_jwt_field: "team_groups",
        },
      },
    ]);
    await waitFor(() => expect(onSuccess).toHaveBeenCalledTimes(1));
  });

  it('collapses the payload to ui_access_mode "none" for all authenticated users', async () => {
    const { onSuccess, user } = renderForm();

    await chooseAccessMode(user, "All Authenticated Users");
    await typeInto(user, JWT_FIELD_PLACEHOLDER, "team_groups");
    await submit(user);

    await waitFor(() => expect(mockUpdateSSOSettings).toHaveBeenCalledTimes(1));
    expect(submittedPayload()).toStrictEqual(["sk-test", { ui_access_mode: "none" }]);
    expect(screen.queryByPlaceholderText(RESTRICTED_GROUP_PLACEHOLDER)).not.toBeInTheDocument();
    await waitFor(() => expect(onSuccess).toHaveBeenCalledTimes(1));
  });

  it("sends undefined for every field the user never touched", async () => {
    const { user } = renderForm();

    await submit(user);

    await waitFor(() => expect(mockUpdateSSOSettings).toHaveBeenCalledTimes(1));
    expect(submittedPayload()).toStrictEqual([
      "sk-test",
      {
        ui_access_mode: {
          type: undefined,
          restricted_sso_group: undefined,
          sso_group_jwt_field: undefined,
        },
      },
    ]);
  });

  it("blocks submission while the restricted SSO group is empty", async () => {
    const { onSuccess, user } = renderForm();

    await chooseAccessMode(user, "Restricted SSO Group");
    await submit(user);

    expect(await screen.findByText("Please enter the restricted SSO group")).toBeInTheDocument();
    expect(mockUpdateSSOSettings).not.toHaveBeenCalled();
    expect(onSuccess).not.toHaveBeenCalled();
  });

  it("seeds the fields from a nested ui_access_mode object and resubmits them unchanged", async () => {
    mockGetSSOSettings.mockResolvedValue({
      values: {
        ui_access_mode: {
          type: "restricted_sso_group",
          restricted_sso_group: "loaded-group",
          sso_group_jwt_field: "loaded_field",
        },
      },
    });
    const { user } = renderForm();

    expect(await screen.findByDisplayValue("loaded-group")).toBeInTheDocument();
    expect(screen.getByDisplayValue("loaded_field")).toBeInTheDocument();

    await submit(user);

    await waitFor(() => expect(mockUpdateSSOSettings).toHaveBeenCalledTimes(1));
    expect(submittedPayload()).toStrictEqual([
      "sk-test",
      {
        ui_access_mode: {
          type: "restricted_sso_group",
          restricted_sso_group: "loaded-group",
          sso_group_jwt_field: "loaded_field",
        },
      },
    ]);
    expect(mockGetSSOSettings).toHaveBeenCalledWith("sk-test");
  });

  it("seeds the fields from the legacy flat structure, preferring team_ids_jwt_field", async () => {
    mockGetSSOSettings.mockResolvedValue({
      values: {
        ui_access_mode: "restricted_sso_group",
        restricted_sso_group: "legacy-group",
        team_ids_jwt_field: "legacy_team_ids",
        sso_group_jwt_field: "legacy_groups",
      },
    });
    const { user } = renderForm();

    expect(await screen.findByDisplayValue("legacy-group")).toBeInTheDocument();
    expect(screen.getByDisplayValue("legacy_team_ids")).toBeInTheDocument();

    await submit(user);

    await waitFor(() => expect(mockUpdateSSOSettings).toHaveBeenCalledTimes(1));
    expect(submittedPayload()).toStrictEqual([
      "sk-test",
      {
        ui_access_mode: {
          type: "restricted_sso_group",
          restricted_sso_group: "legacy-group",
          sso_group_jwt_field: "legacy_team_ids",
        },
      },
    ]);
  });

  it("keeps a seeded restricted SSO group out of the payload while its field is hidden", async () => {
    mockGetSSOSettings.mockResolvedValue({
      values: {
        ui_access_mode: "admin_only",
        restricted_sso_group: "seeded-but-hidden",
        sso_group_jwt_field: "seeded_field",
      },
    });
    const { user } = renderForm();

    expect(await screen.findByDisplayValue("seeded_field")).toBeInTheDocument();
    expect(screen.queryByPlaceholderText(RESTRICTED_GROUP_PLACEHOLDER)).not.toBeInTheDocument();

    await submit(user);

    await waitFor(() => expect(mockUpdateSSOSettings).toHaveBeenCalledTimes(1));
    expect(submittedPayload()).toStrictEqual([
      "sk-test",
      {
        ui_access_mode: {
          type: "admin_only",
          restricted_sso_group: undefined,
          sso_group_jwt_field: "seeded_field",
        },
      },
    ]);
  });

  it("restores a typed restricted SSO group when its field is shown again", async () => {
    const { user } = renderForm();

    await chooseAccessMode(user, "Restricted SSO Group");
    await typeInto(user, RESTRICTED_GROUP_PLACEHOLDER, "typed-then-hidden");
    await chooseAccessMode(user, "All Authenticated Users");
    await submit(user);

    await waitFor(() => expect(mockUpdateSSOSettings).toHaveBeenCalledTimes(1));
    expect(submittedPayload()).toStrictEqual(["sk-test", { ui_access_mode: "none" }]);

    await chooseAccessMode(user, "Restricted SSO Group");
    expect(screen.getByPlaceholderText(RESTRICTED_GROUP_PLACEHOLDER)).toHaveValue("typed-then-hidden");

    await submit(user);

    await waitFor(() => expect(mockUpdateSSOSettings).toHaveBeenCalledTimes(2));
    expect(mockUpdateSSOSettings.mock.calls[1]).toStrictEqual([
      "sk-test",
      {
        ui_access_mode: {
          type: "restricted_sso_group",
          restricted_sso_group: "typed-then-hidden",
          sso_group_jwt_field: undefined,
        },
      },
    ]);
  });

  it("never calls the API without an access token", async () => {
    const { onSuccess, user } = renderForm(null);

    await submit(user);

    await waitFor(() => expect(mockUpdateSSOSettings).not.toHaveBeenCalled());
    expect(mockGetSSOSettings).not.toHaveBeenCalled();
    expect(onSuccess).not.toHaveBeenCalled();
  });
});
