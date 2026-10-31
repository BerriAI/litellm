import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../../../../tests/test-utils";
import EditSSOSettingsModal from "./EditSSOSettingsModal";
import { useSSOSettings } from "@/app/(dashboard)/hooks/sso/useSSOSettings";
import { useEditSSOSettings } from "@/app/(dashboard)/hooks/sso/useEditSSOSettings";

vi.mock("@/app/(dashboard)/hooks/sso/useSSOSettings", () => ({ useSSOSettings: vi.fn() }));
vi.mock("@/app/(dashboard)/hooks/sso/useEditSSOSettings", () => ({ useEditSSOSettings: vi.fn() }));

const mutateAsync = vi.fn().mockResolvedValue({});

const storedGoogleConfig = {
  google_client_id: "stored-client-id",
  google_client_secret: "stored-client-secret",
  user_email: "admin@example.com",
  proxy_base_url: "https://gateway.example.com",
};

const seed = (values: Record<string, unknown> = storedGoogleConfig) => {
  vi.mocked(useSSOSettings).mockReturnValue({
    data: { values },
    isLoading: false,
    error: null,
  } as unknown as ReturnType<typeof useSSOSettings>);
  vi.mocked(useEditSSOSettings).mockReturnValue({
    mutateAsync,
    isPending: false,
  } as unknown as ReturnType<typeof useEditSSOSettings>);
};

const saveButton = () => screen.getByRole("button", { name: "Save" });

describe("EditSSOSettingsModal (real form tree)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    seed();
  });

  it("blocks the save when a required credential is cleared", async () => {
    const user = userEvent.setup({ pointerEventsCheck: 0 });
    renderWithProviders(<EditSSOSettingsModal isVisible={true} onCancel={vi.fn()} onSuccess={vi.fn()} />);

    await user.clear(await screen.findByLabelText("Google Client ID"));
    await user.click(saveButton());

    expect(await screen.findByText("Please enter the google client id")).toBeInTheDocument();
    expect(mutateAsync).not.toHaveBeenCalled();
  });

  it("sends the mounted fields and clears every other provider's fields and the unchecked mappings", async () => {
    const user = userEvent.setup({ pointerEventsCheck: 0 });
    renderWithProviders(<EditSSOSettingsModal isVisible={true} onCancel={vi.fn()} onSuccess={vi.fn()} />);

    await user.clear(await screen.findByLabelText("Google Client ID"));
    fireEvent.change(screen.getByLabelText("Google Client ID"), { target: { value: "rotated-client-id" } });
    await user.click(saveButton());

    await waitFor(() => expect(mutateAsync).toHaveBeenCalledTimes(1));
    expect(mutateAsync.mock.calls[0][0]).toEqual({
      sso_provider: "google",
      google_client_id: "rotated-client-id",
      google_client_secret: "stored-client-secret",
      user_email: "admin@example.com",
      proxy_base_url: "https://gateway.example.com",
      microsoft_client_id: null,
      microsoft_client_secret: null,
      microsoft_tenant: null,
      generic_client_id: null,
      generic_client_secret: null,
      generic_authorization_endpoint: null,
      generic_token_endpoint: null,
      generic_userinfo_endpoint: null,
      generic_scope: null,
      saml_idp_metadata_url: null,
      saml_idp_metadata_xml: null,
      saml_sp_entity_id: null,
      saml_allow_unsolicited: null,
      role_mappings: null,
      team_mappings: null,
    });
  });
});
