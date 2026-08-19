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

  it("sends only the mounted fields when the form is valid", async () => {
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
    });
  });
});
