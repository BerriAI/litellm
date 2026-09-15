import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "../../../../../tests/test-utils";
import EditCyberArkModal from "./EditCyberArkModal";
import { useCyberArkConfig } from "@/app/(dashboard)/hooks/configOverrides/useCyberArkConfig";
import { useUpdateCyberArkConfig } from "@/app/(dashboard)/hooks/configOverrides/useUpdateCyberArkConfig";

vi.mock("@/app/(dashboard)/hooks/configOverrides/useCyberArkConfig", () => ({
  useCyberArkConfig: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/configOverrides/useUpdateCyberArkConfig", () => ({
  useUpdateCyberArkConfig: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "sk-access-token" }),
}));

vi.mock("@/lib/toast", () => ({
  toast: { success: vi.fn(), fromError: vi.fn() },
}));

const ALL_FIELDS = [
  "cyberark_api_base",
  "cyberark_account",
  "cyberark_username",
  "cyberark_api_key",
  "client_cert",
  "client_key",
  "ssl_verify",
  "refresh_interval",
] as const;

const propertiesFor = (fields: readonly string[]) =>
  Object.fromEntries(fields.map((name) => [name, { description: `${name} description` }]));

const mutate = vi.fn();

const setup = (options?: { values?: Record<string, unknown>; fields?: readonly string[] }) => {
  vi.mocked(useCyberArkConfig).mockReturnValue({
    data: {
      field_schema: { properties: propertiesFor(options?.fields ?? ALL_FIELDS) },
      values: options?.values ?? {},
    },
  } as unknown as ReturnType<typeof useCyberArkConfig>);

  vi.mocked(useUpdateCyberArkConfig).mockReturnValue({
    mutate,
    isPending: false,
  } as unknown as ReturnType<typeof useUpdateCyberArkConfig>);
};

const renderModal = (onSuccess = vi.fn(), onCancel = vi.fn()) =>
  renderWithProviders(<EditCyberArkModal isVisible={true} onCancel={onCancel} onSuccess={onSuccess} />);

const save = async (user: ReturnType<typeof userEvent.setup>) =>
  user.click(screen.getByRole("button", { name: "Save" }));

describe("EditCyberArkModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("clears untouched non-sensitive fields and omits untouched sensitive fields", async () => {
    setup({
      values: {
        cyberark_api_base: "https://conjur.example.com",
        cyberark_account: "myorg",
        cyberark_api_key: "super-secret-key",
        client_key: "super-secret-pem",
      },
    });
    const user = userEvent.setup();
    renderModal();

    await save(user);

    await waitFor(() => {
      expect(mutate).toHaveBeenCalledTimes(1);
    });
    const expectedPayload = {
      cyberark_api_base: "https://conjur.example.com",
      cyberark_account: "myorg",
      cyberark_username: "",
      client_cert: "",
      ssl_verify: "",
      refresh_interval: "",
    };
    expect(mutate.mock.calls[0][0]).toEqual(expectedPayload);
  });

  it("sends a sensitive field only once it is typed into", async () => {
    setup({ values: { cyberark_api_base: "https://conjur.example.com", cyberark_api_key: "super-secret-key" } });
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("API Key"), { target: { value: "rotated-key" } });
    await save(user);

    await waitFor(() => {
      expect(mutate).toHaveBeenCalledTimes(1);
    });
    expect(mutate.mock.calls[0][0]).toMatchObject({ cyberark_api_key: "rotated-key" });
  });

  it("never seeds a stored secret into its input", () => {
    setup({ values: { cyberark_api_key: "super-secret-key", client_key: "super-secret-pem" } });
    renderModal();

    expect(screen.getByLabelText("API Key")).toHaveValue("");
    expect(screen.getByLabelText("Client Key")).toHaveValue("");
  });

  it("renders only the fields the schema declares, and sends only those", async () => {
    setup({
      fields: ["cyberark_api_base", "cyberark_api_key"],
      values: { cyberark_api_base: "https://conjur.example.com" },
    });
    const user = userEvent.setup();
    renderModal();

    expect(screen.queryByLabelText("Account")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Client Key")).not.toBeInTheDocument();

    await save(user);

    await waitFor(() => {
      expect(mutate).toHaveBeenCalledTimes(1);
    });
    expect(mutate.mock.calls[0][0]).toEqual({ cyberark_api_base: "https://conjur.example.com" });
  });

  it("blocks the submit when the server url does not start with http", async () => {
    setup({ values: {} });
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Conjur Server URL"), { target: { value: "conjur.example.com" } });
    await save(user);

    expect(await screen.findByText("Must start with http:// or https://")).toBeInTheDocument();
    expect(mutate).not.toHaveBeenCalled();
  });

  it("tells the admin a stored secret is kept when the field is left blank", () => {
    setup({ values: { cyberark_api_key: "super-secret-key" } });
    renderModal();

    expect(screen.getByLabelText("API Key")).toHaveAttribute(
      "placeholder",
      "Leave blank to keep existing (super-secret-key)",
    );
  });

  it("falls back to the schema description when no secret is stored yet", () => {
    setup({ values: {} });
    renderModal();

    expect(screen.getByLabelText("API Key")).toHaveAttribute("placeholder", "cyberark_api_key description");
  });

  it("closes without saving when cancelled", async () => {
    setup({ values: {} });
    const onCancel = vi.fn();
    const user = userEvent.setup();
    renderModal(vi.fn(), onCancel);

    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(mutate).not.toHaveBeenCalled();
  });
});
