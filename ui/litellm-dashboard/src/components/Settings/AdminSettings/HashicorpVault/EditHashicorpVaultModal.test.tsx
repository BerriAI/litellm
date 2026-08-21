import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "../../../../../tests/test-utils";
import EditHashicorpVaultModal from "./EditHashicorpVaultModal";
import { useHashicorpVaultConfig } from "@/app/(dashboard)/hooks/configOverrides/useHashicorpVaultConfig";
import { useUpdateHashicorpVaultConfig } from "@/app/(dashboard)/hooks/configOverrides/useUpdateHashicorpVaultConfig";

vi.mock("@/app/(dashboard)/hooks/configOverrides/useHashicorpVaultConfig", () => ({
  useHashicorpVaultConfig: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/configOverrides/useUpdateHashicorpVaultConfig", () => ({
  useUpdateHashicorpVaultConfig: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "sk-access-token" }),
}));

vi.mock("@/lib/toast", () => ({
  toast: { success: vi.fn(), fromError: vi.fn() },
}));

const ALL_FIELDS = [
  "vault_addr",
  "vault_namespace",
  "vault_mount_name",
  "vault_path_prefix",
  "vault_token",
  "approle_role_id",
  "approle_secret_id",
  "approle_mount_path",
  "client_cert",
  "client_key",
  "vault_cert_role",
] as const;

const propertiesFor = (fields: readonly string[]) =>
  Object.fromEntries(fields.map((name) => [name, { description: `${name} description` }]));

const mutate = vi.fn();

const setup = (options?: { values?: Record<string, unknown>; fields?: readonly string[] }) => {
  vi.mocked(useHashicorpVaultConfig).mockReturnValue({
    data: {
      field_schema: { properties: propertiesFor(options?.fields ?? ALL_FIELDS) },
      values: options?.values ?? {},
    },
  } as unknown as ReturnType<typeof useHashicorpVaultConfig>);

  vi.mocked(useUpdateHashicorpVaultConfig).mockReturnValue({
    mutate,
    isPending: false,
  } as unknown as ReturnType<typeof useUpdateHashicorpVaultConfig>);
};

const renderModal = (onSuccess = vi.fn(), onCancel = vi.fn()) =>
  renderWithProviders(<EditHashicorpVaultModal isVisible={true} onCancel={onCancel} onSuccess={onSuccess} />);

const save = async (user: ReturnType<typeof userEvent.setup>) =>
  user.click(screen.getByRole("button", { name: "Save" }));

describe("EditHashicorpVaultModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("clears untouched non-sensitive fields and omits untouched sensitive fields", async () => {
    setup({
      values: {
        vault_addr: "https://vault.example.com",
        vault_namespace: "team-ns",
        vault_token: "super-secret-token",
        approle_secret_id: "super-secret-id",
      },
    });
    const user = userEvent.setup();
    renderModal();

    await save(user);

    await waitFor(() => {
      expect(mutate).toHaveBeenCalledTimes(1);
    });
    expect(mutate.mock.calls[0][0]).toEqual({
      vault_addr: "https://vault.example.com",
      vault_namespace: "team-ns",
      vault_mount_name: "",
      vault_path_prefix: "",
      approle_role_id: "",
      approle_mount_path: "",
      client_cert: "",
      vault_cert_role: "",
    });
  });

  it("sends a sensitive field only once it is typed into", async () => {
    setup({ values: { vault_addr: "https://vault.example.com", vault_token: "super-secret-token" } });
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Token"), { target: { value: "rotated-token" } });
    await save(user);

    await waitFor(() => {
      expect(mutate).toHaveBeenCalledTimes(1);
    });
    expect(mutate.mock.calls[0][0]).toMatchObject({ vault_token: "rotated-token" });
  });

  it("never seeds a stored secret into its input", () => {
    setup({ values: { vault_token: "super-secret-token", approle_secret_id: "super-secret-id" } });
    renderModal();

    expect(screen.getByLabelText("Token")).toHaveValue("");
    expect(screen.getByLabelText("Secret ID")).toHaveValue("");
  });

  it("renders only the fields the schema declares, and sends only those", async () => {
    setup({ fields: ["vault_addr", "vault_token"], values: { vault_addr: "https://vault.example.com" } });
    const user = userEvent.setup();
    renderModal();

    expect(screen.queryByLabelText("Namespace")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Role ID")).not.toBeInTheDocument();

    await save(user);

    await waitFor(() => {
      expect(mutate).toHaveBeenCalledTimes(1);
    });
    expect(mutate.mock.calls[0][0]).toEqual({ vault_addr: "https://vault.example.com" });
  });

  it("blocks the submit when the vault address does not start with http", async () => {
    setup({ values: {} });
    const user = userEvent.setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Vault Address"), { target: { value: "vault.example.com" } });
    await save(user);

    expect(await screen.findByText("Must start with http:// or https://")).toBeInTheDocument();
    expect(mutate).not.toHaveBeenCalled();
  });

  it("accepts an empty vault address, because the pattern rule is not a required rule", async () => {
    setup({ values: {} });
    const user = userEvent.setup();
    renderModal();

    await save(user);

    await waitFor(() => {
      expect(mutate).toHaveBeenCalledTimes(1);
    });
    expect(mutate.mock.calls[0][0]).toMatchObject({ vault_addr: "" });
  });

  it("tells the admin a stored secret is kept when the field is left blank", () => {
    setup({ values: { vault_token: "super-secret-token" } });
    renderModal();

    expect(screen.getByLabelText("Token")).toHaveAttribute(
      "placeholder",
      "Leave blank to keep existing (super-secret-token)",
    );
  });

  it("falls back to the schema description when no secret is stored yet", () => {
    setup({ values: {} });
    renderModal();

    expect(screen.getByLabelText("Token")).toHaveAttribute("placeholder", "vault_token description");
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
