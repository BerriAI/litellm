import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../../../tests/test-utils";
import HashicorpVault from "./HashicorpVault";

const mockUseAuthorized = vi.hoisted(() => vi.fn());
const mockUseHashicorpVaultConfig = vi.hoisted(() => vi.fn());

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: mockUseAuthorized,
}));

vi.mock("@/app/(dashboard)/hooks/configOverrides/useHashicorpVaultConfig", () => ({
  useHashicorpVaultConfig: mockUseHashicorpVaultConfig,
}));

vi.mock("@/app/(dashboard)/hooks/configOverrides/useDeleteHashicorpVaultConfig", () => ({
  useDeleteHashicorpVaultConfig: () => ({ mutate: vi.fn(), isPending: false }),
}));

vi.mock("@/app/(dashboard)/hooks/configOverrides/useUpdateHashicorpVaultConfig", () => ({
  useUpdateHashicorpVaultConfig: () => ({ mutate: vi.fn(), isPending: false }),
}));

vi.mock("./EditHashicorpVaultModal", () => ({
  default: ({ isVisible }: { isVisible: boolean }) => (isVisible ? <div>Edit Vault Configuration</div> : null),
}));

vi.mock("@/components/common_components/DeleteResourceModal", () => ({
  default: () => null,
}));

describe("HashicorpVault", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseAuthorized.mockReturnValue({ accessToken: "test-token" });
    mockUseHashicorpVaultConfig.mockReturnValue({
      data: { values: {} },
      isLoading: false,
      isError: false,
      error: null,
    });
  });

  it("should render", () => {
    renderWithProviders(<HashicorpVault />);

    expect(screen.getByRole("heading", { name: "Hashicorp Vault" })).toBeInTheDocument();
  });

  it("should open the configuration editor from the empty state", async () => {
    const user = userEvent.setup();
    renderWithProviders(<HashicorpVault />);

    await user.click(screen.getByRole("button", { name: /configure vault/i }));

    expect(screen.getByText("Edit Vault Configuration")).toBeInTheDocument();
  });

  it("should display configured values and management actions", () => {
    mockUseHashicorpVaultConfig.mockReturnValue({
      data: { values: { vault_addr: "https://vault.example.com", vault_token: "secret" } },
      isLoading: false,
      isError: false,
      error: null,
    });

    renderWithProviders(<HashicorpVault />);

    expect(screen.getByText("https://vault.example.com")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /test connection/i })).toBeInTheDocument();
  });
});
