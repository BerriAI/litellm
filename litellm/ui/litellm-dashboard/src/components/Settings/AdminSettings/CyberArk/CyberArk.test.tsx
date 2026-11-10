import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../../../tests/test-utils";
import CyberArk from "./CyberArk";

const mockUseAuthorized = vi.hoisted(() => vi.fn());
const mockUseCyberArkConfig = vi.hoisted(() => vi.fn());

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: mockUseAuthorized,
}));

vi.mock("@/app/(dashboard)/hooks/configOverrides/useCyberArkConfig", () => ({
  useCyberArkConfig: mockUseCyberArkConfig,
}));

vi.mock("@/app/(dashboard)/hooks/configOverrides/useDeleteCyberArkConfig", () => ({
  useDeleteCyberArkConfig: () => ({ mutate: vi.fn(), isPending: false }),
}));

vi.mock("@/app/(dashboard)/hooks/configOverrides/useUpdateCyberArkConfig", () => ({
  useUpdateCyberArkConfig: () => ({ mutate: vi.fn(), isPending: false }),
}));

vi.mock("./EditCyberArkModal", () => ({
  default: ({ isVisible }: { isVisible: boolean }) => (isVisible ? <div>Edit CyberArk Configuration</div> : null),
}));

vi.mock("@/components/common_components/DeleteResourceModal", () => ({
  default: () => null,
}));

describe("CyberArk", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseAuthorized.mockReturnValue({ accessToken: "test-token" });
    const emptyConfigResult = {
      data: { values: {} },
      isLoading: false,
      isError: false,
      error: null,
    };
    mockUseCyberArkConfig.mockReturnValue(emptyConfigResult);
  });

  it("should render", () => {
    renderWithProviders(<CyberArk />);

    expect(screen.getByRole("heading", { name: "CyberArk Conjur" })).toBeInTheDocument();
  });

  it("should open the configuration editor from the empty state", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CyberArk />);

    await user.click(screen.getByRole("button", { name: /configure cyberark/i }));

    expect(screen.getByText("Edit CyberArk Configuration")).toBeInTheDocument();
  });

  it("should display configured values and management actions", () => {
    const configuredResult = {
      data: { values: { cyberark_api_base: "https://conjur.example.com", cyberark_api_key: "secret" } },
      isLoading: false,
      isError: false,
      error: null,
    };
    mockUseCyberArkConfig.mockReturnValue(configuredResult);

    renderWithProviders(<CyberArk />);

    expect(screen.getByText("https://conjur.example.com")).toBeInTheDocument();
    expect(screen.getByText("Auth Method")).toBeInTheDocument();
    expect(screen.getAllByText("API Key")).toHaveLength(2);
    expect(screen.getByRole("button", { name: /test connection/i })).toBeInTheDocument();
  });
});
