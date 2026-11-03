import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ModelHubTablePage from "./page";

const mockUseAuthorized = vi.fn();
const hubProps = vi.fn();

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => mockUseAuthorized() }));
vi.mock("@/components/AIHub/ModelHubTable", () => ({
  default: (props: { canAddModels: boolean }) => {
    hubProps(props);
    return <div data-testid="model-hub-table" />;
  },
}));
vi.mock("@/components/public_model_hub", () => ({
  default: () => <div data-testid="public-model-hub" />,
}));

const session = (userRole: string, isViewOnly = false) => ({
  accessToken: "token",
  userRole,
  userId: "user-1",
  premiumUser: false,
  isViewOnly,
});

describe("ModelHubTablePage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("lets a proxy admin add models from the garden", () => {
    mockUseAuthorized.mockReturnValue(session("Admin"));
    render(<ModelHubTablePage />);
    expect(hubProps).toHaveBeenCalledWith(expect.objectContaining({ canAddModels: true }));
  });

  it("keeps the garden read-only for a view-only admin", () => {
    mockUseAuthorized.mockReturnValue(session("Admin", true));
    render(<ModelHubTablePage />);
    expect(hubProps).toHaveBeenCalledWith(expect.objectContaining({ canAddModels: false }));
  });

  it("sends an org admin to the public hub under the role string a session carries", () => {
    mockUseAuthorized.mockReturnValue(session("Org Admin"));
    render(<ModelHubTablePage />);
    expect(screen.getByTestId("public-model-hub")).toBeInTheDocument();
    expect(hubProps).not.toHaveBeenCalled();
  });

  it("sends a non-admin to the public hub", () => {
    mockUseAuthorized.mockReturnValue(session("internal_user"));
    render(<ModelHubTablePage />);
    expect(screen.getByTestId("public-model-hub")).toBeInTheDocument();
    expect(hubProps).not.toHaveBeenCalled();
  });
});
