import { render, screen } from "@testing-library/react";
import OnboardingModal, { InvitationLink } from "./onboarding_link";

const baseLinkData: InvitationLink = {
  id: "inv-123",
  user_id: "user-456",
  is_accepted: false,
  accepted_at: null,
  expires_at: new Date("2026-04-01"),
  created_at: new Date("2026-03-01"),
  created_by: "admin",
  updated_at: new Date("2026-03-01"),
  updated_by: "admin",
  has_user_setup_sso: false,
};

describe("OnboardingModal", () => {
  const defaultProps = {
    isInvitationLinkModalVisible: true,
    setIsInvitationLinkModalVisible: vi.fn(),
    baseUrl: "https://proxy.example.com",
    invitationLinkData: baseLinkData,
  };

  it("should render", () => {
    render(<OnboardingModal {...defaultProps} />);
    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });

  it("should display the user ID", () => {
    render(<OnboardingModal {...defaultProps} />);
    expect(screen.getByText("user-456")).toBeInTheDocument();
  });

  it("should generate an invitation URL with invitation_id param", () => {
    render(<OnboardingModal {...defaultProps} />);
    expect(
      screen.getByText("https://proxy.example.com/ui?invitation_id=inv-123")
    ).toBeInTheDocument();
  });

  it("should show reset password title when modalType is resetPassword", () => {
    render(<OnboardingModal {...defaultProps} modalType="resetPassword" />);
    expect(screen.getAllByText("Reset Password Link").length).toBeGreaterThanOrEqual(1);
  });

  it("should append action=reset_password to URL for resetPassword type", () => {
    render(<OnboardingModal {...defaultProps} modalType="resetPassword" />);
    expect(
      screen.getByText("https://proxy.example.com/ui?invitation_id=inv-123&action=reset_password")
    ).toBeInTheDocument();
  });

  it("should show plain UI URL when user has SSO setup", () => {
    const ssoLink = { ...baseLinkData, has_user_setup_sso: true };
    render(<OnboardingModal {...defaultProps} invitationLinkData={ssoLink} />);
    expect(screen.getByText("https://proxy.example.com/ui")).toBeInTheDocument();
  });

  it("should show copy invitation link button by default", () => {
    render(<OnboardingModal {...defaultProps} />);
    expect(screen.getByRole("button", { name: /copy invitation link/i })).toBeInTheDocument();
  });

  it("should show copy password reset link button for resetPassword type", () => {
    render(<OnboardingModal {...defaultProps} modalType="resetPassword" />);
    expect(screen.getByRole("button", { name: /copy password reset link/i })).toBeInTheDocument();
  });
});
