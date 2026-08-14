import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import OnboardingModal, { buildOnboardingUrl, InvitationLink } from "./onboarding_link";

vi.mock("./molecules/notifications_manager", () => ({ default: { success: vi.fn() } }));

const invitation: InvitationLink = {
  id: "inv-123",
  user_id: "user-abc",
  is_accepted: false,
  accepted_at: null,
  expires_at: new Date("2026-09-01"),
  created_at: new Date("2026-08-01"),
  created_by: "admin",
  updated_at: new Date("2026-08-01"),
  updated_by: "admin",
  has_user_setup_sso: false,
};

describe("buildOnboardingUrl", () => {
  it("points the invitation link at the dedicated /ui/onboarding route", () => {
    expect(
      buildOnboardingUrl({
        baseUrl: "http://localhost:4000/",
        invitationId: "inv-123",
        hasUserSetupSso: false,
        resetPassword: false,
      }),
    ).toBe("http://localhost:4000/ui/onboarding?invitation_id=inv-123");
  });

  it("preserves a server_root_path prefix before /ui/onboarding", () => {
    expect(
      buildOnboardingUrl({
        baseUrl: "https://proxy.example.com/litellm",
        invitationId: "inv-123",
        hasUserSetupSso: false,
        resetPassword: false,
      }),
    ).toBe("https://proxy.example.com/litellm/ui/onboarding?invitation_id=inv-123");
  });

  it("appends action=reset_password for the reset-password flow", () => {
    expect(
      buildOnboardingUrl({
        baseUrl: "http://localhost:4000/",
        invitationId: "inv-123",
        hasUserSetupSso: false,
        resetPassword: true,
      }),
    ).toBe("http://localhost:4000/ui/onboarding?invitation_id=inv-123&action=reset_password");
  });

  it("sends SSO users to the dashboard root, not the onboarding form", () => {
    expect(
      buildOnboardingUrl({
        baseUrl: "http://localhost:4000/",
        invitationId: "inv-123",
        hasUserSetupSso: true,
        resetPassword: false,
      }),
    ).toBe("http://localhost:4000/ui");
  });

  it("returns an empty string when no base URL is known yet", () => {
    expect(
      buildOnboardingUrl({
        baseUrl: "",
        invitationId: "inv-123",
        hasUserSetupSso: false,
        resetPassword: false,
      }),
    ).toBe("");
  });

  it("returns an empty string rather than an invitation_id=undefined link when the id is not ready", () => {
    expect(
      buildOnboardingUrl({
        baseUrl: "http://localhost:4000/",
        invitationId: undefined,
        hasUserSetupSso: false,
        resetPassword: false,
      }),
    ).toBe("");
  });
});

describe("OnboardingModal", () => {
  it("renders nothing until it is opened", () => {
    render(
      <OnboardingModal
        isInvitationLinkModalVisible={false}
        setIsInvitationLinkModalVisible={vi.fn()}
        baseUrl="http://localhost:4000/"
        invitationLinkData={invitation}
      />,
    );

    expect(screen.queryByText("http://localhost:4000/ui/onboarding?invitation_id=inv-123")).not.toBeInTheDocument();
  });

  it("shows the invitation url, the user id and an invitation-flavoured copy button", async () => {
    render(
      <OnboardingModal
        isInvitationLinkModalVisible
        setIsInvitationLinkModalVisible={vi.fn()}
        baseUrl="http://localhost:4000/"
        invitationLinkData={invitation}
      />,
    );

    expect(await screen.findByText("http://localhost:4000/ui/onboarding?invitation_id=inv-123")).toBeInTheDocument();
    expect(screen.getByText("user-abc")).toBeInTheDocument();
    expect(screen.getAllByText("Invitation Link").length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "Copy invitation link" })).toBeInTheDocument();
    expect(screen.getByText(/Copy and send the generated link to onboard this user/)).toBeInTheDocument();
  });

  it("switches every label and the url to the reset-password flow", async () => {
    render(
      <OnboardingModal
        isInvitationLinkModalVisible
        setIsInvitationLinkModalVisible={vi.fn()}
        baseUrl="http://localhost:4000/"
        invitationLinkData={invitation}
        modalType="resetPassword"
      />,
    );

    expect(
      await screen.findByText("http://localhost:4000/ui/onboarding?invitation_id=inv-123&action=reset_password"),
    ).toBeInTheDocument();
    expect(screen.getAllByText("Reset Password Link").length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "Copy password reset link" })).toBeInTheDocument();
    expect(
      screen.getByText(/Copy and send the generated link to the user to reset their password/),
    ).toBeInTheDocument();
  });

  it("closes through setIsInvitationLinkModalVisible when the close control is used", async () => {
    const user = userEvent.setup();
    const setVisible = vi.fn();
    render(
      <OnboardingModal
        isInvitationLinkModalVisible
        setIsInvitationLinkModalVisible={setVisible}
        baseUrl="http://localhost:4000/"
        invitationLinkData={invitation}
      />,
    );

    await user.click(await screen.findByRole("button", { name: /close/i }));

    expect(setVisible).toHaveBeenCalledWith(false);
  });
});
