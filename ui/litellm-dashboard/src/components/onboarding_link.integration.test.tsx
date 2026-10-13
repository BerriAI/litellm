import { afterEach, describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import OnboardingModal, { InvitationLink } from "./onboarding_link";

const invitation: InvitationLink = {
  id: "inv-123",
  user_id: "user-abc",
  is_accepted: false,
  accepted_at: null,
  expires_at: new Date("2030-01-01"),
  created_at: new Date("2029-12-01"),
  created_by: "admin",
  updated_at: new Date("2029-12-01"),
  updated_by: "admin",
  has_user_setup_sso: false,
};

const renderModal = (modalType: "invitation" | "resetPassword", setVisible = vi.fn()) =>
  render(
    <OnboardingModal
      isInvitationLinkModalVisible
      setIsInvitationLinkModalVisible={setVisible}
      baseUrl="http://localhost:4000/"
      invitationLinkData={invitation}
      modalType={modalType}
    />,
  );

describe("OnboardingModal", () => {
  it("shows the reset password link in a read-only field labelled for that flow", () => {
    renderModal("resetPassword");

    const field = screen.getByRole("textbox", { name: "Reset password link" });
    expect(field).toHaveValue("http://localhost:4000/ui/onboarding?invitation_id=inv-123&action=reset_password");
    expect(field).toHaveAttribute("readonly");
    expect(screen.getByText("user-abc")).toBeInTheDocument();
  });

  it("shows the invitation link in a read-only field labelled for that flow", () => {
    renderModal("invitation");

    const field = screen.getByRole("textbox", { name: "Invitation link" });
    expect(field).toHaveValue("http://localhost:4000/ui/onboarding?invitation_id=inv-123");
    expect(field).toHaveAttribute("readonly");
  });

  it("focuses the copy button on open so the link field shows the start of the URL", async () => {
    renderModal("resetPassword");

    const copyButton = screen.getByRole("button", { name: "Copy password reset link" });
    await waitFor(() => expect(copyButton).toHaveFocus());
  });

  it.each([
    ["invitation", "Copy invitation link", "http://localhost:4000/ui/onboarding?invitation_id=inv-123"],
    [
      "resetPassword",
      "Copy password reset link",
      "http://localhost:4000/ui/onboarding?invitation_id=inv-123&action=reset_password",
    ],
  ] as const)("copies exactly the displayed %s link when the copy button is pressed", async (modalType, label, url) => {
    const user = userEvent.setup();
    const writeText = vi.spyOn(navigator.clipboard, "writeText").mockResolvedValue();
    renderModal(modalType);

    await user.click(screen.getByRole("button", { name: label }));

    expect(writeText).toHaveBeenCalledWith(url);
  });

  describe("without the Clipboard API, as on a plain-http deployment", () => {
    const originalClipboard = Object.getOwnPropertyDescriptor(navigator, "clipboard");

    afterEach(() => {
      if (originalClipboard) Object.defineProperty(navigator, "clipboard", originalClipboard);
      Reflect.deleteProperty(document, "execCommand");
      vi.restoreAllMocks();
    });

    it("still copies exactly the displayed link through the selection fallback", () => {
      Object.defineProperty(navigator, "clipboard", { value: undefined, configurable: true });
      const selectedTexts: string[] = [];
      vi.spyOn(HTMLTextAreaElement.prototype, "select").mockImplementation(function (this: HTMLTextAreaElement) {
        selectedTexts.push(this.value);
      });
      const execCommand = vi.fn(() => true);
      document.execCommand = execCommand;
      renderModal("resetPassword");

      fireEvent.click(screen.getByRole("button", { name: "Copy password reset link" }));

      expect(selectedTexts).toEqual([
        "http://localhost:4000/ui/onboarding?invitation_id=inv-123&action=reset_password",
      ]);
      expect(execCommand).toHaveBeenCalledWith("copy");
    });

    it("keeps focus on the copy button after a fallback copy so Enter copies again", async () => {
      const user = userEvent.setup();
      Object.defineProperty(navigator, "clipboard", { value: undefined, configurable: true });
      const execCommand = vi.fn(() => true);
      document.execCommand = execCommand;
      renderModal("resetPassword");
      const copyButton = screen.getByRole("button", { name: "Copy password reset link" });
      await waitFor(() => expect(copyButton).toHaveFocus());

      await user.keyboard("{Enter}");
      await user.keyboard("{Enter}");

      expect(copyButton).toHaveFocus();
      expect(execCommand).toHaveBeenCalledTimes(2);
    });
  });

  it("asks the caller to hide the dialog when Escape is pressed", async () => {
    const user = userEvent.setup();
    const setVisible = vi.fn();
    renderModal("invitation", setVisible);

    await user.keyboard("{Escape}");

    expect(setVisible).toHaveBeenCalledWith(false);
  });
});
