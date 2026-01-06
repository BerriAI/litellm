import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import DeleteSSOSettingsModal from "./DeleteSSOSettingsModal";

describe("DeleteSSOSettingsModal", () => {
  it("should render", () => {
    const onCancel = vi.fn();
    const onSuccess = vi.fn();

    render(
      <DeleteSSOSettingsModal isVisible={true} onCancel={onCancel} onSuccess={onSuccess} accessToken="test-token" />,
    );

    expect(screen.getByText("Confirm Clear SSO Settings")).toBeInTheDocument();
    expect(
      screen.getByText("Are you sure you want to clear all SSO settings? This action cannot be undone."),
    ).toBeInTheDocument();
    expect(screen.getByText("Users will no longer be able to login using SSO after this change.")).toBeInTheDocument();
  });
});
