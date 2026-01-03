import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import AddSSOSettingsModal from "./AddSSOSettingsModal";

// Mock networking functions
vi.mock("@/components/networking", () => ({
  updateSSOSettings: vi.fn(),
}));

// Mock error utils
vi.mock("@/components/shared/errorUtils", () => ({
  parseErrorMessage: vi.fn((error) => error?.message || "Unknown error"),
}));

describe("AddSSOSettingsModal", () => {
  it("should render", () => {
    const onCancel = vi.fn();
    const onSuccess = vi.fn();

    render(<AddSSOSettingsModal isVisible={true} onCancel={onCancel} onSuccess={onSuccess} accessToken="test-token" />);

    expect(screen.getByText("SSO Provider")).toBeInTheDocument();
    expect(screen.getByText("Cancel")).toBeInTheDocument();
    expect(screen.getAllByText("Add SSO")).toHaveLength(2); // Title and button
  });
});
