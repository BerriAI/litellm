import { screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../../../../tests/test-utils";
import AddSSOSettingsModal from "./AddSSOSettingsModal";

// Mock networking functions
vi.mock("@/components/networking", () => ({
  updateSSOSettings: vi.fn(),
}));

// Mock error utils
vi.mock("@/components/shared/errorUtils", () => ({
  parseErrorMessage: vi.fn((error) => error?.message || "Unknown error"),
}));

// Mock the useAuthorized hook
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({
    accessToken: "test-access-token",
    userId: "test-user-id",
    userEmail: "test@example.com",
    userRole: "admin",
  }),
}));

// Mock NotificationsManager
vi.mock("@/components/molecules/notifications_manager", () => ({
  default: {
    success: vi.fn(),
    fromBackend: vi.fn(),
  },
}));

describe("AddSSOSettingsModal", () => {
  it("should render", () => {
    const onCancel = vi.fn();
    const onSuccess = vi.fn();

    renderWithProviders(<AddSSOSettingsModal isVisible={true} onCancel={onCancel} onSuccess={onSuccess} />);

    expect(screen.getByText("SSO Provider")).toBeInTheDocument();
    expect(screen.getByText("Cancel")).toBeInTheDocument();
    expect(screen.getAllByText("Add SSO")).toHaveLength(2); // Title and button
  });
});
