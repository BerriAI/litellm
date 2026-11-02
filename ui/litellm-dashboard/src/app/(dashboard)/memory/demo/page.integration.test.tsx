import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "../../../../../tests/test-utils";
import SharedKnowledgeDemoPage from "./page";

const { useAuthorizedMock } = vi.hoisted(() => ({ useAuthorizedMock: vi.fn() }));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: useAuthorizedMock }));

function renderAs(userRole: string) {
  const session = { accessToken: "demo-token", userId: "demo-user", userRole, premiumUser: false };
  useAuthorizedMock.mockReturnValue(session);
  return renderWithProviders(<SharedKnowledgeDemoPage />, { searchParams: "?step=3" });
}

describe("Shared knowledge demo page access", () => {
  beforeEach(() => {
    testQueryClient.clear();
    vi.clearAllMocks();
  });

  it("should render the requested demo step for an admin", () => {
    renderAs("Admin");

    expect(screen.getByRole("heading", { name: "Shared knowledge", exact: true })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Review corrections. Share the better answer." })).toBeInTheDocument();
  });

  it.each(["Internal User", "Internal Viewer", "Org Admin", "Unknown Role"])(
    "should show the Memory access notice instead of the demo for %s",
    (userRole) => {
      renderAs(userRole);

      expect(screen.getByText("Memory is only available to admin users.")).toBeInTheDocument();
      expect(screen.queryByRole("navigation", { name: "Demo steps" })).not.toBeInTheDocument();
    },
  );
});
