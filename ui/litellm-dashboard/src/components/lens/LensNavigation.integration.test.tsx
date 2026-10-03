import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "@/../tests/test-utils";
import LensPage from "@/app/(dashboard)/lens/page";

const { auth } = vi.hoisted(() => ({ auth: vi.fn() }));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: auth }));
vi.mock("@/components/view_logs/TraceView/AgentTracesPage", () => ({
  default: ({ isActive }: { isActive: boolean }) => <div>Trace polling {isActive ? "active" : "paused"}</div>,
}));
vi.mock("./investigations/InvestigationsView", () => ({
  InvestigationsView: ({ readOnly }: { readOnly: boolean }) => (
    <div>{readOnly ? "Read-only investigations" : "Manage investigations"}</div>
  ),
}));

describe("Lens navigation", () => {
  beforeEach(() => auth.mockReturnValue({ accessToken: "test-token", userRole: "Admin", isViewOnly: false }));

  it("opens traces by default and pauses polling while viewing investigations", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWithProviders(<LensPage />, { onUrlUpdate });
    expect(screen.getByRole("tab", { name: "Traces" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("Trace polling active")).toBeVisible();
    await user.click(screen.getByRole("tab", { name: "Investigations" }));
    expect(screen.getByText("Manage investigations")).toBeVisible();
    expect(screen.getByText("Trace polling paused")).not.toBeVisible();
    expect(onUrlUpdate.mock.lastCall?.[0].searchParams.get("tab")).toBe("investigations");
    await user.click(screen.getByRole("tab", { name: "Traces" }));
    expect(screen.getByText("Trace polling active")).toBeVisible();
  });

  it("opens existing lens links on findings", () => {
    renderWithProviders(<LensPage />, { searchParams: "?lens=saved-lens" });
    expect(screen.getByRole("tab", { name: "Findings" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("Manage investigations")).toBeVisible();
  });

  it("honors an explicit traces tab even when a saved investigation is in the URL", () => {
    renderWithProviders(<LensPage />, { searchParams: "?tab=traces&lens=saved-lens" });
    expect(screen.getByRole("tab", { name: "Traces" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("Trace polling active")).toBeVisible();
  });

  it.each(["Internal User", "Internal Viewer", "Org Admin"])(
    "preserves trace access without granting investigations to %s",
    async (userRole) => {
      auth.mockReturnValue({ accessToken: "test-token", userRole, isViewOnly: false });
      const user = userEvent.setup();
      renderWithProviders(<LensPage />);
      expect(screen.getByText("Trace polling active")).toBeVisible();
      await user.click(screen.getByRole("tab", { name: "Investigations" }));
      expect(screen.getByText(/Investigations require proxy administrator access/)).toBeVisible();
      expect(screen.queryByText("Manage investigations")).not.toBeInTheDocument();
    },
  );

  it.each([
    { userRole: "Admin Viewer", isViewOnly: false },
    { userRole: "Admin", isViewOnly: true },
  ])("preserves read-only investigation access for $userRole with isViewOnly=$isViewOnly", (session) => {
    auth.mockReturnValue({ accessToken: "test-token", ...session });
    renderWithProviders(<LensPage />, { searchParams: "?tab=investigations" });
    expect(screen.getByText("Read-only investigations")).toBeVisible();
    expect(screen.queryByText("Manage investigations")).not.toBeInTheDocument();
  });
});
