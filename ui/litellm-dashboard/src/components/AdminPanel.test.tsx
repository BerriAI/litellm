import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AdminPanel from "./AdminPanel";

const mockGetSSOSettings = vi.fn();
const mockGetAllowedIPs = vi.fn();
const mockAddAllowedIP = vi.fn();
const mockDeleteAllowedIP = vi.fn();

vi.mock("./networking", () => ({
  getSSOSettings: (...args: unknown[]) => mockGetSSOSettings(...args),
  getAllowedIPs: (...args: unknown[]) => mockGetAllowedIPs(...args),
  addAllowedIP: (...args: unknown[]) => mockAddAllowedIP(...args),
  deleteAllowedIP: (...args: unknown[]) => mockDeleteAllowedIP(...args),
}));

vi.mock("./constants", () => ({
  useBaseUrl: () => "http://localhost:4000",
}));

vi.mock("./Settings/AdminSettings/SSOSettings/SSOSettings", () => ({
  default: () => <div>SSO Settings</div>,
}));

vi.mock("./Settings/AdminSettings/UISettings/UISettings", () => ({
  default: () => <div>UI Settings</div>,
}));

vi.mock("./SCIM", () => ({
  default: () => <div>SCIM Config</div>,
}));

vi.mock("./SSOModals", () => ({
  default: () => <div>SSO Modals</div>,
}));

vi.mock("./UIAccessControlForm", () => ({
  default: () => <div>UI Access Control Form</div>,
}));

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

describe("AdminPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseAuthorized.mockReturnValue({
      premiumUser: false,
      accessToken: "test-token",
      userId: "user-1",
    });
    mockGetSSOSettings.mockResolvedValue({
      values: {},
    });
    mockGetAllowedIPs.mockResolvedValue([]);
    mockAddAllowedIP.mockResolvedValue({});
    mockDeleteAllowedIP.mockResolvedValue({});
  });

  it("should render the admin panel", () => {
    render(<AdminPanel />);
    expect(screen.getByRole("heading", { name: /admin access/i })).toBeInTheDocument();
    expect(screen.getByText(/go to 'internal users' page to add other admins/i)).toBeInTheDocument();
  });

  describe("Tabs", () => {
    it("should render all tabs", () => {
      render(<AdminPanel />);
      expect(screen.getByRole("tab", { name: /sso settings/i })).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: /security settings/i })).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: /scim/i })).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: /ui settings/i })).toBeInTheDocument();
    });

    it("should display Security Settings content when Security Settings tab is clicked", async () => {
      const user = userEvent.setup();
      render(<AdminPanel />);
      const securityTab = screen.getByRole("tab", { name: /security settings/i });
      await user.click(securityTab);
      expect(screen.getByRole("heading", { name: /security settings/i })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /add sso/i })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /allowed ips/i })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /ui access control/i })).toBeInTheDocument();
    });

    it("should display SCIM content when SCIM tab is clicked", async () => {
      const user = userEvent.setup();
      render(<AdminPanel />);
      const scimTab = screen.getByRole("tab", { name: /scim/i });
      await user.click(scimTab);
      expect(screen.getByText("SCIM Config")).toBeInTheDocument();
    });
  });

  describe("SSO Configuration", () => {
    it("should check SSO configuration on mount when accessToken is available", async () => {
      render(<AdminPanel />);
      await waitFor(() => {
        expect(mockGetSSOSettings).toHaveBeenCalledWith("test-token");
      });
    });

    it("should display 'Add SSO' button when SSO is not configured", async () => {
      const user = userEvent.setup();
      mockGetSSOSettings.mockResolvedValue({
        values: {},
      });
      render(<AdminPanel />);
      const securityTab = screen.getByRole("tab", { name: /security settings/i });
      await user.click(securityTab);
      await waitFor(() => {
        expect(screen.getByRole("button", { name: /add sso/i })).toBeInTheDocument();
      });
    });

    it("should display 'Edit SSO Settings' button when SSO is configured", async () => {
      const user = userEvent.setup();
      mockGetSSOSettings.mockResolvedValue({
        values: {
          google_client_id: "test-id",
          google_client_secret: "test-secret",
        },
      });
      render(<AdminPanel />);
      const securityTab = screen.getByRole("tab", { name: /security settings/i });
      await user.click(securityTab);
      await waitFor(() => {
        expect(screen.getByRole("button", { name: /edit sso settings/i })).toBeInTheDocument();
      });
    });

    it("should detect Google SSO configuration", async () => {
      mockGetSSOSettings.mockResolvedValue({
        values: {
          google_client_id: "test-id",
          google_client_secret: "test-secret",
        },
      });
      render(<AdminPanel />);
      await waitFor(() => {
        expect(mockGetSSOSettings).toHaveBeenCalled();
      });
    });

    it("should detect Microsoft SSO configuration", async () => {
      mockGetSSOSettings.mockResolvedValue({
        values: {
          microsoft_client_id: "test-id",
          microsoft_client_secret: "test-secret",
        },
      });
      render(<AdminPanel />);
      await waitFor(() => {
        expect(mockGetSSOSettings).toHaveBeenCalled();
      });
    });

    it("should detect Generic SSO configuration", async () => {
      mockGetSSOSettings.mockResolvedValue({
        values: {
          generic_client_id: "test-id",
          generic_client_secret: "test-secret",
        },
      });
      render(<AdminPanel />);
      await waitFor(() => {
        expect(mockGetSSOSettings).toHaveBeenCalled();
      });
    });

    it("should handle SSO configuration check error gracefully", async () => {
      mockGetSSOSettings.mockRejectedValue(new Error("Network error"));
      render(<AdminPanel />);
      await waitFor(() => {
        expect(mockGetSSOSettings).toHaveBeenCalled();
      });
    });
  });

  describe("Allowed IPs", () => {
    beforeEach(async () => {
      const user = userEvent.setup();
      mockUseAuthorized.mockReturnValue({
        premiumUser: true,
        accessToken: "test-token",
        userId: "user-1",
      });
      render(<AdminPanel />);
      const securityTab = screen.getByRole("tab", { name: /security settings/i });
      await user.click(securityTab);
    });

    it("should open allowed IPs modal when premium user clicks Allowed IPs button", async () => {
      const user = userEvent.setup();
      mockGetAllowedIPs.mockResolvedValue(["192.168.1.1", "10.0.0.1"]);
      const allowedIPsButton = screen.getByRole("button", { name: /allowed ips/i });
      await user.click(allowedIPsButton);
      await waitFor(() => {
        expect(screen.getByRole("dialog", { name: /manage allowed ip addresses/i })).toBeInTheDocument();
      });
    });

    it("should display 'All IP Addresses Allowed' when no IPs are configured", async () => {
      const user = userEvent.setup();
      mockGetAllowedIPs.mockResolvedValue([]);
      const allowedIPsButton = screen.getByRole("button", { name: /allowed ips/i });
      await user.click(allowedIPsButton);
      await waitFor(() => {
        expect(screen.getByText("All IP Addresses Allowed")).toBeInTheDocument();
      });
    });

    it("should display list of allowed IPs", async () => {
      const user = userEvent.setup();
      mockGetAllowedIPs.mockResolvedValue(["192.168.1.1", "10.0.0.1"]);
      const allowedIPsButton = screen.getByRole("button", { name: /allowed ips/i });
      await user.click(allowedIPsButton);
      await waitFor(() => {
        expect(screen.getByText("192.168.1.1")).toBeInTheDocument();
        expect(screen.getByText("10.0.0.1")).toBeInTheDocument();
      });
    });

    it("should show delete button for IP addresses except 'All IP Addresses Allowed'", async () => {
      const user = userEvent.setup();
      mockGetAllowedIPs.mockResolvedValue(["192.168.1.1", "All IP Addresses Allowed"]);
      const allowedIPsButton = screen.getByRole("button", { name: /allowed ips/i });
      await user.click(allowedIPsButton);
      await waitFor(() => {
        const deleteButtons = screen.queryAllByRole("button", { name: /delete/i });
        expect(deleteButtons.length).toBeGreaterThan(0);
      });
    });

    it("should not show delete button for 'All IP Addresses Allowed'", async () => {
      const user = userEvent.setup();
      mockGetAllowedIPs.mockResolvedValue(["All IP Addresses Allowed"]);
      const allowedIPsButton = screen.getByRole("button", { name: /allowed ips/i });
      await user.click(allowedIPsButton);
      await waitFor(() => {
        expect(screen.getByText("All IP Addresses Allowed")).toBeInTheDocument();
      });
      const deleteButtons = screen.queryAllByRole("button", { name: /delete/i });
      expect(deleteButtons.length).toBe(0);
    });

    it("should handle error when fetching allowed IPs fails", async () => {
      const user = userEvent.setup();
      mockGetAllowedIPs.mockRejectedValue(new Error("Network error"));
      const allowedIPsButton = screen.getByRole("button", { name: /allowed ips/i });
      await user.click(allowedIPsButton);
      await waitFor(() => {
        expect(mockGetAllowedIPs).toHaveBeenCalled();
      });
    });
  });

  describe("UI Access Control", () => {
    it("should show premium user message when non-premium user tries to access UI Access Control", async () => {
      const user = userEvent.setup();
      mockUseAuthorized.mockReturnValue({
        premiumUser: false,
        accessToken: "test-token",
        userId: "user-1",
      });
      render(<AdminPanel />);
      const securityTab = screen.getByRole("tab", { name: /security settings/i });
      await user.click(securityTab);
      const uiAccessControlButton = screen.getByRole("button", { name: /ui access control/i });
      await user.click(uiAccessControlButton);
      await waitFor(() => {
        expect(screen.queryByRole("dialog", { name: /ui access control settings/i })).not.toBeInTheDocument();
      });
    });

    it("should open UI Access Control modal when premium user clicks button", async () => {
      const user = userEvent.setup();
      mockUseAuthorized.mockReturnValue({
        premiumUser: true,
        accessToken: "test-token",
        userId: "user-1",
      });
      render(<AdminPanel />);
      const securityTab = screen.getByRole("tab", { name: /security settings/i });
      await user.click(securityTab);
      const uiAccessControlButton = screen.getByRole("button", { name: /ui access control/i });
      await user.click(uiAccessControlButton);
      await waitFor(() => {
        expect(screen.getByRole("dialog", { name: /ui access control settings/i })).toBeInTheDocument();
        expect(screen.getByText("UI Access Control Form")).toBeInTheDocument();
      });
    });
  });

  describe("Login without SSO", () => {
    it("should display fallback login URL", async () => {
      const user = userEvent.setup();
      render(<AdminPanel />);
      const securityTab = screen.getByRole("tab", { name: /security settings/i });
      await user.click(securityTab);
      const link = screen.getByRole("link", { name: /http:\/\/localhost:4000\/fallback\/login/i });
      expect(link).toBeInTheDocument();
      expect(link).toHaveAttribute("href", "http://localhost:4000/fallback/login");
      expect(link).toHaveAttribute("target", "_blank");
    });
  });

  describe("SSO Configuration Deprecation Warning", () => {
    it("should display deprecation warning in Security Settings tab", async () => {
      const user = userEvent.setup();
      render(<AdminPanel />);
      const securityTab = screen.getByRole("tab", { name: /security settings/i });
      await user.click(securityTab);
      await waitFor(() => {
        expect(screen.getByText(/sso configuration deprecated/i)).toBeInTheDocument();
        expect(
          screen.getByText(/editing sso settings on this page is deprecated and will be removed/i),
        ).toBeInTheDocument();
      });
    });
  });
});
