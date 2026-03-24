import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { LicenseExpiryBanner } from "./LicenseExpiryBanner";

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: vi.fn(() => ({ accessToken: "test-token" })),
}));

vi.mock("./networking", () => ({
  getLicenseInfo: vi.fn().mockResolvedValue(null),
}));

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { getLicenseInfo } from "./networking";

const mockUseAuthorized = vi.mocked(useAuthorized);
const mockGetLicenseInfo = vi.mocked(getLicenseInfo);

const FIXED_NOW = new Date("2026-03-15T12:00:00Z").getTime();
let realDateNow: () => number;
let OrigDate: typeof Date;

function makeLicense(expirationDate: string) {
  return {
    has_license: true,
    license_type: "enterprise",
    expiration_date: expirationDate,
    allowed_features: [] as string[],
    limits: { max_users: null, max_teams: null },
  };
}

describe("LicenseExpiryBanner", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    realDateNow = Date.now;
    OrigDate = Date; // capture before patching so we can restore in afterEach
    Date.now = () => FIXED_NOW;
    const MockDate = class extends OrigDate {
      constructor(...args: any[]) {
        if (args.length === 0) {
          super(FIXED_NOW);
        } else {
          // @ts-ignore
          super(...args);
        }
      }
    };
    // @ts-ignore
    global.Date = MockDate;
    // Preserve static methods
    global.Date.now = () => FIXED_NOW;
    global.Date.parse = OrigDate.parse;
    global.Date.UTC = OrigDate.UTC;

    mockUseAuthorized.mockReturnValue({
      accessToken: "test-token",
      isLoading: false,
      isAuthorized: true,
      token: "tok",
      userId: "u1",
      userEmail: "u@e.com",
      userRole: "proxy_admin",
      premiumUser: true,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    } as any);
    mockGetLicenseInfo.mockResolvedValue(null);
  });

  afterEach(() => {
    global.Date = OrigDate;
    Date.now = realDateNow;
  });

  it("should render nothing when there is no license info", async () => {
    const { container } = render(<LicenseExpiryBanner />);
    await waitFor(() => {
      expect(mockGetLicenseInfo).toHaveBeenCalledWith("test-token");
    });
    expect(container.innerHTML).toBe("");
  });

  it("should render nothing when expiration is more than 14 days away", async () => {
    mockGetLicenseInfo.mockResolvedValue(makeLicense("2026-04-15"));

    const { container } = render(<LicenseExpiryBanner />);
    await waitFor(() => {
      expect(mockGetLicenseInfo).toHaveBeenCalled();
    });
    await waitFor(() => {
      expect(screen.queryByRole("alert")).toBeNull();
    });
  });

  it("should show a warning banner when license expires within 14 days", async () => {
    // Now is 2026-03-15T12:00:00Z, exp is 2026-03-22T23:59:59Z (UTC) => ~7.5 days => ceil = 8
    mockGetLicenseInfo.mockResolvedValue(makeLicense("2026-03-22"));

    render(<LicenseExpiryBanner />);

    expect(
      await screen.findByText("Enterprise License Expiring in 8 days")
    ).toBeInTheDocument();
  });

  it("should show singular 'day' when 1 day remains", async () => {
    // Now is 2026-03-15T12:00:00Z, exp is 2026-03-15T23:59:59Z (UTC) => ~0.5 days => ceil = 1
    mockGetLicenseInfo.mockResolvedValue(makeLicense("2026-03-15"));

    render(<LicenseExpiryBanner />);

    expect(
      await screen.findByText("Enterprise License Expiring in 1 day")
    ).toBeInTheDocument();
  });

  it("should show an error banner when the license has expired", async () => {
    mockGetLicenseInfo.mockResolvedValue(makeLicense("2026-03-10"));

    render(<LicenseExpiryBanner />);

    expect(
      await screen.findByText("Enterprise License Expired")
    ).toBeInTheDocument();
  });

  it("should include renewal instructions in the description", async () => {
    mockGetLicenseInfo.mockResolvedValue(makeLicense("2026-03-20"));

    render(<LicenseExpiryBanner />);

    expect(
      await screen.findByText(/Please contact support to renew before expiration/i)
    ).toBeInTheDocument();
  });

  it("should not render when accessToken is null", async () => {
    mockUseAuthorized.mockReturnValue({ accessToken: null } as any);

    const { container } = render(<LicenseExpiryBanner />);
    await waitFor(() => {
      expect(mockGetLicenseInfo).not.toHaveBeenCalled();
    });
    expect(container.innerHTML).toBe("");
  });

  it("should handle expiration_date with a full ISO timestamp (not just date-only)", async () => {
    // Backend might return "2026-03-22T15:00:00Z" instead of "2026-03-22"
    // The component should strip the time part and still render correctly.
    mockGetLicenseInfo.mockResolvedValue(makeLicense("2026-03-22T15:00:00Z"));

    render(<LicenseExpiryBanner />);

    expect(
      await screen.findByText("Enterprise License Expiring in 8 days")
    ).toBeInTheDocument();
  });

  it("should not render when has_license is false", async () => {
    mockGetLicenseInfo.mockResolvedValue({
      ...makeLicense("2026-03-20"),
      has_license: false,
    });

    const { container } = render(<LicenseExpiryBanner />);
    await waitFor(() => {
      expect(mockGetLicenseInfo).toHaveBeenCalled();
    });
    await waitFor(() => {
      expect(screen.queryByRole("alert")).toBeNull();
    });
  });

  it("should show a warning when getLicenseInfo API call fails", async () => {
    mockGetLicenseInfo.mockRejectedValue(new Error("Network error"));

    render(<LicenseExpiryBanner />);

    expect(
      await screen.findByText("Unable to verify enterprise license")
    ).toBeInTheDocument();
  });

  it("should render nothing when expiration_date is unparseable", async () => {
    mockGetLicenseInfo.mockResolvedValue(makeLicense("N/A"));

    const { container } = render(<LicenseExpiryBanner />);
    await waitFor(() => {
      expect(mockGetLicenseInfo).toHaveBeenCalled();
    });
    await waitFor(() => {
      expect(screen.queryByRole("alert")).toBeNull();
    });
  });
});
