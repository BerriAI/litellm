import React from "react";
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { LicenseExpiryBannerView } from "./LicenseExpiryBanner";
import { LicenseInfo } from "./networking";

const daysFromNow = (n: number): string => {
  const date = new Date();
  date.setUTCDate(date.getUTCDate() + n);
  return date.toISOString().slice(0, 10);
};

const licenseWith = (expiration_date: string | null): LicenseInfo => ({
  has_license: expiration_date !== null,
  license_type: expiration_date !== null ? "enterprise" : "community",
  expiration_date,
  allowed_features: [],
  limits: { max_users: null, max_teams: null },
});

describe("LicenseExpiryBannerView", () => {
  beforeEach(() => {
    sessionStorage.clear();
  });

  afterEach(() => {
    sessionStorage.clear();
  });

  it("renders nothing when there is no license info", () => {
    const { container } = render(<LicenseExpiryBannerView licenseInfo={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when expiration_date is null (community or remote-validated)", () => {
    const { container } = render(<LicenseExpiryBannerView licenseInfo={licenseWith(null)} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when expiry is more than 30 days out", () => {
    const { container } = render(<LicenseExpiryBannerView licenseInfo={licenseWith(daysFromNow(40))} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows a dismissible amber warning within 30 days", () => {
    const { container } = render(<LicenseExpiryBannerView licenseInfo={licenseWith(daysFromNow(20))} />);
    expect(screen.getByText(/expires in 20 days/)).toBeInTheDocument();
    expect(container.querySelector(".ant-alert-warning")).toBeInTheDocument();
    expect(screen.queryByRole("button")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "sales@berri.ai" })).toHaveAttribute("href", "mailto:sales@berri.ai");
  });

  it("shows a non-dismissible red critical alert within 7 days", () => {
    const { container } = render(<LicenseExpiryBannerView licenseInfo={licenseWith(daysFromNow(5))} />);
    expect(screen.getByText(/expires in 5 days/)).toBeInTheDocument();
    expect(container.querySelector(".ant-alert-error")).toBeInTheDocument();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("says 'expires today' on the expiration day", () => {
    render(<LicenseExpiryBannerView licenseInfo={licenseWith(daysFromNow(0))} />);
    expect(screen.getByText(/expires today/)).toBeInTheDocument();
  });

  it("shows a non-dismissible red expired alert stating features are disabled", () => {
    const { container } = render(<LicenseExpiryBannerView licenseInfo={licenseWith(daysFromNow(-3))} />);
    expect(screen.getByText(/expired on/)).toBeInTheDocument();
    expect(screen.getByText(/features are now disabled/i)).toBeInTheDocument();
    expect(container.querySelector(".ant-alert-error")).toBeInTheDocument();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("hides the warning after dismissal and stays hidden within the session", () => {
    const expiration = daysFromNow(20);
    const { unmount } = render(<LicenseExpiryBannerView licenseInfo={licenseWith(expiration)} />);
    fireEvent.click(screen.getByRole("button"));
    expect(screen.queryByText(/expires in 20 days/)).not.toBeInTheDocument();

    unmount();
    render(<LicenseExpiryBannerView licenseInfo={licenseWith(expiration)} />);
    expect(screen.queryByText(/expires in 20 days/)).not.toBeInTheDocument();
  });

  it("still shows a critical alert even when its date was previously dismissed", () => {
    const expiration = daysFromNow(5);
    sessionStorage.setItem(`litellm:licenseExpiryBannerDismissed:${expiration}`, "true");
    render(<LicenseExpiryBannerView licenseInfo={licenseWith(expiration)} />);
    expect(screen.getByText(/expires in 5 days/)).toBeInTheDocument();
  });
});
