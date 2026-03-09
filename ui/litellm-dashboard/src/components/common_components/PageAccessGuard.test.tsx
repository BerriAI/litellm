import React, { useEffect } from "react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import PageAccessGuard from "./PageAccessGuard";

const isPageAccessibleForUserMock = vi.fn();

vi.mock("@/components/page_utils", () => ({
  getPageDisplayName: vi.fn(() => "Teams"),
}));

vi.mock("@/utils/page_access", () => ({
  isPageAccessibleForUser: (...args: unknown[]) => isPageAccessibleForUserMock(...args),
}));

const defaultPageAccessSettings = {
  enabledPagesInternalUsers: ["api-keys"] as string[] | null,
  enableProjectsUI: false,
  disableAgentsForInternalUsers: false,
  allowAgentsForTeamAdmins: false,
  disableVectorStoresForInternalUsers: false,
  allowVectorStoresForTeamAdmins: false,
  isLoading: false,
};

describe("PageAccessGuard", () => {
  it("should not mount child content while access settings are loading", () => {
    const onChildMount = vi.fn();
    isPageAccessibleForUserMock.mockReturnValue(true);

    const Child = () => {
      useEffect(() => {
        onChildMount();
      }, []);
      return <div data-testid="guard-child">child content</div>;
    };

    render(
      <PageAccessGuard
        page="teams"
        userRole="Internal User"
        teams={[]}
        organizations={[]}
        userId="u1"
        pageAccessSettings={{ ...defaultPageAccessSettings, isLoading: true }}
        onNavigateToDefault={vi.fn()}
      >
        <Child />
      </PageAccessGuard>,
    );

    expect(onChildMount).not.toHaveBeenCalled();
    expect(screen.queryByTestId("guard-child")).not.toBeInTheDocument();
  });

  it("should not mount child content when page access is denied", () => {
    const onChildMount = vi.fn();
    isPageAccessibleForUserMock.mockReturnValue(false);

    const Child = () => {
      useEffect(() => {
        onChildMount();
      }, []);
      return <div data-testid="guard-child">child content</div>;
    };

    render(
      <PageAccessGuard
        page="teams"
        userRole="Internal User"
        teams={[]}
        organizations={[]}
        userId="u1"
        pageAccessSettings={defaultPageAccessSettings}
        onNavigateToDefault={vi.fn()}
      >
        <Child />
      </PageAccessGuard>,
    );

    expect(screen.getByText("Access to Teams is restricted")).toBeInTheDocument();
    expect(onChildMount).not.toHaveBeenCalled();
    expect(screen.queryByTestId("guard-child")).not.toBeInTheDocument();
  });

  it("should mount child content when page access is allowed", () => {
    const onChildMount = vi.fn();
    isPageAccessibleForUserMock.mockReturnValue(true);

    const Child = () => {
      useEffect(() => {
        onChildMount();
      }, []);
      return <div data-testid="guard-child">child content</div>;
    };

    render(
      <PageAccessGuard
        page="teams"
        userRole="Internal User"
        teams={[]}
        organizations={[]}
        userId="u1"
        pageAccessSettings={defaultPageAccessSettings}
        onNavigateToDefault={vi.fn()}
      >
        <Child />
      </PageAccessGuard>,
    );

    expect(screen.getByTestId("guard-child")).toBeInTheDocument();
    expect(onChildMount).toHaveBeenCalledTimes(1);
  });
});
