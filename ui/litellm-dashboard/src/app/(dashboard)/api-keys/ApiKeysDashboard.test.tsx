import { render } from "@testing-library/react";
import { beforeEach, describe, it, expect, vi } from "vitest";

const translationState = vi.hoisted(() => ({ language: "en" as "en" | "ru" }));

vi.mock("react-i18next", async () => {
  const { resources } = await import("@/i18n/catalog");
  return {
    useTranslation: () => ({
      t: (key: string) =>
        key.split(".").reduce<unknown>((copy, segment) => {
          if (typeof copy !== "object" || copy === null) return undefined;
          return (copy as Record<string, unknown>)[segment];
        }, resources[translationState.language].gateway) ?? key,
    }),
  };
});

const { userDashboardSpy } = vi.hoisted(() => ({
  userDashboardSpy: vi.fn((_props: Record<string, unknown>) => null),
}));

vi.mock("@/components/user_dashboard", () => ({
  default: (props: Record<string, unknown>) => userDashboardSpy(props),
}));

// AuthContext is still hydrating: userID has not been populated yet (the regression).
vi.mock("@/contexts/AuthContext", () => ({
  useAuth: () => ({
    userID: null,
    userRole: "",
    userEmail: null,
    accessToken: null,
    premiumUser: false,
    setUserRole: vi.fn(),
    setUserEmail: vi.fn(),
  }),
}));

// useAuthorized decodes the cookie synchronously, so identity is already available.
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({
    isLoading: false,
    isAuthorized: true,
    token: "jwt",
    accessToken: "sk-access",
    userId: "u-123",
    userEmail: "admin@example.com",
    userRole: "Admin",
    premiumUser: false,
    disabledPersonalKeyCreation: false,
    showSSOBanner: false,
  }),
}));

vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  teamListCall: vi.fn(() => new Promise(() => {})),
}));

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(""),
}));

import ApiKeysDashboard from "./ApiKeysDashboard";

describe("ApiKeysDashboard identity source", () => {
  beforeEach(() => {
    translationState.language = "en";
    userDashboardSpy.mockClear();
  });

  it("passes the useAuthorized userID through even while AuthContext.userID is still null", () => {
    render(<ApiKeysDashboard />);

    expect(userDashboardSpy).toHaveBeenCalled();
    const props = userDashboardSpy.mock.calls[0][0];
    expect(props.userID).toBe("u-123");
  });

  it("passes Russian page-level copy to the legacy key dashboard", () => {
    translationState.language = "ru";
    render(<ApiKeysDashboard />);

    const props = userDashboardSpy.mock.calls[0][0];
    expect(props.createKeyLabel).toBe("+ Создать ключ");
    expect(props.missingUserLabel).toBe("ID пользователя не задан");
  });
});
