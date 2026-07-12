import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import LDAPSettings from "./LDAPSettings";

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "test-token" }),
}));

vi.mock("@/app/(dashboard)/hooks/ldap/useLDAPSettings", () => ({
  useLDAPSettings: () => ({
    data: { values: { ldap_enabled: false } },
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn(),
  }),
}));

vi.mock("@/app/(dashboard)/hooks/ldap/useUpdateLDAPSettings", () => ({
  useUpdateLDAPSettings: () => ({ mutate: vi.fn(), isPending: false }),
}));

describe("LDAPSettings", () => {
  it("exposes stable identity and explicit insecure transport settings", () => {
    render(<LDAPSettings />);

    expect(screen.getByText("Stable User ID Attribute")).toBeInTheDocument();
    expect(screen.getByText("Allow Insecure LDAP")).toBeInTheDocument();
  });
});
