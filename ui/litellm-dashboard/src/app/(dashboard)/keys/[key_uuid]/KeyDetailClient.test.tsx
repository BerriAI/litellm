import { render } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { useRouter, useParams } from "next/navigation";
import KeyDetailClient from "./KeyDetailClient";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

// Mock next/navigation
vi.mock("next/navigation", () => ({
  useRouter: vi.fn(),
  useParams: vi.fn(),
}));

// Mock useAuthorized hook
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: vi.fn(),
}));

describe("KeyDetailClient", () => {
  beforeEach(() => {
    vi.mocked(useRouter).mockReturnValue({
      push: vi.fn(),
      replace: vi.fn(),
      back: vi.fn(),
    } as any);

    vi.mocked(useParams).mockReturnValue({
      key_uuid: "test-key-uuid-123",
    });

    vi.mocked(useAuthorized).mockReturnValue({
      accessToken: "test-token",
      userRole: "admin",
    } as any);
  });

  it("should render the key detail client", () => {
    const { getByText } = render(<KeyDetailClient />);
    expect(getByText("Key Details")).toBeInTheDocument();
  });

  it("should display the key UUID", () => {
    const { getByText } = render(<KeyDetailClient />);
    expect(getByText("test-key-uuid-123")).toBeInTheDocument();
  });
});
