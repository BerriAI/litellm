import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import SSOSettings from "./SSOSettings";

// Mock the useSSOSettings hook
vi.mock("@/app/(dashboard)/hooks/sso/useSSOSettings", () => ({
  useSSOSettings: () => ({
    data: null,
    refetch: vi.fn(),
  }),
}));

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });

describe("SSOSettings", () => {
  it("should render", () => {
    const queryClient = createQueryClient();

    render(
      <QueryClientProvider client={queryClient}>
        <SSOSettings />
      </QueryClientProvider>,
    );

    expect(screen.getByText("SSO Configuration")).toBeInTheDocument();
    expect(screen.getByText("Manage Single Sign-On authentication settings")).toBeInTheDocument();
  });
});
