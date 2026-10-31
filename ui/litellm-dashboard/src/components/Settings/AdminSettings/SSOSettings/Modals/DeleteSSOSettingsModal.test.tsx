import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import DeleteSSOSettingsModal from "./DeleteSSOSettingsModal";

vi.mock("@/app/(dashboard)/hooks/sso/useSSOSettings", () => ({
  useSSOSettings: vi.fn(() => ({
    data: {
      values: {
        google_client_id: "test-client-id",
      },
    },
  })),
}));

const mutateAsync = vi.hoisted(() => vi.fn());

vi.mock("@/app/(dashboard)/hooks/sso/useEditSSOSettings", () => ({
  useEditSSOSettings: vi.fn(() => ({
    mutateAsync,
    isPending: false,
  })),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: vi.fn(() => ({
    accessToken: "test-token",
    userId: "test-user-id",
    userRole: "proxy_admin",
  })),
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

describe("DeleteSSOSettingsModal", () => {
  it("should render", () => {
    const onCancel = vi.fn();
    const onSuccess = vi.fn();
    const queryClient = createQueryClient();

    render(
      <QueryClientProvider client={queryClient}>
        <DeleteSSOSettingsModal isVisible={true} onCancel={onCancel} onSuccess={onSuccess} />
      </QueryClientProvider>,
    );

    expect(screen.getByText("Confirm Clear SSO Settings")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Are you sure you want to clear all SSO settings? Users will no longer be able to login using SSO after this change.",
      ),
    ).toBeInTheDocument();
  });

  it("nulls every stored field so nothing survives a clear", async () => {
    const queryClient = createQueryClient();
    const clearedFields = {
      generic_scope: null,
      generic_client_secret: null,
      role_mappings: null,
      team_mappings: null,
      ui_access_mode: null,
    };

    render(
      <QueryClientProvider client={queryClient}>
        <DeleteSSOSettingsModal isVisible={true} onCancel={vi.fn()} onSuccess={vi.fn()} />
      </QueryClientProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "Delete" }));

    await waitFor(() =>
      expect(mutateAsync).toHaveBeenCalledWith(expect.objectContaining(clearedFields), expect.anything()),
    );
  });
});
