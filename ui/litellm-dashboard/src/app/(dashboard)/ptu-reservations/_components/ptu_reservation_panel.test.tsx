import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: vi.fn().mockReturnValue({
    accessToken: "test-token",
    userId: "test-user",
    userRole: "proxy_admin",
  }),
}));

vi.mock("@/app/(dashboard)/hooks/ptuReservations/useIsPtuCostAttributionEnabled", () => ({
  useIsPtuCostAttributionEnabled: vi.fn().mockReturnValue({ enabled: true, isLoading: false }),
}));

vi.mock("@/app/(dashboard)/hooks/ptuReservations/usePtuReservations", () => ({
  usePtuReservations: vi.fn().mockReturnValue({ data: [], isLoading: false }),
  useClosePtuReservation: vi.fn().mockReturnValue({ mutateAsync: vi.fn(), isPending: false }),
  useCreatePtuReservation: vi.fn().mockReturnValue({ mutateAsync: vi.fn(), isPending: false }),
}));

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useIsPtuCostAttributionEnabled } from "@/app/(dashboard)/hooks/ptuReservations/useIsPtuCostAttributionEnabled";
import { usePtuReservations } from "@/app/(dashboard)/hooks/ptuReservations/usePtuReservations";
import PtuReservationPanel from "./ptu_reservation_panel";

function renderWithProviders(ui: React.ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>);
}

const activeReservation = {
  id: "res_1",
  team_id: "team_x",
  model: "gpt-4",
  cost_source: "manual" as const,
  ptu_count: 1,
  cost_per_ptu: 200,
  azure_resource_id: null,
  effective_from: "2020-01-01T00:00:00Z",
  effective_to: null,
  created_by: "admin",
  created_at: "2020-01-01T00:00:00Z",
  updated_by: "admin",
  updated_at: "2020-01-01T00:00:00Z",
};

describe("PtuReservationPanel", () => {
  afterEach(() => {
    vi.clearAllMocks();
    vi.mocked(useAuthorized).mockReturnValue({
      accessToken: "test-token",
      userId: "test-user",
      userRole: "proxy_admin",
    } as any);
  });

  it("shows disabled state when feature flag is off", () => {
    vi.mocked(useIsPtuCostAttributionEnabled).mockReturnValue({ enabled: false, isLoading: false });

    renderWithProviders(<PtuReservationPanel accessToken="test-token" />);

    expect(screen.getByText(/PTU cost attribution is disabled/i)).toBeInTheDocument();
    expect(screen.queryByText(/Create Reservation/i)).not.toBeInTheDocument();
  });

  it("shows loading state while flag is being fetched", () => {
    vi.mocked(useIsPtuCostAttributionEnabled).mockReturnValue({ enabled: false, isLoading: true });

    renderWithProviders(<PtuReservationPanel accessToken="test-token" />);

    expect(screen.getByText(/Loading/i)).toBeInTheDocument();
  });

  it("renders empty state when no reservations exist", async () => {
    vi.mocked(useIsPtuCostAttributionEnabled).mockReturnValue({ enabled: true, isLoading: false });
    vi.mocked(usePtuReservations).mockReturnValue({ data: [], isLoading: false } as any);

    renderWithProviders(<PtuReservationPanel accessToken="test-token" />);

    await waitFor(() => {
      expect(screen.getByText(/No reservations yet/i)).toBeInTheDocument();
    });
  });

  it("renders reservations in the table with correct fields", async () => {
    vi.mocked(useIsPtuCostAttributionEnabled).mockReturnValue({ enabled: true, isLoading: false });
    vi.mocked(usePtuReservations).mockReturnValue({
      data: [activeReservation],
      isLoading: false,
    } as any);

    renderWithProviders(<PtuReservationPanel accessToken="test-token" />);

    await waitFor(() => {
      expect(screen.getByText("team_x")).toBeInTheDocument();
      expect(screen.getByText("gpt-4")).toBeInTheDocument();
    });
    // status badge
    expect(screen.getByText("active")).toBeInTheDocument();
  });

  it("opens the create modal from the button", async () => {
    vi.mocked(useIsPtuCostAttributionEnabled).mockReturnValue({ enabled: true, isLoading: false });
    vi.mocked(usePtuReservations).mockReturnValue({ data: [], isLoading: false } as any);

    renderWithProviders(<PtuReservationPanel accessToken="test-token" />);

    // Modal must be closed initially — no "Team ID" form label visible
    expect(screen.queryByLabelText(/Team ID/i)).not.toBeInTheDocument();

    fireEvent.click(screen.getByText("+ Create Reservation"));
    await waitFor(() => {
      expect(screen.getByLabelText(/Team ID/i)).toBeInTheDocument();
    });
  });

  it("shows a friendly denial for non-admin roles", async () => {
    vi.mocked(useAuthorized).mockReturnValue({
      accessToken: "t",
      userId: "u",
      userRole: "internal_user",
    } as any);
    vi.mocked(useIsPtuCostAttributionEnabled).mockReturnValue({ enabled: true, isLoading: false });
    vi.mocked(usePtuReservations).mockReturnValue({ data: [], isLoading: false } as any);

    renderWithProviders(<PtuReservationPanel accessToken="t" />);

    await waitFor(() => {
      expect(screen.getByText(/proxy-admin access/i)).toBeInTheDocument();
    });
    expect(screen.queryByText("+ Create Reservation")).not.toBeInTheDocument();
  });
});
