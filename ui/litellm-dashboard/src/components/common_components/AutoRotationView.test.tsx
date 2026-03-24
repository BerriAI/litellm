import { renderWithProviders, screen } from "../../../tests/test-utils";
import { vi, describe, it, expect } from "vitest";
import AutoRotationView from "./AutoRotationView";

// Mock heroicons since jsdom doesn't support SVG rendering
vi.mock("@heroicons/react/outline", () => ({
  RefreshIcon: (props: Record<string, unknown>) => <svg data-testid="refresh-icon" {...props} />,
  ClockIcon: (props: Record<string, unknown>) => <svg data-testid="clock-icon" {...props} />,
}));

describe("AutoRotationView", () => {
  it("should render", () => {
    renderWithProviders(<AutoRotationView />);
    expect(screen.getAllByText("Auto-Rotation").length).toBeGreaterThan(0);
  });

  it("should show Disabled badge when autoRotate is false", () => {
    renderWithProviders(<AutoRotationView autoRotate={false} />);
    expect(screen.getByText("Disabled")).toBeInTheDocument();
  });

  it("should show Enabled badge when autoRotate is true", () => {
    renderWithProviders(<AutoRotationView autoRotate={true} />);
    expect(screen.getByText("Enabled")).toBeInTheDocument();
  });

  it("should show the rotation interval when autoRotate is true", () => {
    renderWithProviders(
      <AutoRotationView autoRotate={true} rotationInterval="7d" />
    );
    expect(screen.getByText("Every 7d")).toBeInTheDocument();
  });

  it("should show 'No rotation history available' when autoRotate is enabled but no timestamps", () => {
    renderWithProviders(<AutoRotationView autoRotate={true} />);
    expect(
      screen.getByText("No rotation history available")
    ).toBeInTheDocument();
  });

  it("should show disabled message when autoRotate is off and no rotation data", () => {
    renderWithProviders(<AutoRotationView autoRotate={false} />);
    expect(
      screen.getByText("Auto-rotation is not enabled for this key")
    ).toBeInTheDocument();
  });

  it("should show last rotation timestamp when provided", () => {
    renderWithProviders(
      <AutoRotationView
        autoRotate={true}
        lastRotationAt="2024-06-15T10:30:00Z"
      />
    );
    expect(screen.getByText("Last Rotation")).toBeInTheDocument();
  });

  it("should show next scheduled rotation when nextRotationAt is provided", () => {
    renderWithProviders(
      <AutoRotationView
        autoRotate={true}
        nextRotationAt="2025-01-01T00:00:00Z"
      />
    );
    expect(screen.getByText("Next Scheduled Rotation")).toBeInTheDocument();
  });

  it("should show next scheduled rotation when keyRotationAt is provided", () => {
    renderWithProviders(
      <AutoRotationView
        autoRotate={true}
        keyRotationAt="2025-01-01T00:00:00Z"
      />
    );
    expect(screen.getByText("Next Scheduled Rotation")).toBeInTheDocument();
  });

  describe("when variant is 'inline'", () => {
    it("should render without the card wrapper", () => {
      renderWithProviders(
        <AutoRotationView variant="inline" autoRotate={false} />
      );
      expect(screen.getByText("Disabled")).toBeInTheDocument();
      // The card variant has a subtitle; inline does not
      expect(
        screen.queryByText(
          "Automatic key rotation settings and status for this key"
        )
      ).not.toBeInTheDocument();
    });
  });

  describe("when variant is 'card'", () => {
    it("should render with the card subtitle", () => {
      renderWithProviders(
        <AutoRotationView variant="card" autoRotate={false} />
      );
      expect(
        screen.getByText(
          "Automatic key rotation settings and status for this key"
        )
      ).toBeInTheDocument();
    });
  });
});
