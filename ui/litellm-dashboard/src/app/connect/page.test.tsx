import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import ConnectPage from "./page";

interface SurfaceProps {
  accessToken: string;
  selectedServers: string[];
  onChange: (servers: string[]) => void;
}

const { mockSurface } = vi.hoisted(() => ({
  mockSurface: vi.fn((_props: SurfaceProps) => <div data-testid="connect-flow-surface" />),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "token-123" }),
}));
vi.mock("@/components/chat/ConnectFlowSurface", () => ({ default: mockSurface }));

describe("ConnectPage", () => {
  afterEach(() => {
    mockSurface.mockClear();
  });

  it("renders the gateway connect surface with the user's access token and an empty selection", () => {
    render(<ConnectPage />);
    expect(screen.getByTestId("connect-flow-surface")).toBeInTheDocument();
    expect(mockSurface.mock.calls[0][0]).toMatchObject({ accessToken: "token-123", selectedServers: [] });
  });
});
