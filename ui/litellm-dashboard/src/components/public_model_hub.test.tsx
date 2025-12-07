import { describe, it, expect, vi, beforeAll, beforeEach } from "vitest";
import { render } from "@testing-library/react";
import PublicModelHub from "./public_model_hub";

vi.mock("next/navigation", () => ({
  useRouter: vi.fn(() => ({
    replace: vi.fn(),
    push: vi.fn(),
    refresh: vi.fn(),
  })),
}));

vi.mock("./networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./networking")>();
  return {
    ...actual,
    modelHubPublicModelsCall: vi.fn().mockResolvedValue([]),
    getPublicModelHubInfo: vi.fn().mockResolvedValue({
      docs_title: "LiteLLM Gateway",
      custom_docs_description: null,
      litellm_version: "1.0.0",
      useful_links: {},
    }),
    agentHubPublicModelsCall: vi.fn().mockResolvedValue([]),
    mcpHubPublicServersCall: vi.fn().mockResolvedValue([]),
    getUiConfig: vi.fn().mockResolvedValue({}),
  };
});

beforeAll(() => {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  });
});

beforeEach(() => {
  Storage.prototype.getItem = vi.fn(() => "false");
  Storage.prototype.setItem = vi.fn();
  Object.defineProperty(window, "location", {
    writable: true,
    value: {
      pathname: "/",
      origin: "http://localhost:3000",
    },
  });
});

describe("PublicModelHub", () => {
  it("renders", () => {
    const { container } = render(<PublicModelHub />);
    expect(container).toBeInTheDocument();
  });
});
