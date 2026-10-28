import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import OpenAPIQuickPicker, { type OpenAPIRegistryEntry } from "./OpenAPIQuickPicker";
import { fetchOpenAPIRegistry } from "@/components/networking";

vi.mock("@/components/networking", () => ({
  fetchOpenAPIRegistry: vi.fn(),
}));

const stripe: OpenAPIRegistryEntry = {
  name: "stripe",
  title: "Stripe",
  description: "Payments API",
  icon_url: "https://cdn.example.com/stripe.svg",
  spec_url: "https://example.com/stripe.json",
};

const github: OpenAPIRegistryEntry = {
  name: "github",
  title: "GitHub",
  description: "Code hosting API",
  icon_url: "https://cdn.example.com/github.svg",
  spec_url: "https://example.com/github.json",
};

describe("OpenAPIQuickPicker", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders one selectable entry per registry API", async () => {
    vi.mocked(fetchOpenAPIRegistry).mockResolvedValue({ apis: [stripe, github] });

    render(<OpenAPIQuickPicker accessToken="tok" selectedName={null} onSelect={vi.fn()} />);

    expect(await screen.findByRole("button", { name: /Stripe/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /GitHub/ })).toBeInTheDocument();
    expect(screen.getByText("Popular APIs")).toBeInTheDocument();
  });

  it("passes the whole registry entry to onSelect when one is clicked", async () => {
    vi.mocked(fetchOpenAPIRegistry).mockResolvedValue({ apis: [stripe, github] });
    const onSelect = vi.fn();

    render(<OpenAPIQuickPicker accessToken="tok" selectedName={null} onSelect={onSelect} />);
    await userEvent.click(await screen.findByRole("button", { name: /Stripe/ }));

    expect(onSelect).toHaveBeenCalledWith(stripe);
  });

  it("renders nothing when the registry is empty", async () => {
    vi.mocked(fetchOpenAPIRegistry).mockResolvedValue({ apis: [] });

    const { container } = render(<OpenAPIQuickPicker accessToken="tok" selectedName={null} onSelect={vi.fn()} />);

    await waitFor(() => expect(container).toBeEmptyDOMElement());
  });

  it("renders nothing when the registry fetch fails", async () => {
    vi.mocked(fetchOpenAPIRegistry).mockRejectedValue(new Error("boom"));

    const { container } = render(<OpenAPIQuickPicker accessToken="tok" selectedName={null} onSelect={vi.fn()} />);

    await waitFor(() => expect(container).toBeEmptyDOMElement());
  });

  it("does not fetch without an access token", () => {
    render(<OpenAPIQuickPicker accessToken={null} selectedName={null} onSelect={vi.fn()} />);

    expect(fetchOpenAPIRegistry).not.toHaveBeenCalled();
  });

  it("falls back to a letter avatar when the icon fails to load", async () => {
    vi.mocked(fetchOpenAPIRegistry).mockResolvedValue({ apis: [stripe] });

    render(<OpenAPIQuickPicker accessToken="tok" selectedName={null} onSelect={vi.fn()} />);

    fireEvent.error(await screen.findByAltText("Stripe"));

    await waitFor(() => expect(screen.queryByAltText("Stripe")).not.toBeInTheDocument());
    expect(screen.getByText("S")).toBeInTheDocument();
  });
});
