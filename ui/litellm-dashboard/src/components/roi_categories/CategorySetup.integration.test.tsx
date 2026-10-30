import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { CategorySetup } from "./CategorySetup";
import type { Settings } from "./data";

const post = vi.hoisted(() => vi.fn());
vi.mock("@/components/networking", () => ({ apiClient: { post } }));

beforeEach(() => {
  post.mockReset();
});

describe("ROI category setup", () => {
  it("includes extracted business context in discovery setup", async () => {
    post.mockResolvedValueOnce({ name: "context.md", text: "Support resolves tickets" }).mockResolvedValueOnce({});
    const saved = vi.fn();
    render(<CategorySetup token="fixture" settings={null} onSaved={saved} />);
    fireEvent.change(screen.getByLabelText("Discovery model"), { target: { value: "discover" } });
    fireEvent.change(screen.getByLabelText("Jev model"), { target: { value: "jev" } });
    fireEvent.change(screen.getByLabelText("Business context files"), {
      target: { files: [new File(["Support resolves tickets"], "context.md", { type: "text/markdown" })] },
    });
    expect(await screen.findByText("context.md · Ready")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Start discovery" }));
    await waitFor(() =>
      expect(post).toHaveBeenLastCalledWith("/roi/categories/setup", {
        accessToken: "fixture",
        body: {
          sample_size: 100,
          discovery_model: "discover",
          classifier_model: "jev",
          team_id: "",
          context: "Support resolves tickets",
        },
      }),
    );
    await waitFor(() => expect(saved).toHaveBeenCalledOnce());
  });

  it("activates the user's edited category definitions", async () => {
    post.mockResolvedValue({});
    const settings: Settings = {
      version: "v1",
      phase: "review",
      categories: [{ id: "support", name: "Support", description: "Resolve tickets" }],
      sample_size: 5,
      sampled: 5,
      error: null,
      discovery_model: "discover",
      classifier_model: "jev",
      team_id: "",
    };
    render(<CategorySetup token="fixture" settings={settings} onSaved={() => undefined} />);
    fireEvent.change(screen.getByLabelText("Category 1 name"), { target: { value: "Customer support" } });
    fireEvent.click(screen.getByRole("button", { name: "Activate categories" }));
    await waitFor(() =>
      expect(post).toHaveBeenCalledWith("/roi/categories/activate", {
        accessToken: "fixture",
        body: { categories: [{ id: "support", name: "Customer support", description: "Resolve tickets" }] },
      }),
    );
  });

  it("keeps failed setup visible instead of reporting success", async () => {
    post.mockRejectedValue(new Error("ClickHouse unavailable"));
    const saved = vi.fn();
    render(<CategorySetup token="fixture" settings={null} onSaved={saved} />);
    fireEvent.change(screen.getByLabelText("Discovery model"), { target: { value: "discover" } });
    fireEvent.change(screen.getByLabelText("Jev model"), { target: { value: "jev" } });
    fireEvent.click(screen.getByRole("button", { name: "Start discovery" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("ClickHouse unavailable");
    expect(saved).not.toHaveBeenCalled();
  });
});
