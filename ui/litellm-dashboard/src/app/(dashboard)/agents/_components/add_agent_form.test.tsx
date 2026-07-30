import React from "react";
import { render, screen, fireEvent, within } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import AddAgentForm from "./add_agent_form";
import * as networking from "@/components/networking";
import type { AgentCreateInfo } from "@/components/networking";

vi.mock("@/components/networking", () => ({
  createAgentCall: vi.fn(),
  getAgentCreateMetadata: vi.fn(),
  getAgentsList: vi.fn(),
  keyCreateForAgentCall: vi.fn(),
  keyListCall: vi.fn(),
  keyUpdateCall: vi.fn(),
  modelAvailableCall: vi.fn(),
}));

vi.mock("./agent_card_discovery", () => ({
  default: () => <div data-testid="agent-card-discovery" />,
}));

vi.mock("./agent_form_fields", () => ({
  default: () => <div data-testid="agent-form-fields" />,
}));

const a2aInfo: AgentCreateInfo = {
  agent_type: "a2a",
  agent_type_display_name: "A2A Agent",
  description: "Agent-to-agent protocol",
  logo_url: "/ui/assets/logos/a2a_agent.png",
  credential_fields: [],
  use_a2a_form_fields: true,
};

const renderForm = () =>
  render(<AddAgentForm visible={true} onClose={vi.fn()} accessToken="test-token" onSuccess={vi.fn()} />);

describe("AddAgentForm logos", () => {
  beforeEach(() => {
    vi.mocked(networking.getAgentCreateMetadata).mockReset().mockResolvedValue([a2aInfo]);
    vi.mocked(networking.getAgentsList).mockReset().mockResolvedValue({ agents: [] });
    vi.mocked(networking.keyListCall).mockReset().mockResolvedValue({ keys: [] });
    vi.mocked(networking.modelAvailableCall).mockReset().mockResolvedValue({ data: [] });
  });

  it("renders the modal title and agent type selection logos as images from logo_url", async () => {
    renderForm();

    const titleLogo = await screen.findByAltText("Agent logo");
    expect(titleLogo).toBeInstanceOf(HTMLImageElement);
    expect(titleLogo).toHaveAttribute("src", expect.stringContaining("assets/logos/a2a_agent.png"));

    const selectionLogo = await screen.findByAltText("A2A Agent logo");
    expect(selectionLogo).toBeInstanceOf(HTMLImageElement);
    expect(selectionLogo).toHaveAttribute("src", expect.stringContaining("assets/logos/a2a_agent.png"));
  });

  it("renders the option logo when the agent type dropdown is opened", async () => {
    renderForm();

    await screen.findByAltText("A2A Agent logo");
    fireEvent.mouseDown(screen.getByRole("combobox"));

    const optionLogos = await screen.findAllByAltText("A2A Agent logo");
    expect(optionLogos.length).toBeGreaterThanOrEqual(2);
    optionLogos.forEach((img) => {
      expect(img).toHaveAttribute("src", expect.stringContaining("assets/logos/a2a_agent.png"));
    });
  });

  it("swaps a failing logo for a letter avatar and warns with the url", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    renderForm();

    const titleLogo = await screen.findByAltText("Agent logo");
    const header = screen.getByText("Add New Agent").parentElement!;
    fireEvent.error(titleLogo);

    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining("assets/logos/a2a_agent.png"));
    expect(screen.queryByAltText("Agent logo")).not.toBeInTheDocument();
    expect(within(header).getByText("A")).toBeInTheDocument();

    const selectionLogo = screen.getByAltText("A2A Agent logo");
    fireEvent.error(selectionLogo);
    expect(screen.queryByAltText("A2A Agent logo")).not.toBeInTheDocument();
    expect(warnSpy).toHaveBeenCalledTimes(2);
    warnSpy.mockRestore();
  });
});
