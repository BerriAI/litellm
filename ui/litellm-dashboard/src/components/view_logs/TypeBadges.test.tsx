import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LlmBadge, McpBadge, AgentBadge, RelayBadge, RelaySourceBadge, getRelaySource } from "./TypeBadges";

describe("TypeBadges", () => {
  describe("LlmBadge", () => {
    it("should render with default 'LLM' text when no count is provided", () => {
      render(<LlmBadge />);
      expect(screen.getByText("LLM")).toBeInTheDocument();
    });

    it("should render the count when provided", () => {
      render(<LlmBadge count={5} />);
      expect(screen.getByText("5")).toBeInTheDocument();
    });

    it("should render count of 0 instead of default text", () => {
      render(<LlmBadge count={0} />);
      expect(screen.getByText("0")).toBeInTheDocument();
    });
  });

  describe("McpBadge", () => {
    it("should render with default 'MCP' text when no count is provided", () => {
      render(<McpBadge />);
      expect(screen.getByText("MCP")).toBeInTheDocument();
    });

    it("should render the count when provided", () => {
      render(<McpBadge count={3} />);
      expect(screen.getByText("3")).toBeInTheDocument();
    });
  });

  describe("AgentBadge", () => {
    it("should render with default 'Agent' text when no count is provided", () => {
      render(<AgentBadge />);
      expect(screen.getByText("Agent")).toBeInTheDocument();
    });

    it("should render the count when provided", () => {
      render(<AgentBadge count={12} />);
      expect(screen.getByText("12")).toBeInTheDocument();
    });
  });

  describe("RelayBadge", () => {
    it("should render with default 'litellm-relay' text when no count is provided", () => {
      render(<RelayBadge />);
      expect(screen.getByText("litellm-relay")).toBeInTheDocument();
    });
  });

  describe("RelaySourceBadge", () => {
    it("should render Notion source with logo text", () => {
      render(<RelaySourceBadge source="notion" />);
      expect(screen.getByRole("img", { name: "Notion logo" })).toBeInTheDocument();
      expect(screen.getByText("Notion")).toBeInTheDocument();
    });

    it("should render Codex source with logo", () => {
      render(<RelaySourceBadge source="codex" />);
      expect(screen.getByRole("img", { name: "Codex logo" })).toBeInTheDocument();
      expect(screen.getByText("Codex")).toBeInTheDocument();
    });

    it("should derive source from relay metadata app", () => {
      expect(getRelaySource({ metadata: { app: "notion" }, model: "local-ai" })).toBe("notion");
    });

    it("should derive source from relay model when metadata is missing", () => {
      expect(getRelaySource({ model: "codex-ai" })).toBe("codex");
    });
  });
});
