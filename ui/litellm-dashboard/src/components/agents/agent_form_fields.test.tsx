import React from "react";
import { describe, it, expect } from "vitest";
import { Form } from "antd";
import { render, screen, fireEvent, act, waitFor } from "@testing-library/react";
import AgentFormFields from "./agent_form_fields";

function renderSkillsPanel(initialValues: any = {}) {
  return render(
    <Form initialValues={initialValues}>
      <AgentFormFields showAgentName={false} visiblePanels={["skills"]} />
    </Form>,
  );
}

describe("AgentFormFields (Skills > Tags)", () => {
  it("should render the Skills panel header", () => {
    renderSkillsPanel();
    expect(screen.getByText(/Skills \(Required\)/i)).toBeInTheDocument();
  });

  it("should render a real tag input control after Add Skill is clicked (LIT-3153)", async () => {
    renderSkillsPanel();
    await act(async () => {
      fireEvent.click(screen.getByText(/Skills \(Required\)/i));
    });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Add Skill/i }));
    });
    await waitFor(() =>
      expect(
        screen.getByText("Tags", { selector: "label, .ant-form-item-label *" }),
      ).toBeInTheDocument(),
    );
    const combos = screen.getAllByRole("combobox");
    expect(combos.length).toBeGreaterThan(0);
    expect(
      screen.getByText("Add tags (press Enter to confirm)"),
    ).toBeInTheDocument();
    expect(screen.queryByText(/comma-separated/i)).not.toBeInTheDocument();
  });

  it("should preserve existing tag values from initialValues as chips", async () => {
    renderSkillsPanel({
      skills: [
        {
          id: "hello_world",
          name: "Hello world",
          description: "Returns hello",
          tags: ["hello", "greeting"],
          examples: ["hi", "hello world"],
        },
      ],
    });
    await act(async () => {
      fireEvent.click(screen.getByText(/Skills \(Required\)/i));
    });
    await waitFor(() => {
      expect(screen.getByText("hello")).toBeInTheDocument();
      expect(screen.getByText("greeting")).toBeInTheDocument();
      expect(screen.getByText("hi")).toBeInTheDocument();
      expect(screen.getByText("hello world")).toBeInTheDocument();
    });
  });
});
