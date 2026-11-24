import { createRef } from "react";
import { createPortal } from "react-dom";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { EndpointType } from "@/components/chat_ui/mode_endpoint_mapping";
import type { MCPToolArgumentsFormRef } from "@/components/mcp_tools/MCPToolArgumentsForm";
import { Composer, type ChecklistProps, type DraftState } from "./Composer";
import { capabilitiesFor } from "./endpointCapabilities";

const checklist: ChecklistProps = { groups: [], value: [], onValueChange: vi.fn(), isLoading: false };

const draft: DraftState = {
  text: "",
  setText: vi.fn(),
  files: [],
  attach: vi.fn(),
  removeFile: vi.fn(),
  isLoading: false,
  send: vi.fn().mockResolvedValue(undefined),
  cancel: vi.fn(),
};

const renderComposer = () =>
  render(
    <Composer
      capabilities={capabilitiesFor(EndpointType.CHAT)}
      endpoint={EndpointType.CHAT}
      targetRow={<input aria-label="Model search" />}
      draft={draft}
      directTool={null}
      toolArgsForm={createRef<MCPToolArgumentsFormRef>()}
      mcpControl={null}
      vectorStores={checklist}
      guardrails={checklist}
      codeInterpreter={{ enabled: false, onToggle: vi.fn() }}
      settingsControl={createPortal(<input aria-label="Request tags" />, document.body)}
      sendDisabled
    />,
  );

describe("Composer", () => {
  it("keeps focus in a popover field opened from the composer toolbar", async () => {
    const user = userEvent.setup();
    renderComposer();

    await user.click(screen.getByLabelText("Request tags"));

    expect(screen.getByLabelText("Request tags")).toHaveFocus();
  });

  it("focuses the message box when empty toolbar space is clicked", async () => {
    const user = userEvent.setup();
    const { container } = renderComposer();
    const toolbar = container.querySelectorAll<HTMLElement>('[data-slot="input-group-addon"]');

    await user.click(toolbar[toolbar.length - 1]);

    expect(screen.getByLabelText("Message")).toHaveFocus();
  });
});
