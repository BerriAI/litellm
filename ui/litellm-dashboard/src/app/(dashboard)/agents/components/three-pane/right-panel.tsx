"use client";

/**
 * RightPanel — third pane in the session view, with Git/Terminal tabs.
 *
 * The Git tab needs both the active Run snapshot (branches, current PR)
 * and live events (commits, pr_opened). The Terminal tab only needs
 * terminal_chunk events.
 */
import { Tabs } from "antd";
import GitTab from "@/app/(dashboard)/agents/components/three-pane/git-tab";
import TerminalTab from "@/app/(dashboard)/agents/components/three-pane/terminal-tab";
import type { CloudAgentRun, CloudAgentRunEvent } from "@/types/cloud-agents";

interface RightPanelProps {
  run: CloudAgentRun | null;
  events: CloudAgentRunEvent[];
}

export default function RightPanel({ run, events }: RightPanelProps) {
  return (
    <Tabs
      defaultActiveKey="git"
      className="flex h-full flex-col"
      data-testid="right-panel"
      tabBarStyle={{ paddingLeft: 12, marginBottom: 0 }}
      items={[
        {
          key: "git",
          label: "Git",
          children: (
            <div className="h-[calc(100%-46px)]">
              <GitTab run={run} events={events} />
            </div>
          ),
        },
        {
          key: "terminal",
          label: "Terminal",
          children: (
            <div className="h-[calc(100%-46px)]">
              <TerminalTab events={events} />
            </div>
          ),
        },
      ]}
    />
  );
}
