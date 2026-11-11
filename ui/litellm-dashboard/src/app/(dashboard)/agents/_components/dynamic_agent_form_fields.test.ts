import { describe, expect, it } from "vitest";
import type { AgentCreateInfo } from "@/components/networking";
import type { AgentFormValues } from "./AgentFormKit";
import { AGENT_FORM_CONFIG } from "./agent_config";
import { buildDynamicAgentData, unmountedDynamicFieldNames } from "./dynamic_agent_form_fields";
import {
  EMPTY_KILL_SWITCH_FORM,
  KILL_SWITCH_PANEL_KEY,
  type KillSwitchConfig,
  type KillSwitchFormValue,
} from "./kill_switch_config";

const langgraphInfo: AgentCreateInfo = {
  agent_type: "langgraph",
  agent_type_display_name: "LangGraph",
  model_template: "langgraph/{assistant_id}",
  credential_fields: [{ key: "assistant_id", label: "Assistant ID", required: true, include_in_litellm_params: false }],
};

const killSwitchForm: KillSwitchFormValue = {
  ...EMPTY_KILL_SWITCH_FORM,
  url: "https://ops.example.com/kill",
  method: "DELETE",
  headers: [{ key: "X-Env", value: "prod" }],
  auth_type: "bearer",
  auth_token: "tok",
};

const baseValues: AgentFormValues = { agent_name: "lg-agent", assistant_id: "asst_1" };

describe("buildDynamicAgentData kill switch", () => {
  it("serializes the kill switch section into the payload", () => {
    const payload = buildDynamicAgentData({ ...baseValues, kill_switch: killSwitchForm }, langgraphInfo);

    const expected: KillSwitchConfig = {
      url: "https://ops.example.com/kill",
      method: "DELETE",
      headers: { "X-Env": "prod" },
      query_params: {},
      body: null,
      auth: { type: "bearer", token: "tok" },
    };
    expect(payload.kill_switch).toEqual(expected);
    expect(payload.litellm_params).toMatchObject({ model: "langgraph/asst_1" });
  });

  it("leaves kill_switch off the payload when the section was never mounted", () => {
    expect("kill_switch" in buildDynamicAgentData(baseValues, langgraphInfo)).toBe(false);
  });

  it("clears a stored kill switch when the URL is blanked, but not on a fresh create", () => {
    const blanked: AgentFormValues = { ...baseValues, kill_switch: { ...EMPTY_KILL_SWITCH_FORM } };

    expect("kill_switch" in buildDynamicAgentData(blanked, langgraphInfo)).toBe(false);
    const stored = { kill_switch: { url: "https://old.example", method: "POST" as const } };
    expect(buildDynamicAgentData(blanked, langgraphInfo, stored).kill_switch).toBeNull();
  });
});

describe("unmountedDynamicFieldNames", () => {
  it("drops kill_switch from the submit only while its panel is unmounted", () => {
    expect(unmountedDynamicFieldNames([AGENT_FORM_CONFIG.cost.key])).toEqual(["kill_switch"]);
    expect(unmountedDynamicFieldNames([AGENT_FORM_CONFIG.cost.key, KILL_SWITCH_PANEL_KEY])).toEqual([]);
  });
});
