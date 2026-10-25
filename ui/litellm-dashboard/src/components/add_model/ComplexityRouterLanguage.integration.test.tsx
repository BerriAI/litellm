import { fireEvent, renderWithProviders, screen } from "../../../tests/test-utils";
import { LanguageProvider, useLanguage } from "@/contexts/LanguageContext";
import { describe, expect, it, vi } from "vitest";
import ComplexityRouterAdvancedSections from "./ComplexityRouterAdvancedSections";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";
import { DEFAULT_AUTO_ROUTER_COMPRESSION } from "./buildAutoRouterCompression";
import StallEscalationConfig from "./StallEscalationConfig";
import { Restricted } from "./TierRestrictions";
import { CUSTOM_TIER_RESTRICTIONS } from "./tier_rows";
import CacheAwareRoutingConfig from "./CacheAwareRoutingConfig";
import HousekeepingRoutingControls from "./HousekeepingRoutingControls";
import NonReasoningTierToggle from "./NonReasoningTierToggle";
import PlanModeOverrideControls from "./PlanModeOverrideControls";
import TierConfigIntro from "./TierConfigIntro";

function SwitchLanguage() {
  const { setPreference } = useLanguage();
  return <button onClick={() => setPreference("en")}>English</button>;
}

const value: ComplexityRouterConfigValue = {
  tiers: { SIMPLE: ["Home"], MEDIUM: ["Home"], COMPLEX: ["Home"], REASONING: ["Home"] },
  classifier_type: "heuristic",
};

describe("自动路由语言切换", () => {
  it("条件限制和附加说明随语言切换，不修改路由配置", () => {
    const onChange = vi.fn();
    renderWithProviders(
      <LanguageProvider defaultLanguage="zh-CN" userId="router-conditions" storageScope="router-conditions">
        <SwitchLanguage />
        <CacheAwareRoutingConfig value={{ ...value, adaptive: true, cache_aware_routing: true }} onChange={onChange} />
        <HousekeepingRoutingControls
          value={{ ...value, route_housekeeping_to_cheapest_tier: false }}
          onChange={onChange}
        />
        <NonReasoningTierToggle value={value} onChange={onChange} available={false} />
        <PlanModeOverrideControls value={value} onChange={onChange} planModeTierOptions={[]} />
        <TierConfigIntro value={{ ...value, classifier_type: "llm" }} />
      </LanguageProvider>,
    );
    expect(screen.getByRole("status")).toHaveTextContent("当前已跳过缓存感知路由");
    expect(screen.getByRole("status")).toHaveTextContent("关闭自适应路由以启用缓存感知路由");
    expect(screen.getByText(/启用日常任务路由后这些设置才会生效/)).toBeInTheDocument();
    expect(screen.getByText(/需要使用 LLM 或 Jev 分类方式/)).toBeInTheDocument();
    expect(screen.getByText(/向分层添加模型后才能启用此功能/)).toBeInTheDocument();
    expect(screen.getByText(/显示名称用于控制台和支出日志/)).toBeInTheDocument();
    expect(screen.getByText(/评判模型也会使用这些名称对请求进行分类/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "English" }));
    expect(screen.getByRole("status")).toHaveTextContent("Turn off Adaptive Routing");
    expect(onChange).not.toHaveBeenCalled();
  });

  it("切换语言后保留已展开的高级设置分组", () => {
    renderWithProviders(
      <LanguageProvider defaultLanguage="zh-CN" userId="router-language" storageScope="router-language-groups">
        <SwitchLanguage />
        <ComplexityRouterAdvancedSections
          value={value}
          onChange={vi.fn()}
          forecast={false}
          modelOptions={[]}
          classifierEffortOptionsByModel={{}}
          showValidationErrors={false}
          planModeTierOptions={[]}
          keywordTierRules={[]}
          semanticMatchingEnabled={false}
          onEmbeddingModelChange={vi.fn()}
          matchThreshold={0.5}
          onMatchThresholdChange={vi.fn()}
          escalationKeywords={[]}
          autoRouterCompression={DEFAULT_AUTO_ROUTER_COMPRESSION}
          modelInfo={[]}
          tierRows={[]}
          customTierSet={undefined}
        />
      </LanguageProvider>,
    );
    const trigger = screen.getByRole("button", { name: "路由规则与恢复" });
    fireEvent.click(trigger);
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    fireEvent.click(screen.getByRole("button", { name: "English" }));
    expect(screen.getByRole("button", { name: "Routing rules and recovery" })).toHaveAttribute("aria-expanded", "true");
  });

  it("翻译动态限制说明并保留内部分类频率值", () => {
    const onChange = vi.fn();
    renderWithProviders(
      <LanguageProvider defaultLanguage="zh-CN" userId="router-language" storageScope="router-language-reasons">
        <SwitchLanguage />
        <Restricted by={CUSTOM_TIER_RESTRICTIONS.heuristicClassifier}>
          <span>Home</span>
        </Restricted>
        <StallEscalationConfig value={{ ...value, session_affinity: true }} onChange={onChange} />
        <StallEscalationConfig
          value={{ ...value, session_affinity: false, classification_mode: "user_turn" }}
          onChange={onChange}
        />
      </LanguageProvider>,
    );
    expect(screen.getByText(/启发式评分器仅生成内置层级/)).toBeInTheDocument();
    expect(screen.getByText(/每个会话仅评分一次会复用已选模型/)).toBeInTheDocument();
    expect(screen.getByText(/仅对新用户消息评分会跳过发生停滞的工具调用轮次/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "English" }));
    expect(screen.getByText(/The heuristic scorer only produces the built-in tiers/)).toBeInTheDocument();
    expect(onChange).not.toHaveBeenCalled();
  });
});
