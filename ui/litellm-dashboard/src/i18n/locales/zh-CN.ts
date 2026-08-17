import type en from "./en";

type TranslationShape<T> = {
  [K in keyof T]: T[K] extends string ? string : TranslationShape<T[K]>;
};

const zhCN = {
  common: {
    language: "语言",
    english: "English",
    simplifiedChinese: "简体中文",
    docs: "文档",
  },
  navigation: {
    groups: {
      aiGateway: "AI 网关",
      observability: "可观测性",
      accessControl: "访问控制",
      developerTools: "开发者工具",
      settings: "设置",
    },
    items: {
      "api-keys": "虚拟密钥",
      "llm-playground": "模型调试",
      models: "模型与端点",
      agentic: "智能体",
      agents: "智能体",
      workflows: "工作流运行记录",
      memory: "记忆",
      "mcp-servers": "MCP 服务器",
      skills: "技能",
      guardrails: "安全护栏",
      policies: "策略",
      tools: "工具",
      "search-tools": "搜索工具",
      "vector-stores": "向量存储",
      "tool-policies": "工具策略",
      new_usage: "用量",
      "cost-optimization": "成本优化",
      logs: "日志",
      "guardrails-monitor": "护栏监控",
      teams: "团队",
      projects: "项目",
      users: "内部用户",
      organizations: "组织",
      "access-groups": "访问组",
      budgets: "预算",
      api_ref: "API 参考",
      "model-hub-table": "AI 中心",
      "learning-resources": "学习资源",
      caching: "响应缓存",
      experimental: "实验功能",
      prompts: "提示词",
      "transform-request": "API 调试",
      "tag-management": "标签管理",
      usage: "旧版用量",
      settings: "设置",
      "router-settings": "路由设置",
      "logging-and-alerts": "日志与告警",
      "admin-panel": "管理设置",
      "cost-tracking": "成本跟踪",
      "ui-theme": "界面主题",
    },
    expandSidebar: "展开侧边栏",
    collapseSidebar: "收起侧边栏",
    home: "LiteLLM 首页",
  },
  viewSwitcher: {
    aiGateway: "AI 网关",
    chat: "对话",
    enableChatHint: "管理员可在设置中启用",
  },
  login: {
    title: "登录",
    subtitle: "访问 LiteLLM 管理后台。",
    disabled: {
      title: "管理后台已禁用",
      description: "管理员已禁用管理后台。如需重新启用，请更新以下环境变量：",
    },
    defaultCredentials: {
      title: "默认凭据",
      description:
        "默认用户名为 <username>admin</username>，密码为 LiteLLM Proxy 中设置的 <masterKey>MASTER_KEY</masterKey>。",
      help: "需要配置界面登录凭据或 SSO？<docs>查看文档</docs>。",
    },
    worker: "工作节点",
    workerPlaceholder: "选择要连接的工作节点",
    username: "用户名",
    usernameRequired: "请输入用户名",
    usernamePlaceholder: "输入用户名",
    password: "密码",
    passwordRequired: "请输入密码",
    passwordPlaceholder: "输入密码",
    submit: "登录",
    submitting: "正在登录……",
    sso: "使用 SSO 登录",
    ssoNotConfigured: "请先配置 SSO。",
    ssoEnabled:
      "单点登录（SSO）已启用。LiteLLM 加载此页面时将不再自动跳转到 SSO 登录流程。如需恢复自动跳转，请在环境配置中设置 <setting>AUTO_REDIRECT_UI_LOGIN_TO_SSO=true</setting>。",
  },
} satisfies TranslationShape<typeof en>;

export default zhCN;
