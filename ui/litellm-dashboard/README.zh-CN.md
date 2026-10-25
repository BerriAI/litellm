# 管理后台简体中文支持

## 功能范围

本批为社区功能和共享控件提供 English 与简体中文切换，沿用现有界面结构、路由和 API 契约，不新增国际化依赖或数据表

覆盖导航、首页、模型与端点、虚拟密钥、请求日志与详情、用量与费用、团队、用户、访问组、预算、智能体、MCP 服务、工具、向量存储、护栏、策略、调试工作台和常用管理设置。登录、邀请和密码重置页使用同一语言上下文。未纳入本批的页面和功能区域保持上游原样

只翻译应用提供的固定标签、说明、提示、空状态和本地校验文案。社区文章、模型回复、用户输入、模型名、密钥别名、资源标识、业务标签、插件名称、API 字段、枚举值、代码示例和服务端动态文案保持原文。切换语言不应改写业务值、清空草稿或重新发起业务请求

## 语言选择与默认值

桌面端在顶部“界面语言”选择器切换，窄屏端在顶部“更多选项”中操作，选择后立即生效。“跟随全站默认”取消个人覆盖

语言优先级依次为个人选择、全站默认和英文。个人选择保存在当前浏览器，按代理地址和登录账户隔离，不跨设备同步。语言存储不包含令牌或 API 密钥。浏览器禁止本地存储时仍可当次切换，并提示无法保存偏好

管理员在“设置 / 管理设置 / 界面设置 / 全站默认语言”中选择 `en` 或 `zh-CN`。保存成功后当前页面立即更新，其他已打开的后台页面每 60 秒重新读取设置，已有个人覆盖不受影响

在界面中持久化默认语言需要连接数据库并启用 `STORE_MODEL_IN_DB=True`。也可以在配置文件中指定

```yaml
general_settings:
  default_ui_language: zh-CN
```

配置文件值优先于数据库值，界面显示“由配置文件管理”并只读。恢复为 `en` 后，没有个人覆盖的用户使用英文。仅部署新版前端时仍可个人切换，但设置全站默认语言需要对应后端支持

## 接口与权限

复用 `GET /get/ui_settings` 和 `PATCH /update/ui_settings`，不新增端点。`values.default_ui_language` 返回当前生效值，`source.default_ui_language` 标明来源，`field_schema.properties.default_ui_language` 描述字段。现有 OpenAPI 文档的 UI Settings 分类展示端点，前端固定提供两种支持语言，后端用 `Literal` 校验

服务端沿用现有认证、角色检查、存储和配置修改流程。只有具有写权限的代理管理员可保存设置，只读管理员返回 403。普通用户在通用路由或端点权限检查处被拒绝。非法语言值返回 422 且不落库，部分更新保留其他设置。前端加载和保存失败显示固定提示与错误码，不展示内部堆栈

## 扩展翻译

静态文案通过 `useLanguage().t()` 读取，英文原文同时作为词典键和回退文案。`src/i18n/zh-CN.json` 与 `src/i18n/messages/` 下的词典在模块初始化时合并。动态值用命名参数插入，不把用户内容作为翻译键

```tsx
const { t } = useLanguage();
return <span>{t("Usage, last {days} days", { days: 7 })}</span>;
```

表格列和纯函数显式接收翻译函数，默认保留英文，不从 DOM 或全局变量读取语言。异步流程的本地通知使用 `TranslatedMessage` 在显示时读取语言，避免保存提交时的旧语言。`toast.fromError` 只翻译其生成的固定标题，服务端错误详情不自动查词典

本地校验保存固定原文键，在渲染时翻译已有错误。语言变化不应重置表单，不应成为业务加载 effect 或 OAuth 流程的重启条件。新增翻译同时提供独立中文预期，并测试切换后的草稿、请求次数和提交值

## 验证方式与边界

定向测试覆盖语言回退、账户与代理隔离、个人覆盖、全站默认设置、只读控件、配置文件优先级，以及相关业务表单和通知。按修改路径传入具体测试文件，不无路径运行全量 Vitest，也不反复执行生产构建

```powershell
npm run test:unit -- src/i18n/language.test.ts src/i18n/managementSchemasLanguage.test.ts
npm run test:integration -- src/contexts/LanguageContext.integration.test.tsx src/components/Settings/AdminSettings/UISettings/UISettings.integration.test.tsx
npm run typecheck
```

隔离验收使用仅监听本机的 PostgreSQL 和测试后端、合成管理员及普通用户。已验证真实 HTTP 权限与非法值拒绝、默认语言持久化、服务重启后设置保持、个人覆盖与跟随默认、只读管理员的语言控件，以及模型、密钥、请求日志和窄屏页面。日志样本是合成数据，不代表已验收模型调用或日志采集链路

本地测试、真实 HTTP 与浏览器检查分别作为证据，不等同于远端 CI、维护者审查或生产验收。部署前仍需在自己的配置和受支持浏览器中检查布局、登录方式和常用业务流程

模型列表的[改造前截图](docs/images/language-models-before.jpg)与[中文截图](docs/images/language-models-after.jpg)使用同一 URL 和合成数据。红框标出固定文案及新增语言入口，模型名称 `Home` 保持原文
