# LiteLLM Dashboard i18n — 技术风险清单（TECH_RISKS）

> 角色：Agent 1（`i18n-architect`）· Wave 0
> 编号 R1..Rn。每条含：风险 / 影响 / 缓解 / Owner。Owner 为「最需要跟踪/缓解」的角色，不一定是唯一。
> 关联：`I18N_TECH_DESIGN.md`、`I18N_ADR.md`、`POC_REPORT.md`。

---

## R1：静态导出下首屏语言闪烁 / 错误语言闪现

- **风险**：切 zh-CN 用户首帧可能短暂显示英文或暴露 key，体验倒退。D7 要求不得停留在「再评估」。
- **影响**：高（首屏是 G0/G1 门禁项；直接违背 §3.1）。
- **缓解**：已选定「语言就绪门禁 + 挂载后同步 `<html lang>`」（ADR-i18n-04）；就绪前不渲染业务 `t()`/`<Trans>`（ADR-i18n-03）。PoC-2/3 验证并量化门禁延时。
- **Owner**：A4（实现）→ A7（验证）。

## R2：`react-i18next` / `i18next` 与 React 19 / Next 16 的运行时兼容性未实测

- **风险**：`npm view` peer 通过，但实际打包（turbopack/next export）与 React 19 可能有边界问题（如 `useSyncExternalStore`、HMR）。
- **影响**：中（构建失败或运行时异常会阻断 Wave 1）。
- **缓解**：PoC-1 先做最小依赖安装 + `next build`；锁定 `^17`/`^26` 稳定大版本；若发现问题回退 17/26 的补丁小版本并记录。
- **Owner**：A4（先做）→ A7（复验）。

## R3：`<Trans>` 因资源未就绪而瞬态暴露占位/key

- **风险**：若个别组件绕开统一门禁，`<Trans>` 可能先渲染占位符再替换。
- **影响**：中高（违背 §3.1，且 `<Trans>` 难以用简单 `t()` 兜底）。
- **缓解**：统一顶层门禁覆盖全局子树；组件 `ready` 兜底；CI 键集合一致性检查（R9）+ 代码评审拦截绕过。
- **Owner**：A4（平台）→ A5/A6（使用）→ A7（回归）。

## R4：语言偏好 cookie/localStorage 不一致或跨跳转丢失

- **风险**：cookie 与 localStorage 双写可能不一致；`SameSite` 过严导致 SSO/整页跳转回带失败，或过宽带来安全面。
- **影响**：中（刷新/回跳语言恢复失败，D10 不满足）。
- **缓解**：以显式写入顺序为准的收敛规则（ADR-i18n-05）；`SameSite=Lax` + 生产 `Secure`；PoC-5/7 覆盖 cookie 清空/localStorage 留存、整页回跳各分支。
- **Owner**：A4 → A7。

## R5：依赖文件被多 Agent 并行写坏（锁分叉）

- **风险**：`package.json`/lock 在多 worktree 并行 `npm install/ci` 产生分叉，G1 集成冲突。
- **影响**：高（阻塞集成、浪费波次）。
- **缓解**：D12 单一写入者（默认仅 A4）；下游从 G1 基线 `npm ci` 且不写 lock；v1.4 变更摘要已固化。
- **Owner**：A4（执行）→ A0（review）。

## R6：功能 Agent 与平台共享单点（`src/i18n/**`、注册表、公共 JSON）写冲突

- **风险**：多人同时改初始化代码/注册表/大 JSON。
- **影响**：中（合并冲突、注册表被破坏导致类型/加载异常）。
- **缓解**：FILE_OWNERSHIP 规则：`src/i18n/**` 与骨架归 A4 永久独占；新增 namespace 走 A0 派单由 A4 注册；功能 Agent 只写自己 namespace。
- **Owner**：A0（调度）→ A4（执行）。

## R7：en/zh 字典键集合漂移（缺 key/多 key）

- **风险**：中文更新新增 key 但 en 未同步（或反之），导致运行期 missingKey 回退/多出 key。
- **影响**：中（D2 要求 en 为真源、zh 缺 key 回退；键集合须一致 §3.4）。
- **缓解**：CI 增加「en vs zh-CN 键集合 diff」校验（可在 `test:types` 或独立脚本，纳入 Agent 7 质量门禁）；`CustomTypeOptions` 类型约束 + 开发模式 missingKey warning。
- **Owner**：A7（CI/校验）→ A4（类型）→ 各功能 Agent（补 key）。

## R8：cookie 安全与隐私（`SameSite`、`Secure`、XSS 可读）

- **风险**：语言非秘密，但 cookie 可被 XSS 读取；`SameSite=None` 若被误用扩大攻击面。
- **影响**：低-中（隐私面小；但需遵守 CLAUDE.md「勿把 token 放 localStorage」精神，语言偏好非敏感）。
- **缓解**：语言偏好非敏感可放心；cookie `SameSite=Lax; path=/;`，生产 `Secure`；绝不用此 cookie 存令牌；仅当 SSO 第三方域确需带 cookie 时，再评估 `None` 并单独评审。
- **Owner**：A4。

## R9：硬编码文案漏网（中文环境下仍有英文句/key 可见）

- **风险**：1400+ TSX，批量改造难免遗漏；产物残留 key 串或英文句。
- **影响**：中（本地化不完整、观感差）。
- **缓解**：扫描工具只报告候选（不改写，§3.4）；PoC-9 对产物 grep 残留；纳入 v1 完成定义与 A7 回归；模块按 high-traffic 优先级（导航/登录/Models/API Keys/Usage/Cost/Budgets）。
- **Owner**：A5/A6（改造）→ A7（扫描/回归）。

## R10：静态导出 + 多语言无 SEO，`<title>`/meta 恒为英文

- **风险**：搜索引擎与分享卡片恒为英文，非本地化。
- **影响**：低（D8 明确 v1 不做多语言 SEO）。
- **缓解**：接受为 v1 边界；文档记录未来可引入 per-locale 静态 `title`（ADR-i18n-06 后果）。
- **Owner**：A0（确认范围）。

## R11：管理员全局语言设置未来引入（P1）可能改变优先级语义

- **风险**：后端未来新增 `UI settings.language` 字段，需决定「覆盖用户选择」语义，否则现有优先级迁移需返工。
- **影响**：低（v1 无此字段，ADR-i18n-07 已封闭）；中（若 v1.1 临时加入则会波及优先级链）。
- **缓解**：预留 L5 判定位（I18N_TECH_DESIGN §3.2），把「用户显式」与「探测」分开存储，便于未来叠加管理员全局层而不混淆。
- **Owner**：A1（设计预留）→ A0（接产品决策）。

## R12：中文文本在窄布局/表格溢出（文本长度不同导致 UI 破坏）

- **风险**：中文普遍比英文紧凑，但长译句或 `{{count}}` 拼接可能导致表格/按钮溢出或换行异常。
- **影响**：中（布局回归）。
- **缓解**：Wave 1 起在 Login/Navbar/表格样例做中英文并排截图比对；纳入 Agent 3 测试与 A7 回归；`whitespace-nowrap`/截断类妥善处理。
- **Owner**：A6（功能页）→ A8（集成验收）。

## R13：`.json` 资源体积随 namespace 增长、首包加大

- **风险**：所有 locale 静态打在一起可能增大 bundle；`turbopack` 打包 json 的方式需确认。
- **影响**：低-中（Dashboard 体量下影响有限；但静态导出无 CDN 分片语言包，加载即全量）。
- **缓解**：PoC-1 检查产物 json 是否被正确 tree-shake/分 chunk；v1 语言仅 2 个，体积可控；若未来多语言再评估代码分割。
- **Owner**：A4 → A8。

---

## 风险热力（供 A0 优先级参考）

| 等级 | 风险 |
|---|---|
| 高 | R1（首屏）、R5（锁分叉） |
| 中高 | R3（`<Trans>`）、R9（硬编码遗漏） |
| 中 | R2、R4、R6、R7、R12 |
| 低 | R8、R10、R11、R13 |
