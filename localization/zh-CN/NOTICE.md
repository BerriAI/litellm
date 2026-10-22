# 来源说明

LiteLLM 上游源码：https://github.com/BerriAI/litellm

上游许可证保存在仓库根目录 `LICENSE`，企业版目录适用 `enterprise/LICENSE.md`。本补丁在临时源码副本中替换界面文案，并提供中文地区设置与构建脚本。

中文词库初稿使用 Argos Translate English–Chinese 1.9 离线模型生成，随后进行术语统一和人工修订。模型原作者为 Jörg Tiedemann 与 Santhosh Thottingal，相关论文为 *OPUS-MT — Building open translation services for the World*，EAMT 2020，Lisbon。

模型说明标注原始 OPUS 模型采用 CC BY 4.0：https://creativecommons.org/licenses/by/4.0/

补丁包仅包含译文与构建所需源文件，翻译模型与翻译运行环境在任务结束后清理。
