pub const DEFAULT_HEADERS: &[(&str, &str)] = &[
    ("content-type", "application/json"),
    ("copilot-integration-id", "vscode-chat"),
    ("editor-version", "vscode/1.95.0"),
    ("editor-plugin-version", "copilot-chat/0.26.7"),
    ("user-agent", "GitHubCopilotChat/0.26.7"),
    ("x-vscode-user-agent-library-version", "electron-fetch"),
    ("anthropic-version", "2023-06-01"),
];

pub const MESSAGES_PATH: &str = "/v1/messages";
