# Cursor API Streaming Format

## Overview

Cursor uses **Protocol Buffers (protobuf)** over gRPC for its streaming API, with the API server at `https://api2.cursor.sh`. The main message types are defined in the `aiserver.v1` package.

## API Endpoint
```
https://api2.cursor.sh
```

## Request Format

### Main Request: `StreamUnifiedChatRequest`

The primary request message for streaming chat is `aiserver.v1.StreamUnifiedChatRequest`:

```protobuf
message StreamUnifiedChatRequest {
  // Core conversation data
  repeated ConversationMessage conversation = 1;
  repeated ConversationMessageHeader full_conversation_headers_only = 30;
  
  // Model configuration
  ModelDetails model_details = 5;
  
  // Context options
  bool allow_long_file_scan = 2;
  ExplicitContext explicit_context = 3;
  CurrentFileInfo current_file = 15;
  RecentEdits recent_edits = 16;
  
  // Code context
  repeated LinterErrors linter_errors = 6;
  repeated LinterErrors multi_file_linter_errors = 14;
  repeated ChatQuote quotes = 21;
  repeated RankedContext additional_ranked_context = 20;
  
  // Mode configuration
  bool is_chat = 22;                              // true for chat mode
  bool is_agentic = 27;                           // true for agent mode
  string conversation_id = 23;
  UnifiedMode unified_mode = 46;                  // CHAT=1, AGENT=2, EDIT=3, CUSTOM=4, PLAN=5, DEBUG=6
  string unified_mode_name = 54;
  
  // Repository info
  RepositoryInfo repository_info = 24;
  
  // Tool support
  repeated ClientSideToolV2 supported_tools = 29;  // List of tools the client supports
  repeated ClientSideToolV2 tools_requiring_accepted_return = 47;
  repeated MCPParams.Tool mcp_tools = 34;          // MCP tools available
  bool should_disable_tools = 48;
  
  // Agent/Agentic features
  bool enable_yolo_mode = 31;                      // Auto-run mode
  string yolo_prompt = 32;
  bool mode_uses_auto_apply = 53;
  
  // Thinking/reasoning  
  ThinkingLevel thinking_level = 49;              // MEDIUM=1, HIGH=2
  bool should_cache = 13;
  
  // Documentation and web
  repeated string documentation_identifiers = 7;
  string use_web = 8;
  repeated ComposerExternalLink external_links = 9;
  
  // Background/headless mode
  bool is_headless = 45;
  bool is_background_composer = 68;
  string background_composer_id = 55;
  
  // Cursor rules
  bool uses_rules = 51;
  bool use_generate_rules_prompt = 56;
  
  // Project context
  repeated ProjectLayout project_layouts = 58;
  repeated WorkspaceFolder workspace_folders = 81;
  
  // Environment info
  EnvironmentInfo environment_info = 26;
}
```

### Message Types

```protobuf
message ConversationMessage {
  string text = 1;                                 // The message content
  MessageType type = 2;                            // HUMAN=1, AI=2
  string bubble_id = 13;
  string server_bubble_id = 32;
  string request_id = 74;
  
  // Code attachments
  repeated CodeChunk attached_code_chunks = 3;
  repeated CodeBlock codebase_context_chunks = 4;
  repeated string attached_folders = 11;
  repeated FolderInfo attached_folders_new = 14;
  
  // Images
  repeated ImageProto images = 10;
  
  // Lints/errors
  repeated LinterErrors multi_file_linter_errors = 25;
  repeated Lints lints = 15;
  
  // Tool results
  repeated ToolResult tool_results = 18;
  
  // Suggested code blocks
  repeated SuggestedCodeBlock suggested_code_blocks = 23;
  
  // Git context
  repeated Commit commits = 5;
  repeated PullRequest pull_requests = 6;
  repeated GitDiff git_diffs = 7;
  ViewableGitContext git_context = 37;
  
  // Thinking
  Thinking thinking = 45;
  repeated Thinking all_thinking_blocks = 46;
  ThinkingStyle thinking_style = 85;              // DEFAULT=1, CODEX=2, GPT5=3
  int32 thinking_duration_ms = 65;
  
  // Rules
  repeated CursorRule cursor_rules = 43;
  
  // Agentic features
  bool is_agentic = 29;
  repeated ClientSideToolV2 supported_tools = 51;
  repeated TodoItem todos = 71;
}

enum MessageType {
  UNSPECIFIED = 0;
  HUMAN = 1;
  AI = 2;
}
```

### Model Details

```protobuf
message ModelDetails {
  string model_name = 1;
  string api_key = 2;                              // Optional: for BYOK
  bool enable_ghost_mode = 3;
  AzureState azure_state = 4;                      // Azure-specific config
  bool enable_slow_pool = 5;
  string openai_api_base_url = 6;                  // Custom base URL
  BedrockState bedrock_state = 7;                  // AWS Bedrock config
  bool max_mode = 8;                               // Max/premium mode
}
```

## Response Format

### Main Response: `StreamUnifiedChatResponse`

The streaming response message `aiserver.v1.StreamUnifiedChatResponse`:

```protobuf
message StreamUnifiedChatResponse {
  // Main text content (streamed in chunks)
  string text = 1;
  string server_bubble_id = 22;
  
  // Debugging info
  string debugging_only_chat_prompt = 2;
  int32 debugging_only_token_count = 3;
  string filled_prompt = 5;
  
  // Intermediate/thinking content
  string intermediate_text = 7;
  Thinking thinking = 25;
  ThinkingStyle thinking_style = 37;
  
  // Tool calls (the key streaming feature)
  StreamedBackToolCall tool_call = 13;
  StreamedBackToolCallV2 tool_call_v2 = 36;
  StreamedBackPartialToolCall partial_tool_call = 15;
  FinalToolResult final_tool_result = 16;
  bool parallel_tool_calls_complete = 32;
  
  // Citations and references
  DocumentationCitation document_citation = 4;
  DocsReference docs_reference = 9;
  WebCitation web_citation = 11;
  AiWebSearchResults ai_web_search_results = 33;
  
  // Code references
  SymbolLink symbol_link = 17;
  FileLink file_link = 19;
  UsedCode used_code = 24;
  
  // Status updates
  StatusUpdates status_updates = 12;
  ServiceStatusUpdate service_status_update = 20;
  ContextWindowStatus context_window_status = 30;
  
  // Chunk metadata
  ChunkIdentity chunk_identity = 8;
  
  // Git context
  ViewableGitContext viewable_git_context = 21;
  ContextPieceUpdate context_piece_update = 23;
  
  // Conversation management
  ConversationSummary conversation_summary = 18;
  ConversationSummaryStarter conversation_summary_starter = 28;
  bool should_break_ai_message = 14;
  
  // Subagent/task returns
  SubagentReturnCall subagent_return = 29;
  
  // Image descriptions
  ImageDescription image_description = 31;
  
  // Feedback
  StarsFeedbackRequest stars_feedback_request = 34;
  
  // Debug: raw model request
  string model_provider_request_json = 35;
  
  // Usage tracking
  string usage_uuid = 27;
  bool is_using_slow_request = 10;
  bool is_big_file = 6;
}
```

### Streamed Tool Calls

```protobuf
message StreamedBackToolCall {
  ClientSideToolV2 tool = 1;
  string tool_call_id = 2;
  
  // Tool-specific params (oneof)
  oneof params {
    ReadSemsearchFilesStream read_semsearch_files_stream = 3;
    RipgrepSearchStream ripgrep_search_stream = 5;
    ReadFileStream read_file_stream = 7;
    ListDirStream list_dir_stream = 12;
    EditFileStream edit_file_stream = 13;
    ToolCallFileSearchStream file_search_stream = 14;
    SemanticSearchFullStream semantic_search_full_stream = 19;
    DeleteFileStream delete_file_stream = 21;
    ReapplyStream reapply_stream = 22;
    RunTerminalCommandV2Stream run_terminal_command_v2_stream = 25;
    FetchRulesStream fetch_rules_stream = 26;
    WebSearchStream web_search_stream = 28;
    MCPStream mcp_stream = 29;
    SearchSymbolsStream search_symbols_stream = 33;
    GotodefStream gotodef_stream = 41;
    BackgroundComposerFollowupStream background_composer_followup_stream = 34;
    KnowledgeBaseStream knowledge_base_stream = 35;
    FetchPullRequestStream fetch_pull_request_stream = 36;
    DeepSearchStream deep_search_stream = 37;
    CreateDiagramStream create_diagram_stream = 38;
    FixLintsStream fix_lints_stream = 39;
    ReadLintsStream read_lints_stream = 40;
    TaskStream task_stream = 42;
    AwaitTaskStream await_task_stream = 43;
    TodoReadStream todo_read_stream = 44;
    TodoWriteStream todo_write_stream = 45;
    EditFileV2Stream edit_file_v2_stream = 52;
    // ... more tool streams
  }
}
```

## Supported Client-Side Tools

```protobuf
enum ClientSideToolV2 {
  UNSPECIFIED = 0;
  READ_SEMSEARCH_FILES = 1;
  RIPGREP_SEARCH = 3;
  READ_FILE = 5;
  LIST_DIR = 6;
  EDIT_FILE = 7;
  FILE_SEARCH = 8;
  SEMANTIC_SEARCH_FULL = 9;
  DELETE_FILE = 11;
  REAPPLY = 12;
  RUN_TERMINAL_COMMAND_V2 = 15;
  FETCH_RULES = 16;
  WEB_SEARCH = 18;
  MCP = 19;
  SEARCH_SYMBOLS = 23;
  BACKGROUND_COMPOSER_FOLLOWUP = 24;
  KNOWLEDGE_BASE = 25;
  FETCH_PULL_REQUEST = 26;
  DEEP_SEARCH = 27;
  CREATE_DIAGRAM = 28;
  FIX_LINTS = 29;
  READ_LINTS = 30;
  GO_TO_DEFINITION = 31;
  TASK = 32;
  AWAIT_TASK = 33;
  TODO_READ = 34;
  TODO_WRITE = 35;
  EDIT_FILE_V2 = 38;
  LIST_DIR_V2 = 39;
  READ_FILE_V2 = 40;
  RIPGREP_RAW_SEARCH = 41;
  GLOB_FILE_SEARCH = 42;
  CREATE_PLAN = 43;
  LIST_MCP_RESOURCES = 44;
  READ_MCP_RESOURCE = 45;
  READ_PROJECT = 46;
  UPDATE_PROJECT = 47;
  TASK_V2 = 48;
  CALL_MCP_TOOL = 49;
  APPLY_AGENT_DIFF = 50;
  ASK_QUESTION = 51;
  SWITCH_MODE = 52;
  GENERATE_IMAGE = 53;
  COMPUTER_USE = 54;
  WRITE_SHELL_STDIN = 55;
}
```

## Unified Modes

```protobuf
enum UnifiedMode {
  UNSPECIFIED = 0;
  CHAT = 1;           // Regular chat mode
  AGENT = 2;          // Agent/agentic mode (tool use)
  EDIT = 3;           // Inline edit mode
  CUSTOM = 4;         // Custom mode
  PLAN = 5;           // Planning mode
  DEBUG = 6;          // Debug mode
}
```

## Thinking Levels

```protobuf
enum ThinkingLevel {
  UNSPECIFIED = 0;
  MEDIUM = 1;
  HIGH = 2;
}

enum ThinkingStyle {
  UNSPECIFIED = 0;
  DEFAULT = 1;
  CODEX = 2;          // OpenAI Codex-style
  GPT5 = 3;           // GPT-5 style
}
```

## Request Wrapper for Tool Calls

When the model needs to call tools and get results back, there's a wrapper:

```protobuf
message StreamUnifiedChatRequestWithTools {
  oneof request {
    StreamUnifiedChatRequest stream_unified_chat_request = 1;
    ClientSideToolV2Result client_side_tool_v2_result = 2;
  }
}
```

## Key Observations

1. **Protocol**: Uses gRPC with Protocol Buffers, not REST/JSON
2. **Streaming**: Responses are streamed as a series of `StreamUnifiedChatResponse` messages
3. **Tool Calls**: Tools are called via `StreamedBackToolCall` messages in the response stream
4. **Client-Server Loop**: Client executes tools locally and returns results via `ClientSideToolV2Result`
5. **Rich Context**: Supports code chunks, linter errors, git context, images, etc.
6. **Multiple Modes**: Chat, Agent, Edit, Plan, Debug modes with different capabilities
7. **Thinking**: Supports extended thinking with configurable levels
8. **MCP Support**: Native support for Model Context Protocol tools

## Tool Parameters

### ReadFileParams
```protobuf
message ReadFileParams {
  string relative_workspace_path = 1;
  bool read_entire_file = 2;
  int32 start_line_one_indexed = 3;           // Optional
  int32 end_line_one_indexed_inclusive = 4;   // Optional
  bool file_is_allowed_to_be_read_entirely = 5;
  int32 max_lines = 6;                        // Optional
  int32 max_chars = 7;                        // Optional
  int32 min_lines = 8;                        // Optional
}
```

### EditFileParams
```protobuf
message EditFileParams {
  string relative_workspace_path = 1;
  string language = 2;
  bool blocking = 4;
  string contents = 3;                        // Full file contents (for write)
  string instructions = 5;                    // Optional instructions
  string old_string = 6;                      // For search-replace
  string new_string = 7;                      // For search-replace
  bool allow_multiple_matches = 8;
  bool use_whitespace_insensitive_fallback = 10;
  bool use_did_you_mean_fuzzy_match = 11;
  bool gracefully_handle_recoverable_errors = 16;
  repeated LineRange line_ranges = 9;
  int32 notebook_cell_idx = 13;               // For notebooks
  bool is_new_cell = 14;
  string cell_language = 15;
  string edit_category = 17;
  bool should_eagerly_process_lints = 18;
}
```

### ListDirParams
```protobuf
message ListDirParams {
  string directory_path = 1;
}
```

### RipgrepSearchParams
```protobuf
message RipgrepSearchParams {
  ITextQueryBuilderOptionsProto options = 1;
  IPatternInfoProto pattern_info = 2;
}
```

## Tool Results

When the client executes a tool, it returns results via `ClientSideToolV2Result`:

```protobuf
message ClientSideToolV2Result {
  ClientSideToolV2 tool = 1;
  
  oneof result {
    ReadSemsearchFilesResult read_semsearch_files_result = 2;
    RipgrepSearchResult ripgrep_search_result = 4;
    ReadFileResult read_file_result = 6;
    ListDirResult list_dir_result = 9;
    EditFileResult edit_file_result = 10;
    ToolCallFileSearchResult file_search_result = 11;
    SemanticSearchFullResult semantic_search_full_result = 18;
    DeleteFileResult delete_file_result = 20;
    ReapplyResult reapply_result = 21;
    RunTerminalCommandV2Result run_terminal_command_v2_result = 24;
    FetchRulesResult fetch_rules_result = 25;
    WebSearchResult web_search_result = 27;
    MCPResult mcp_result = 28;
    SearchSymbolsResult search_symbols_result = 32;
    BackgroundComposerFollowupResult background_composer_followup_result = 33;
    KnowledgeBaseResult knowledge_base_result = 34;
    FetchPullRequestResult fetch_pull_request_result = 36;
    DeepSearchResult deep_search_result = 37;
    // ... more result types
  }
}
```

## Example JSON Representation

While the actual format is protobuf, here's what a conceptual JSON request might look like:

```json
{
  "conversation": [
    {
      "text": "Help me fix this bug",
      "type": "HUMAN",
      "bubble_id": "uuid-1",
      "attached_code_chunks": [
        {
          "file_name": "src/app.py",
          "start_line": 10,
          "end_line": 20,
          "text": "def buggy_function():\n    ..."
        }
      ]
    }
  ],
  "model_details": {
    "model_name": "claude-sonnet-4-20250514"
  },
  "is_agentic": true,
  "unified_mode": "AGENT",
  "supported_tools": [
    "READ_FILE",
    "EDIT_FILE",
    "LIST_DIR",
    "RIPGREP_SEARCH",
    "RUN_TERMINAL_COMMAND_V2"
  ],
  "conversation_id": "conv-uuid",
  "thinking_level": "MEDIUM"
}
```

And a conceptual streaming response:

```json
// Chunk 1: Text
{"text": "I'll help you fix that bug. Let me first "}

// Chunk 2: More text
{"text": "read the full file to understand the context."}

// Chunk 3: Tool call
{
  "tool_call": {
    "tool": "READ_FILE",
    "tool_call_id": "tc-1",
    "read_file_stream": {
      "file_path": "src/app.py"
    }
  }
}

// Chunk 4: Status update
{"status_updates": {"message": "Reading file..."}}

// After client returns tool result, more streaming continues...

// Chunk N: Thinking
{
  "thinking": {
    "text": "The bug is in line 15 where..."
  }
}

// Final chunks: More text with the solution
{"text": "I found the issue. The problem is..."}
```
