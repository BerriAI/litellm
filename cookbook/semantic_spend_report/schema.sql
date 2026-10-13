PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    started_at TEXT NOT NULL, model TEXT NOT NULL, spend_usd TEXT,
    instruction TEXT NOT NULL, response TEXT NOT NULL, context_json TEXT NOT NULL,
    status TEXT NOT NULL, content_missing INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS categories (
    category_id TEXT PRIMARY KEY, name TEXT NOT NULL, definition_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS classifications (
    request_id TEXT PRIMARY KEY REFERENCES requests(request_id),
    category_id TEXT NOT NULL REFERENCES categories(category_id),
    proposed_category TEXT NOT NULL, confidence REAL NOT NULL,
    probabilities_json TEXT NOT NULL, classifier TEXT NOT NULL,
    analysis_call_id TEXT NOT NULL, input_fingerprint TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS summaries (
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    category_id TEXT NOT NULL REFERENCES categories(category_id),
    facts_json TEXT NOT NULL, analysis_call_id TEXT NOT NULL,
    PRIMARY KEY (session_id, category_id)
);
