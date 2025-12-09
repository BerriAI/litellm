/**
 * Type definitions for Content Filter Configuration
 */

export interface PrebuiltPattern {
  name: string;
  display_name: string;
  category: string;
  description: string;
}

export interface Pattern {
  id: string;
  type: "prebuilt" | "custom";
  name: string;
  display_name?: string;
  pattern?: string;
  action: "BLOCK" | "MASK";
}

export interface BlockedWord {
  id: string;
  keyword: string;
  action: "BLOCK" | "MASK";
  description?: string;
}

export interface ContentFilterSettings {
  prebuilt_patterns: PrebuiltPattern[];
  pattern_categories: string[];
  supported_actions: string[];
}

export interface ContentFilterConfigurationProps {
  prebuiltPatterns: PrebuiltPattern[];
  categories: string[];
  selectedPatterns: Pattern[];
  blockedWords: BlockedWord[];
  blockedWordsFile?: string;
  onPatternAdd: (pattern: Pattern) => void;
  onPatternRemove: (id: string) => void;
  onPatternActionChange: (id: string, action: "BLOCK" | "MASK") => void;
  onBlockedWordAdd: (word: BlockedWord) => void;
  onBlockedWordRemove: (id: string) => void;
  onBlockedWordUpdate: (id: string, field: string, value: any) => void;
  onFileUpload: (content: string) => void;
  accessToken: string | null;
}

