# Vision/Multimodal Support Audit

## Current Rust Implementation
The Rust gateway has **minimal vision/multimodal support**:
- `ChatMessageContent` enum with only two variants:
  - `Text(String)` - plain text content
  - `Parts(Vec<Value>)` - generic JSON array (can hold any structure but no type safety)
- No specific types for images, videos, audio, documents, or files
- No detail level support (low, high, auto)
- No base64 vs URL format handling
- No provider-specific vision format transforms
- No validation of vision content types

## Python Implementation

### Core Vision Types
Python has comprehensive multimodal support with multiple content types:

#### Image Support
```python
class ChatCompletionImageUrlObject(TypedDict, total=False):
    url: Required[str]
    detail: str  # "low", "high", "auto"
    format: str

class ChatCompletionImageObject(TypedDict):
    type: Literal["image_url"]
    image_url: str | ChatCompletionImageUrlObject
```

#### Video Support
```python
class ChatCompletionVideoUrlObject(TypedDict, total=False):
    url: Required[str]
    detail: str

class ChatCompletionVideoObject(TypedDict):
    type: Literal["video_url"]
    video_url: str | ChatCompletionVideoUrlObject
```

#### Audio Support
```python
class ChatCompletionAudioObject(ChatCompletionContentPartInputAudioParam):
    pass  # Inherits from OpenAI SDK type
```

#### Document Support
```python
class DocumentObject(TypedDict):
    type: Literal["text"]
    media_type: str
    data: str

class ChatCompletionDocumentObject(TypedDict):
    type: Literal["document"]
    source: DocumentObject
    title: str
    context: str
    citations: CitationsObject | None
```

#### File Support
```python
class ChatCompletionFileObjectFile(TypedDict, total=False):
    file_data: str
    file_id: str
    filename: str
    format: str
    detail: str  # For video/image resolution control
    video_metadata: dict[str, Any]  # For video-specific metadata

class ChatCompletionFileObject(TypedDict):
    type: Literal["file"]
    file: ChatCompletionFileObjectFile
```

#### Content Union
```python
OpenAIMessageContentListBlock = (
    ChatCompletionTextObject
    | ChatCompletionImageObject
    | ChatCompletionAudioObject
    | ChatCompletionDocumentObject
    | ChatCompletionVideoObject
    | ChatCompletionFileObject
)

OpenAIMessageContent = str | Iterable[OpenAIMessageContentListBlock]
```

### Valid Content Types
Python validates user message content types:
```python
ValidUserMessageContentTypes = [
    "text",
    "image_url",
    "input_audio",
    "audio_url",
    "document",
    "guarded_text",
    "grounding_source",
    "query",
    "video_url",
    "file",
]
```

## Provider-Specific Vision Formats

### OpenAI
- Standard `image_url` format with `url` and `detail` fields
- Supports base64 data URLs: `data:image/jpeg;base64,{base64_image}`
- Supports HTTP/HTTPS URLs
- Detail levels: "low", "high", "auto"
- Multiple images per message supported
- Image token calculation based on detail level

### Anthropic
- Different format: `image` content block with `source` field
- Source can be `base64` or `url` type
- No detail levels (uses automatic resolution)
- Supports PDF documents via `document` type
- Different token calculation

### Bedrock (Converse API)
- Uses `image` content block with `format` and `source` fields
- Source is `bytes` (base64) only (no URL support)
- Supports `document` type for PDFs
- Different format than OpenAI/Anthropic

### Azure OpenAI
- Same as OpenAI but with additional validation
- Supports Azure Blob Storage URLs
- Additional authentication for Azure URLs

### Vertex AI
- Uses `inline_data` or `file_data` for images
- Different format than OpenAI
- Supports Google Cloud Storage URLs
- MIME type required

### Other Providers
- **Mistral**: Supports image_url similar to OpenAI
- **Cohere**: Different image format
- **Gemini**: Uses `inline_data` or `file_data`

## Major Gaps in Rust Implementation

### 1. Missing Core Vision Types
**Critical Gap**: No specific vision types defined
- Need `ImageObject` struct with type, image_url fields
- Need `ImageUrlObject` struct with url, detail, format fields
- Need `VideoObject` struct for video support
- Need `AudioObject` struct for audio input
- Need `DocumentObject` struct for document support
- Need `FileObject` struct for file support

### 2. Missing Content Union
**Critical Gap**: No typed content union
- `ChatMessageContent::Parts(Vec<Value>)` is too generic
- Need proper enum with variants for each content type
- Need type-safe access to content fields
- Need serialization/deserialization per type

### 3. Missing Detail Level Support
**High Priority**: No detail level handling
- Need to parse "low", "high", "auto" detail levels
- Need to pass detail level to providers
- Need to calculate tokens based on detail level
- Need to validate detail level per provider

### 4. Missing Base64/URL Format Handling
**High Priority**: No format detection/handling
- Need to detect base64 data URLs vs HTTP URLs
- Need to validate URL formats
- Need to handle different URL schemes (http, https, data, azure, gcs)
- Need to convert between formats if needed

### 5. Missing Provider Transforms
**High Priority**: No provider-specific vision transforms
- OpenAI → Anthropic vision format conversion
- OpenAI → Bedrock vision format conversion
- Anthropic → OpenAI vision format conversion
- Bedrock → OpenAI vision format conversion
- URL to base64 conversion for providers that require it

### 6. Missing Image Token Calculation
**High Priority**: No vision token calculation
- Need to calculate tokens for images based on resolution
- Need to handle detail levels in token calculation
- Need to handle different token calculation per provider
- Need to add vision tokens to prompt token count

### 7. Missing Multiple Image Support
**Medium Priority**: No multi-image handling
- Need to support multiple images in single message
- Need to validate image count per provider
- Need to calculate tokens for multiple images
- Need to handle image ordering

### 8. Missing Content Validation
**Medium Priority**: No vision content validation
- Need to validate image URL format
- Need to validate base64 format
- Need to validate file size limits
- Need to validate supported formats (jpeg, png, gif, webp, etc.)

### 9. Missing Video Support
**Medium Priority**: No video support
- Need to add video content type
- Need to handle video URLs
- Need to handle video frame extraction (if needed)
- Need to calculate video tokens

### 10. Missing Audio Support
**Medium Priority**: No audio input support
- Need to add audio content type
- Need to handle audio URLs and base64
- Need to handle different audio formats
- Need to calculate audio tokens

### 11. Missing Document Support
**Low Priority**: No document support
- Need to add document content type
- Need to handle PDF documents
- Need to extract text from documents (if needed)
- Need to calculate document tokens

### 12. Missing File Support
**Low Priority**: No file support
- Need to add file content type
- Need to handle file uploads
- Need to handle file IDs
- Need to resolve file content

## Priority Gaps for Major Providers (OpenAI, Anthropic, Bedrock)

### Critical (Must Have)
1. **Core Vision Types** - Define ImageObject, ImageUrlObject structs
2. **Content Union** - Replace generic Parts with typed enum
3. **Provider Transforms** - Implement vision transforms for OpenAI/Anthropic/Bedrock
4. **Base64/URL Handling** - Detect and handle different URL formats

### High Priority
5. **Detail Level Support** - Parse and pass detail levels
6. **Image Token Calculation** - Calculate tokens for images
7. **Multiple Image Support** - Handle multiple images per message
8. **Content Validation** - Validate vision content formats

### Medium Priority
9. **Video Support** - Add video content type
10. **Audio Support** - Add audio input support
11. **Provider-Specific Features** - Handle provider-specific vision features

### Low Priority
12. **Document Support** - Add document content type
13. **File Support** - Add file content type
14. **Advanced Features** - Image editing, grounding, etc.

## Implementation Plan

### Phase 1: Core Types (Critical)
1. Define `ImageObject` struct
2. Define `ImageUrlObject` struct
3. Define `VideoObject` struct
4. Define `AudioObject` struct
5. Define `DocumentObject` struct
6. Define `FileObject` struct
7. Replace `ChatMessageContent::Parts` with typed enum
8. Add tests for type serialization/deserialization

### Phase 2: Format Handling (Critical)
9. Implement base64 data URL detection
10. Implement HTTP/HTTPS URL validation
11. Implement URL scheme handling (azure, gcs, etc.)
12. Add format conversion utilities
13. Add tests for format handling

### Phase 3: Provider Transforms (Critical)
14. Implement OpenAI vision transform (passthrough)
15. Implement Anthropic vision transform (OpenAI ↔ Anthropic format)
16. Implement Bedrock vision transform (OpenAI ↔ Bedrock format)
17. Add URL to base64 conversion for Bedrock
18. Add tests for each provider transform

### Phase 4: Detail Levels (High Priority)
19. Parse detail level from request
20. Validate detail level per provider
21. Pass detail level to providers
22. Add tests for detail level handling

### Phase 5: Token Calculation (High Priority)
23. Implement image token calculation for OpenAI
24. Implement image token calculation for Anthropic
25. Implement image token calculation for Bedrock
26. Handle detail levels in token calculation
27. Add vision tokens to prompt token count
28. Add tests for token calculation

### Phase 6: Multi-Image Support (High Priority)
29. Support multiple images in single message
30. Validate image count per provider
31. Calculate tokens for multiple images
32. Handle image ordering
33. Add tests for multi-image support

### Phase 7: Validation (Medium Priority)
34. Validate image URL format
35. Validate base64 format
36. Validate file size limits
37. Validate supported formats
38. Add tests for validation

### Phase 8: Advanced Content Types (Medium Priority)
39. Add video support
40. Add audio input support
41. Add document support
42. Add file support
43. Add tests for advanced content types

## Testing Plan
- Unit tests for each vision type
- Integration tests with real provider responses
- Provider transform tests (OpenAI ↔ Anthropic ↔ Bedrock)
- Format handling tests (base64, URL, etc.)
- Detail level tests
- Token calculation tests
- Multi-image tests
- Validation tests
- Error case tests (invalid formats, unsupported types, etc.)

## Comparison with Previous Audits
This gap is **less critical** than function/tool calling but **more critical** than streaming edge cases because:
1. Vision is a core feature for modern LLM applications (GPT-4V, Claude 3, etc.)
2. Without vision support, the gateway can't support image-based workflows
3. Major providers (OpenAI, Anthropic, Bedrock) all support vision
4. Python has comprehensive vision support that we need to match
5. Vision is used in many production applications

However, it's less critical than tool calling because:
1. Tool calling is more fundamental to agentic workflows
2. Vision is more provider-specific (harder to abstract)
3. Vision token calculation is complex and provider-dependent

## Estimated Effort
- Phase 1 (Core Types): 2-3 days
- Phase 2 (Format Handling): 1-2 days
- Phase 3 (Provider Transforms): 3-4 days
- Phase 4 (Detail Levels): 1 day
- Phase 5 (Token Calculation): 2-3 days
- Phase 6 (Multi-Image): 1-2 days
- Phase 7 (Validation): 1-2 days
- Phase 8 (Advanced Types): 2-3 days

**Total**: 13-20 days for complete vision/multimodal parity
