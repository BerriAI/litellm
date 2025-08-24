# Coding Standards

## Critical Fullstack Rules

- **Type Safety:** All functions must have type hints, use Pydantic for validation
- **API Consistency:** All endpoints return OpenAI-compatible format
- **Error Handling:** Use standard error handler, never expose internal errors
- **Async First:** Use async/await for all I/O operations
- **Cost Tracking:** Every LLM call must record cost and usage
- **Configuration:** Access config only through settings object, never env directly
- **Logging:** Use structured logging with request_id correlation
- **Testing:** Minimum 80% code coverage for new features

## Naming Conventions

| Element | Frontend | Backend | Example |
|---------|----------|---------|---------|
| Components | PascalCase | - | `KeyList.tsx` |
| Hooks | camelCase with 'use' | - | `useAuth.ts` |
| API Routes | - | kebab-case | `/api/key-management` |
| Database Tables | - | PascalCase with prefix | `LiteLLM_UserTable` |
