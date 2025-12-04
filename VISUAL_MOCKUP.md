# Visual Mockup - Polished JSON Viewer

## Before vs After Comparison

### BEFORE (Unpolished)
```
┌────────────────────────────────────────────────────────────────────────────────┐
│  Request                                                    [📋 Copy Icon]      │
├────────────────────────────────────────────────────────────────────────────────┤
│ ▼ {                                                                            │
│   ▼ "model": "a2a_agent/Ishaan-Jaffer"                                        │
│   ▼ "custom_llm_provider": "a2a_agent"                                        │
│   }                                                                            │
│                                                                                │
│  (White background, basic styling, collapsible tree view)                     │
└────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────┐
│  Response • HTTP code 400                                   [📋 Copy Icon]     │
├────────────────────────────────────────────────────────────────────────────────┤
│ ▼ {                                                                            │
│   ▼ "id": "f53f56f4-b096-4cd6-953e-482a4a094e4b"                              │
│   ▼ "error": null                                                              │
│   }                                                                            │
│                                                                                │
│  (White background, basic styling, collapsible tree view)                     │
└────────────────────────────────────────────────────────────────────────────────┘
```

### AFTER (Polished - Matches API Reference)
```
Request                                                        [📋] ← hover effect
╔════════════════════════════════════════════════════════════════════════════════╗
║ 1  {                                                                           ║
║ 2    "model": "a2a_agent/Ishaan-Jaffer",                                      ║
║ 3    "custom_llm_provider": "a2a_agent",                                      ║
║ 4    "messages": [                                                            ║
║ 5      {                                                                      ║
║ 6        "role": "user",                                                      ║
║ 7        "content": "Create an outline for a post about Java"                ║
║ 8      }                                                                      ║
║ 9    ],                                                                       ║
║10    "temperature": 0.7,                                                      ║
║11    "max_tokens": 2048,                                                      ║
║12    "stream": false                                                          ║
║13  }                                                                           ║
║                                                                                ║
║  Light gray background (#fafafa)                                              ║
║  Syntax highlighting with colors                                              ║
║  Professional typography                                                       ║
╚════════════════════════════════════════════════════════════════════════════════╝

Response • HTTP 200                                            [✓] ← copied!
╔════════════════════════════════════════════════════════════════════════════════╗
║ 1  {                                                                           ║
║ 2    "id": "f53f56f4-b096-4cd6-953e-482a4a094e4b",                            ║
║ 3    "error": null,                                                           ║
║ 4    "result": {                                                              ║
║ 5      "eventId": "bc9-64f3-4d8a-95c5-33ed1d974984",                          ║
║ 6      "kind": "task",                                                        ║
║ 7      "status": {                                                            ║
║ 8        "state": "completed",                                                ║
║ 9        "timestamp": "2025-12-04T00:16:01.442801+00:00"                      ║
║10      },                                                                     ║
║11      "history": [ ... ]                                                     ║
║12    }                                                                         ║
║13  }                                                                           ║
║                                                                                ║
║  Light gray background (#fafafa)                                              ║
║  Syntax highlighting with colors                                              ║
║  Professional typography                                                       ║
╚════════════════════════════════════════════════════════════════════════════════╝
```

## Design Specifications

### Color Palette (Matches API Reference)
```
Background:       #fafafa (Light gray - same as CodeBlock)
Border:           #e5e7eb (Gray-200)
Text:             #111827 (Gray-900)
Secondary Text:   #6b7280 (Gray-500)
Button BG:        #f3f4f6 (Gray-100)
Button Hover:     #e5e7eb (Gray-200)
Error Text:       #dc2626 (Red-600)
Success Icon:     #10b981 (Green-500)
```

### Typography
```
Headers:          16px, font-semibold, text-gray-900
JSON Content:     14px (0.875rem), line-height 1.6
Secondary Info:   14px, font-normal, text-gray-500
Monospace:        'SF Mono', 'Monaco', 'Courier New'
```

### Spacing
```
Panel Gap:        24px (gap-6)
Padding:          20px (1.25rem)
Header Margin:    12px bottom (mb-3)
Border Radius:    8px (0.5rem)
Max Height:       500px (scrollable)
```

### Interactive Elements
```
Copy Button:
  - Default: Gray background with clipboard icon
  - Hover: Darker gray background
  - Clicked: Shows checkmark for 2 seconds
  - Disabled: 50% opacity, no cursor
  
Headers:
  - Clean, semibold typography
  - Error badge inline with subtle styling
  - Proper spacing and alignment
```

## Side-by-Side Layout

```
┌──────────────────────────────────┬──────────────────────────────────┐
│         REQUEST PANEL            │        RESPONSE PANEL            │
├──────────────────────────────────┼──────────────────────────────────┤
│                                  │                                  │
│  • Clean header with copy btn    │  • Clean header with copy btn    │
│  • Light gray background         │  • Light gray background         │
│  • Syntax highlighted JSON       │  • Syntax highlighted JSON       │
│  • Subtle border                 │  • Subtle border                 │
│  • Scrollable content (500px)    │  • Scrollable content (500px)    │
│  • Professional typography       │  • Professional typography       │
│                                  │                                  │
└──────────────────────────────────┴──────────────────────────────────┘
                         ↑
                    24px gap
```

## Key Visual Improvements

### 1. Headers
```
BEFORE: text-lg font-medium (18px, medium weight)
AFTER:  text-base font-semibold text-gray-900 (16px, semibold, darker)
        ↓ Better visual hierarchy and consistency
```

### 2. Copy Button
```
BEFORE: p-1 hover:bg-gray-200 rounded
        [📋] → [📋]
        
AFTER:  p-2 rounded-md bg-gray-100 hover:bg-gray-200
        [📋] → [✓] (with 2-second feedback)
        ↓ Better feedback and button styling
```

### 3. JSON Content Area
```
BEFORE: Plain white bg, basic JSON tree
        background: white
        
AFTER:  Professional code block styling
        background: #fafafa
        border: 1px solid #e5e7eb
        syntax highlighting with colors
        ↓ Much more polished and professional
```

### 4. Error Display
```
BEFORE: • HTTP code 400
        
AFTER:  • HTTP 400
        (with refined typography and spacing)
        ↓ Cleaner, more concise
```

## Consistency with API Reference Page

Both now share:
- ✅ Same background color (#fafafa)
- ✅ Same border style (border-gray-200)
- ✅ Same syntax highlighting (oneLight theme)
- ✅ Same copy button design
- ✅ Same typography and spacing
- ✅ Same border radius and shadows

## Responsive Behavior

```
Desktop (lg+):     Two columns side-by-side
Mobile/Tablet:     Stacked vertically

Both maintain:
- Proper overflow handling
- Scrollable content areas
- Touch-friendly buttons
- Readable text sizes
```

---

**Result:** A polished, professional JSON viewer that matches the quality and styling of the API Reference page! 🎨✨
