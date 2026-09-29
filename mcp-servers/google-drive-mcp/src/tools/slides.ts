import { z } from 'zod';
import { v4 as uuidv4 } from 'uuid';
import type { slides_v1 } from 'googleapis';
import type { ToolDefinition, ToolResult, ToolContext } from '../types.js';
import { errorResponse } from '../types.js';
import { uploadImageToDrive, deleteDriveFile } from '../utils/driveImageUpload.js';

// ---------------------------------------------------------------------------
// Zod Schemas
// ---------------------------------------------------------------------------

const CreateGoogleSlidesSchema = z.object({
  name: z.string().min(1, "Presentation name is required"),
  slides: z.array(z.object({
    title: z.string(),
    content: z.string()
  })).min(1, "At least one slide is required"),
  parentFolderId: z.string().optional()
});

const UpdateGoogleSlidesSchema = z.object({
  presentationId: z.string().min(1, "Presentation ID is required"),
  slides: z.array(z.object({
    title: z.string(),
    content: z.string()
  })).min(1, "At least one slide is required")
});

const GetGoogleSlidesContentSchema = z.object({
  presentationId: z.string().min(1, "Presentation ID is required"),
  slideIndex: z.number().min(0).optional()
});

const FormatGoogleSlidesTextSchema = z.object({
  presentationId: z.string().min(1, "Presentation ID is required"),
  objectId: z.string().min(1, "Object ID is required"),
  startIndex: z.number().min(0).optional(),
  endIndex: z.number().min(0).optional(),
  bold: z.boolean().optional(),
  italic: z.boolean().optional(),
  underline: z.boolean().optional(),
  strikethrough: z.boolean().optional(),
  fontSize: z.number().optional(),
  fontFamily: z.string().optional(),
  foregroundColor: z.object({
    red: z.number().min(0).max(1).optional(),
    green: z.number().min(0).max(1).optional(),
    blue: z.number().min(0).max(1).optional()
  }).optional()
});

const FormatGoogleSlidesParagraphSchema = z.object({
  presentationId: z.string().min(1, "Presentation ID is required"),
  objectId: z.string().min(1, "Object ID is required"),
  alignment: z.enum(['START', 'CENTER', 'END', 'JUSTIFIED']).optional(),
  lineSpacing: z.number().optional(),
  bulletStyle: z.enum(['NONE', 'DISC', 'ARROW', 'SQUARE', 'DIAMOND', 'STAR', 'NUMBERED']).optional()
});

const StyleGoogleSlidesShapeSchema = z.object({
  presentationId: z.string().min(1, "Presentation ID is required"),
  objectId: z.string().min(1, "Shape object ID is required"),
  backgroundColor: z.object({
    red: z.number().min(0).max(1).optional(),
    green: z.number().min(0).max(1).optional(),
    blue: z.number().min(0).max(1).optional(),
    alpha: z.number().min(0).max(1).optional()
  }).optional(),
  outlineColor: z.object({
    red: z.number().min(0).max(1).optional(),
    green: z.number().min(0).max(1).optional(),
    blue: z.number().min(0).max(1).optional()
  }).optional(),
  outlineWeight: z.number().optional(),
  outlineDashStyle: z.enum(['SOLID', 'DOT', 'DASH', 'DASH_DOT', 'LONG_DASH', 'LONG_DASH_DOT']).optional()
});

const SetGoogleSlidesBackgroundSchema = z.object({
  presentationId: z.string().min(1, "Presentation ID is required"),
  pageObjectIds: z.array(z.string()).min(1, "At least one page object ID is required"),
  backgroundColor: z.object({
    red: z.number().min(0).max(1).optional(),
    green: z.number().min(0).max(1).optional(),
    blue: z.number().min(0).max(1).optional(),
    alpha: z.number().min(0).max(1).optional()
  })
});

const CreateGoogleSlidesTextBoxSchema = z.object({
  presentationId: z.string().min(1, "Presentation ID is required"),
  pageObjectId: z.string().min(1, "Page object ID is required"),
  text: z.string().min(1, "Text content is required"),
  x: z.number(),
  y: z.number(),
  width: z.number(),
  height: z.number(),
  fontSize: z.number().optional(),
  bold: z.boolean().optional(),
  italic: z.boolean().optional()
});

const CreateGoogleSlidesShapeSchema = z.object({
  presentationId: z.string().min(1, "Presentation ID is required"),
  pageObjectId: z.string().min(1, "Page object ID is required"),
  shapeType: z.enum(['RECTANGLE', 'ELLIPSE', 'DIAMOND', 'TRIANGLE', 'STAR', 'ROUND_RECTANGLE', 'ARROW']),
  x: z.number(),
  y: z.number(),
  width: z.number(),
  height: z.number(),
  backgroundColor: z.object({
    red: z.number().min(0).max(1).optional(),
    green: z.number().min(0).max(1).optional(),
    blue: z.number().min(0).max(1).optional(),
    alpha: z.number().min(0).max(1).optional()
  }).optional()
});

const GetGoogleSlidesSpeakerNotesSchema = z.object({
  presentationId: z.string().min(1, "Presentation ID is required"),
  slideIndex: z.number().min(0, "Slide index must be non-negative")
});

const UpdateGoogleSlidesSpeakerNotesSchema = z.object({
  presentationId: z.string().min(1, "Presentation ID is required"),
  slideIndex: z.number().min(0, "Slide index must be non-negative"),
  notes: z.string()
});

const DeleteGoogleSlideSchema = z.object({
  presentationId: z.string().min(1, "Presentation ID is required"),
  slideObjectId: z.string().min(1, "Slide object ID is required")
});

const DuplicateSlideSchema = z.object({
  presentationId: z.string().min(1, "Presentation ID is required"),
  slideObjectId: z.string().min(1, "Slide object ID is required")
});

const ReorderSlidesSchema = z.object({
  presentationId: z.string().min(1, "Presentation ID is required"),
  slideObjectIds: z.array(z.string().min(1)).min(1, "At least one slide object ID is required"),
  insertionIndex: z.number().int().min(0)
});

const ReplaceAllTextInSlidesSchema = z.object({
  presentationId: z.string().min(1, "Presentation ID is required"),
  containsText: z.string().min(1, "containsText is required"),
  replaceText: z.string(),
  matchCase: z.boolean().optional().default(false)
});

const ExportSlideThumbnailSchema = z.object({
  presentationId: z.string().min(1, "Presentation ID is required"),
  slideObjectId: z.string().min(1, "Slide object ID is required"),
  mimeType: z.enum(["PNG", "JPEG"]).optional().default("PNG"),
  size: z.enum(["SMALL", "MEDIUM", "LARGE"]).optional().default("LARGE")
});

const InsertSlidesImageFromUrlSchema = z.object({
  presentationId: z.string().min(1, "Presentation ID is required"),
  pageObjectId: z.string().min(1, "Slide/page object ID is required"),
  imageUrl: z.string().url("A valid image URL is required"),
  x: z.number().optional().default(0),
  y: z.number().optional().default(0),
  width: z.number().optional(),
  height: z.number().optional(),
});

const MoveSlideElementSchema = z.object({
  presentationId: z.string().min(1, "Presentation ID is required"),
  objectId: z.string().min(1, "Element object ID is required"),
  x: z.number().optional(),
  y: z.number().optional(),
  width: z.number().optional(),
  height: z.number().optional(),
});

const DeleteSlideElementSchema = z.object({
  presentationId: z.string().min(1, "Presentation ID is required"),
  objectId: z.string().min(1, "Element object ID is required"),
});

const GetSlideElementInfoSchema = z.object({
  presentationId: z.string().min(1, "Presentation ID is required"),
  slideObjectId: z.string().optional(),
});

const InsertSlidesLocalImageSchema = z.object({
  presentationId: z.string().min(1, "Presentation ID is required"),
  pageObjectId: z.string().min(1, "Slide/page object ID is required"),
  localImagePath: z.string().min(1, "Local image path is required"),
  x: z.number().optional().default(0),
  y: z.number().optional().default(0),
  width: z.number().optional(),
  height: z.number().optional(),
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function insertImageIntoSlide(
  ctx: ToolContext,
  presentationId: string,
  pageObjectId: string,
  imageUrl: string,
  x: number,
  y: number,
  width?: number,
  height?: number,
): Promise<ToolResult> {
  const slidesService = ctx.google.slides({ version: 'v1', auth: ctx.authClient });
  const objectId = `img_${uuidv4().substring(0, 8)}`;

  const elementProperties: slides_v1.Schema$PageElementProperties = {
    pageObjectId,
    transform: {
      scaleX: 1,
      scaleY: 1,
      translateX: x,
      translateY: y,
      unit: 'EMU',
    },
  };

  if (width != null && height != null) {
    elementProperties.size = {
      width: { magnitude: width, unit: 'EMU' },
      height: { magnitude: height, unit: 'EMU' },
    };
  }

  await slidesService.presentations.batchUpdate({
    presentationId,
    requestBody: {
      requests: [{
        createImage: { objectId, url: imageUrl, elementProperties },
      }],
    },
  });

  return {
    content: [{ type: 'text', text: `Inserted image into slide ${pageObjectId} (objectId: ${objectId})` }],
    isError: false,
  };
}

// ---------------------------------------------------------------------------
// Tool Definitions
// ---------------------------------------------------------------------------

const SetSlideVisibilitySchema = z.object({
  presentationId: z.string().min(1, "Presentation ID is required"),
  slideObjectIds: z.array(z.string().min(1)).min(1, "At least one slide object ID is required"),
  skipped: z.boolean(),
});

const ReplaceSlideImageSchema = z.object({
  presentationId: z.string().min(1, "Presentation ID is required"),
  imageObjectId: z.string().min(1, "Image object ID is required"),
  imageUrl: z.string().url().optional(),
  localImagePath: z.string().min(1).optional(),
  replaceMethod: z.enum(['CENTER_INSIDE', 'CENTER_CROP']).optional().default('CENTER_INSIDE'),
}).refine((d) => (d.imageUrl !== undefined) !== (d.localImagePath !== undefined), {
  message: "Provide exactly one of imageUrl or localImagePath",
});

const SetElementZOrderSchema = z.object({
  presentationId: z.string().min(1, "Presentation ID is required"),
  pageElementObjectIds: z.array(z.string().min(1)).min(1, "At least one element object ID is required"),
  operation: z.enum(['BRING_TO_FRONT', 'BRING_FORWARD', 'SEND_BACKWARD', 'SEND_TO_BACK']),
});

const SetElementTextSchema = z.object({
  presentationId: z.string().min(1, "Presentation ID is required"),
  objectId: z.string().min(1, "Object ID is required"),
  text: z.string(),
});

export const toolDefinitions: ToolDefinition[] = [
  {
    name: "createGoogleSlides",
    description: "Create a new Google Slides presentation",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Presentation name" },
        slides: {
          type: "array",
          description: "Array of slide objects",
          items: {
            type: "object",
            properties: {
              title: { type: "string" },
              content: { type: "string" }
            }
          }
        },
        parentFolderId: { type: "string", description: "Parent folder ID (defaults to root)" }
      },
      required: ["name", "slides"]
    }
  },
  {
    name: "updateGoogleSlides",
    description: "Update an existing Google Slides presentation",
    inputSchema: {
      type: "object",
      properties: {
        presentationId: { type: "string", description: "Presentation ID" },
        slides: {
          type: "array",
          description: "Array of slide objects to replace existing slides",
          items: {
            type: "object",
            properties: {
              title: { type: "string" },
              content: { type: "string" }
            }
          }
        }
      },
      required: ["presentationId", "slides"]
    }
  },
  {
    name: "getGoogleSlidesContent",
    description: "Get content of Google Slides with element IDs for formatting",
    inputSchema: {
      type: "object",
      properties: {
        presentationId: { type: "string", description: "Presentation ID" },
        slideIndex: { type: "number", description: "Specific slide index (optional)" }
      },
      required: ["presentationId"]
    }
  },
  {
    name: "formatGoogleSlidesText",
    description: "Apply text formatting to elements in Google Slides",
    inputSchema: {
      type: "object",
      properties: {
        presentationId: { type: "string", description: "Presentation ID" },
        objectId: { type: "string", description: "Object ID of the text element" },
        startIndex: { type: "number", description: "Start index (0-based)" },
        endIndex: { type: "number", description: "End index (0-based)" },
        bold: { type: "boolean", description: "Make text bold" },
        italic: { type: "boolean", description: "Make text italic" },
        underline: { type: "boolean", description: "Underline text" },
        strikethrough: { type: "boolean", description: "Strikethrough text" },
        fontSize: { type: "number", description: "Font size in points" },
        fontFamily: { type: "string", description: "Font family name" },
        foregroundColor: {
          type: "object",
          description: "Text color (RGB values 0-1)",
          properties: {
            red: { type: "number" },
            green: { type: "number" },
            blue: { type: "number" }
          }
        }
      },
      required: ["presentationId", "objectId"]
    }
  },
  {
    name: "formatGoogleSlidesParagraph",
    description: "Apply paragraph formatting to text in Google Slides",
    inputSchema: {
      type: "object",
      properties: {
        presentationId: { type: "string", description: "Presentation ID" },
        objectId: { type: "string", description: "Object ID of the text element" },
        alignment: {
          type: "string",
          description: "Text alignment",
          enum: ["START", "CENTER", "END", "JUSTIFIED"]
        },
        lineSpacing: { type: "number", description: "Line spacing multiplier" },
        bulletStyle: {
          type: "string",
          description: "Bullet style",
          enum: ["NONE", "DISC", "ARROW", "SQUARE", "DIAMOND", "STAR", "NUMBERED"]
        }
      },
      required: ["presentationId", "objectId"]
    }
  },
  {
    name: "styleGoogleSlidesShape",
    description: "Style shapes in Google Slides",
    inputSchema: {
      type: "object",
      properties: {
        presentationId: { type: "string", description: "Presentation ID" },
        objectId: { type: "string", description: "Shape object ID" },
        backgroundColor: {
          type: "object",
          description: "Background color (RGBA values 0-1)",
          properties: {
            red: { type: "number" },
            green: { type: "number" },
            blue: { type: "number" },
            alpha: { type: "number" }
          }
        },
        outlineColor: {
          type: "object",
          description: "Outline color (RGB values 0-1)",
          properties: {
            red: { type: "number" },
            green: { type: "number" },
            blue: { type: "number" }
          }
        },
        outlineWeight: { type: "number", description: "Outline thickness in points" },
        outlineDashStyle: {
          type: "string",
          description: "Outline dash style",
          enum: ["SOLID", "DOT", "DASH", "DASH_DOT", "LONG_DASH", "LONG_DASH_DOT"]
        }
      },
      required: ["presentationId", "objectId"]
    }
  },
  {
    name: "setGoogleSlidesBackground",
    description: "Set background color for slides",
    inputSchema: {
      type: "object",
      properties: {
        presentationId: { type: "string", description: "Presentation ID" },
        pageObjectIds: {
          type: "array",
          description: "Array of slide IDs to update",
          items: { type: "string" }
        },
        backgroundColor: {
          type: "object",
          description: "Background color (RGBA values 0-1)",
          properties: {
            red: { type: "number" },
            green: { type: "number" },
            blue: { type: "number" },
            alpha: { type: "number" }
          }
        }
      },
      required: ["presentationId", "pageObjectIds", "backgroundColor"]
    }
  },
  {
    name: "createGoogleSlidesTextBox",
    description: "Create a text box in Google Slides",
    inputSchema: {
      type: "object",
      properties: {
        presentationId: { type: "string", description: "Presentation ID" },
        pageObjectId: { type: "string", description: "Slide ID" },
        text: { type: "string", description: "Text content" },
        x: { type: "number", description: "X position in EMU (1/360000 cm)" },
        y: { type: "number", description: "Y position in EMU" },
        width: { type: "number", description: "Width in EMU" },
        height: { type: "number", description: "Height in EMU" },
        fontSize: { type: "number", description: "Font size in points" },
        bold: { type: "boolean", description: "Make text bold" },
        italic: { type: "boolean", description: "Make text italic" }
      },
      required: ["presentationId", "pageObjectId", "text", "x", "y", "width", "height"]
    }
  },
  {
    name: "createGoogleSlidesShape",
    description: "Create a shape in Google Slides",
    inputSchema: {
      type: "object",
      properties: {
        presentationId: { type: "string", description: "Presentation ID" },
        pageObjectId: { type: "string", description: "Slide ID" },
        shapeType: {
          type: "string",
          description: "Shape type",
          enum: ["RECTANGLE", "ELLIPSE", "DIAMOND", "TRIANGLE", "STAR", "ROUND_RECTANGLE", "ARROW"]
        },
        x: { type: "number", description: "X position in EMU" },
        y: { type: "number", description: "Y position in EMU" },
        width: { type: "number", description: "Width in EMU" },
        height: { type: "number", description: "Height in EMU" },
        backgroundColor: {
          type: "object",
          description: "Fill color (RGBA values 0-1)",
          properties: {
            red: { type: "number" },
            green: { type: "number" },
            blue: { type: "number" },
            alpha: { type: "number" }
          }
        }
      },
      required: ["presentationId", "pageObjectId", "shapeType", "x", "y", "width", "height"]
    }
  },
  {
    name: "getGoogleSlidesSpeakerNotes",
    description: "Get speaker notes from a specific slide in Google Slides",
    inputSchema: {
      type: "object",
      properties: {
        presentationId: { type: "string", description: "Presentation ID" },
        slideIndex: { type: "number", description: "Slide index (0-based)" }
      },
      required: ["presentationId", "slideIndex"]
    }
  },
  {
    name: "updateGoogleSlidesSpeakerNotes",
    description: "Update speaker notes for a specific slide in Google Slides",
    inputSchema: {
      type: "object",
      properties: {
        presentationId: { type: "string", description: "Presentation ID" },
        slideIndex: { type: "number", description: "Slide index (0-based)" },
        notes: { type: "string", description: "Speaker notes content" }
      },
      required: ["presentationId", "slideIndex", "notes"]
    }
  },
  {
    name: "deleteGoogleSlide",
    description: "Delete a slide from a presentation",
    inputSchema: {
      type: "object",
      properties: {
        presentationId: { type: "string", description: "Presentation ID" },
        slideObjectId: { type: "string", description: "Slide object ID" }
      },
      required: ["presentationId", "slideObjectId"]
    }
  },
  {
    name: "duplicateSlide",
    description: "Duplicate a slide in a presentation",
    inputSchema: {
      type: "object",
      properties: {
        presentationId: { type: "string", description: "Presentation ID" },
        slideObjectId: { type: "string", description: "Slide object ID" }
      },
      required: ["presentationId", "slideObjectId"]
    }
  },
  {
    name: "reorderSlides",
    description: "Reorder one or more slides in a presentation",
    inputSchema: {
      type: "object",
      properties: {
        presentationId: { type: "string", description: "Presentation ID" },
        slideObjectIds: { type: "array", items: { type: "string" }, description: "Slide object IDs to move" },
        insertionIndex: { type: "number", description: "Target insertion index" }
      },
      required: ["presentationId", "slideObjectIds", "insertionIndex"]
    }
  },
  {
    name: "replaceAllTextInSlides",
    description: "Replace all matching text across presentation slides",
    inputSchema: {
      type: "object",
      properties: {
        presentationId: { type: "string", description: "Presentation ID" },
        containsText: { type: "string", description: "Text to find" },
        replaceText: { type: "string", description: "Replacement text" },
        matchCase: { type: "boolean", description: "Case-sensitive match" }
      },
      required: ["presentationId", "containsText", "replaceText"]
    }
  },
  {
    name: "exportSlideThumbnail",
    description: "Export a slide thumbnail URL",
    inputSchema: {
      type: "object",
      properties: {
        presentationId: { type: "string", description: "Presentation ID" },
        slideObjectId: { type: "string", description: "Slide object ID" },
        mimeType: { type: "string", enum: ["PNG", "JPEG"], description: "Thumbnail MIME type" },
        size: { type: "string", enum: ["SMALL", "MEDIUM", "LARGE"], description: "Thumbnail size" }
      },
      required: ["presentationId", "slideObjectId"]
    }
  },
  {
    name: "insertSlidesImageFromUrl",
    description: "Insert an image into a Google Slides slide from a publicly accessible URL",
    inputSchema: {
      type: "object",
      properties: {
        presentationId: { type: "string", description: "Presentation ID" },
        pageObjectId: { type: "string", description: "Slide/page object ID to insert the image into" },
        imageUrl: { type: "string", description: "Publicly accessible URL of the image" },
        x: { type: "number", description: "X position in EMU (default: 0)" },
        y: { type: "number", description: "Y position in EMU (default: 0)" },
        width: { type: "number", description: "Width in EMU (omit to auto-size)" },
        height: { type: "number", description: "Height in EMU (omit to auto-size)" }
      },
      required: ["presentationId", "pageObjectId", "imageUrl"]
    }
  },
  {
    name: "getSlideElementInfo",
    description: "Get position, size, and transform of all elements on a slide. Returns actual rendered bounds.",
    inputSchema: {
      type: "object",
      properties: {
        presentationId: { type: "string", description: "Presentation ID" },
        slideObjectId: { type: "string", description: "Slide object ID (omit to get all slides)" }
      },
      required: ["presentationId"]
    }
  },
  {
    name: "moveSlideElement",
    description: "Move and/or resize an element (image, text box, shape) on a Google Slides slide",
    inputSchema: {
      type: "object",
      properties: {
        presentationId: { type: "string", description: "Presentation ID" },
        objectId: { type: "string", description: "Element object ID to move/resize" },
        x: { type: "number", description: "New X position in EMU" },
        y: { type: "number", description: "New Y position in EMU" },
        width: { type: "number", description: "New width in EMU" },
        height: { type: "number", description: "New height in EMU" }
      },
      required: ["presentationId", "objectId"]
    }
  },
  {
    name: "deleteSlideElement",
    description: "Delete an element (image, text box, shape) from a Google Slides slide",
    inputSchema: {
      type: "object",
      properties: {
        presentationId: { type: "string", description: "Presentation ID" },
        objectId: { type: "string", description: "Element object ID to delete" }
      },
      required: ["presentationId", "objectId"]
    }
  },
  {
    name: "insertSlidesLocalImage",
    description: "Upload a local image file to Google Drive and insert it into a Google Slides slide",
    inputSchema: {
      type: "object",
      properties: {
        presentationId: { type: "string", description: "Presentation ID" },
        pageObjectId: { type: "string", description: "Slide/page object ID to insert the image into" },
        localImagePath: { type: "string", description: "Absolute path to the local image file" },
        x: { type: "number", description: "X position in EMU (default: 0)" },
        y: { type: "number", description: "Y position in EMU (default: 0)" },
        width: { type: "number", description: "Width in EMU (omit to auto-size)" },
        height: { type: "number", description: "Height in EMU (omit to auto-size)" }
      },
      required: ["presentationId", "pageObjectId", "localImagePath"]
    }
  },
  {
    name: "setSlideVisibility",
    description: "Show or hide (skip) slides in one atomic call. Hidden slides are skipped in present mode and PDF export.",
    inputSchema: {
      type: "object",
      properties: {
        presentationId: { type: "string", description: "Presentation ID" },
        slideObjectIds: { type: "array", items: { type: "string" }, description: "Slide object IDs to change" },
        skipped: { type: "boolean", description: "true = hide (skip) the slides, false = show them" }
      },
      required: ["presentationId", "slideObjectIds", "skipped"]
    }
  },
  {
    name: "replaceSlideImage",
    description: "Replace an existing image in place, preserving its layering (z-order), position, size, and crop. Unlike delete+insert, which lands the new image on top of the stack, this keeps the image under any overlaid text.",
    inputSchema: {
      type: "object",
      properties: {
        presentationId: { type: "string", description: "Presentation ID" },
        imageObjectId: { type: "string", description: "Object ID of the image to replace (from getSlideElementInfo)" },
        imageUrl: { type: "string", description: "Publicly accessible URL of the replacement image (use this OR localImagePath)" },
        localImagePath: { type: "string", description: "Absolute path to a local replacement image (uploaded to Drive briefly, then cleaned up)" },
        replaceMethod: { type: "string", enum: ["CENTER_INSIDE", "CENTER_CROP"], description: "How the new image fits the existing bounds (default: CENTER_INSIDE)" }
      },
      required: ["presentationId", "imageObjectId"]
    }
  },
  {
    name: "setElementZOrder",
    description: "Change the stacking order of page elements (updatePageElementsZOrder). Inserted images land on top of the stack and can cover overlay text — send them backward or to the back. All elements must be on the same page.",
    inputSchema: {
      type: "object",
      properties: {
        presentationId: { type: "string", description: "Presentation ID" },
        pageElementObjectIds: { type: "array", items: { type: "string" }, description: "Element object IDs to reorder (same page)" },
        operation: { type: "string", enum: ["BRING_TO_FRONT", "BRING_FORWARD", "SEND_BACKWARD", "SEND_TO_BACK"], description: "Z-order operation applied to the elements" }
      },
      required: ["presentationId", "pageElementObjectIds", "operation"]
    }
  },
  {
    name: "setElementText",
    description: "Replace the entire text of one shape/text box by objectId (element-scoped delete+insert in one atomic call). Unlike replaceAllTextInSlides, this cannot hit other slides, masters, or layouts, and does not depend on matching existing text. Pass empty text to clear the element.",
    inputSchema: {
      type: "object",
      properties: {
        presentationId: { type: "string", description: "Presentation ID" },
        objectId: { type: "string", description: "Object ID of the shape or text box (from getSlideElementInfo / getGoogleSlidesContent)" },
        text: { type: "string", description: "New text content (empty string clears the element)" }
      },
      required: ["presentationId", "objectId", "text"]
    }
  },
];

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

export async function handleTool(
  toolName: string,
  args: Record<string, unknown>,
  ctx: ToolContext
): Promise<ToolResult | null> {
  switch (toolName) {

    case "createGoogleSlides": {
      const validation = CreateGoogleSlidesSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      const parentFolderId = await ctx.resolveFolderId(a.parentFolderId);

      // Check if presentation already exists
      const existingFileId = await ctx.checkFileExists(a.name, parentFolderId);
      if (existingFileId) {
        return errorResponse(
          `A presentation named "${a.name}" already exists in this location. ` +
          `File ID: ${existingFileId}. To modify it, you can use Google Slides directly.`
        );
      }

      const slidesService = ctx.google.slides({ version: 'v1', auth: ctx.authClient });
      const presentation = await slidesService.presentations.create({
        requestBody: { title: a.name },
      });

      await ctx.getDrive().files.update({
        fileId: presentation.data.presentationId!,
        addParents: parentFolderId,
        removeParents: 'root',
        supportsAllDrives: true
      });

      for (const slide of a.slides) {
        const slideObjectId = `slide_${uuidv4().substring(0, 8)}`;
        await slidesService.presentations.batchUpdate({
          presentationId: presentation.data.presentationId!,
          requestBody: {
            requests: [{
              createSlide: {
                objectId: slideObjectId,
                slideLayoutReference: { predefinedLayout: 'TITLE_AND_BODY' },
              }
            }]
          },
        });

        const slidePage = await slidesService.presentations.pages.get({
          presentationId: presentation.data.presentationId!,
          pageObjectId: slideObjectId,
        });

        let titlePlaceholderId = '';
        let bodyPlaceholderId = '';
        slidePage.data.pageElements?.forEach((el) => {
          if (el.shape?.placeholder?.type === 'TITLE') {
            titlePlaceholderId = el.objectId!;
          } else if (el.shape?.placeholder?.type === 'BODY') {
            bodyPlaceholderId = el.objectId!;
          }
        });

        await slidesService.presentations.batchUpdate({
          presentationId: presentation.data.presentationId!,
          requestBody: {
            requests: [
              { insertText: { objectId: titlePlaceholderId, text: slide.title, insertionIndex: 0 } },
              { insertText: { objectId: bodyPlaceholderId, text: slide.content, insertionIndex: 0 } }
            ]
          },
        });
      }

      return {
        content: [{
          type: 'text',
          text: `Created Google Slides presentation: ${a.name}\nID: ${presentation.data.presentationId}\nLink: https://docs.google.com/presentation/d/${presentation.data.presentationId}`,
        }],
        isError: false,
      };
    }

    case "updateGoogleSlides": {
      const validation = UpdateGoogleSlidesSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      const slidesService = ctx.google.slides({ version: 'v1', auth: ctx.authClient });

      // Get current presentation details
      const currentPresentation = await slidesService.presentations.get({
        presentationId: a.presentationId
      });

      if (!currentPresentation.data.slides) {
        return errorResponse("No slides found in presentation");
      }

      // Collect all slide IDs except the first one (we'll keep it for now)
      const slideIdsToDelete = currentPresentation.data.slides
        .slice(1)
        .map(slide => slide.objectId)
        .filter((id): id is string => id !== undefined);

      // Prepare requests to update presentation
      const requests: any[] = [];

      // Delete all slides except the first one
      if (slideIdsToDelete.length > 0) {
        slideIdsToDelete.forEach(slideId => {
          requests.push({
            deleteObject: { objectId: slideId }
          });
        });
      }

      // Now we need to update the first slide or create new slides
      if (a.slides.length === 0) {
        return errorResponse("At least one slide must be provided");
      }

      // Clear content of the first slide
      const firstSlide = currentPresentation.data.slides[0];
      if (firstSlide && firstSlide.pageElements) {
        // Find text elements to clear
        firstSlide.pageElements.forEach(element => {
          if (element.objectId && element.shape?.text) {
            requests.push({
              deleteText: {
                objectId: element.objectId,
                textRange: { type: 'ALL' }
              }
            });
          }
        });
      }

      // Update the first slide with new content
      const firstSlideContent = a.slides[0];
      if (firstSlide && firstSlide.pageElements) {
        // Find title and body placeholders
        let titlePlaceholderId: string | undefined;
        let bodyPlaceholderId: string | undefined;

        firstSlide.pageElements.forEach(element => {
          if (element.shape?.placeholder?.type === 'TITLE' || element.shape?.placeholder?.type === 'CENTERED_TITLE') {
            titlePlaceholderId = element.objectId || undefined;
          } else if (element.shape?.placeholder?.type === 'BODY' || element.shape?.placeholder?.type === 'SUBTITLE') {
            bodyPlaceholderId = element.objectId || undefined;
          }
        });

        if (titlePlaceholderId) {
          requests.push({
            insertText: {
              objectId: titlePlaceholderId,
              text: firstSlideContent.title,
              insertionIndex: 0
            }
          });
        }

        if (bodyPlaceholderId) {
          requests.push({
            insertText: {
              objectId: bodyPlaceholderId,
              text: firstSlideContent.content,
              insertionIndex: 0
            }
          });
        }
      }

      // Add any additional slides from the request
      for (let i = 1; i < a.slides.length; i++) {
        const slideId = `slide_${Date.now()}_${i}`;

        requests.push({
          createSlide: {
            objectId: slideId,
            slideLayoutReference: {
              predefinedLayout: 'TITLE_AND_BODY'
            }
          }
        });

        // We'll need to add content to these slides in a separate batch update
        // because we need to wait for the slides to be created first
      }

      // Execute the batch update
      await slidesService.presentations.batchUpdate({
        presentationId: a.presentationId,
        requestBody: { requests }
      });

      // If we have additional slides, add their content
      if (a.slides.length > 1) {
        const contentRequests: any[] = [];

        // Get updated presentation to find the new slide IDs
        const updatedPresentation = await slidesService.presentations.get({
          presentationId: a.presentationId
        });

        // Add content to the new slides (starting from the second slide in our args)
        for (let i = 1; i < a.slides.length && updatedPresentation.data.slides; i++) {
          const slide = a.slides[i];
          const presentationSlide = updatedPresentation.data.slides[i];

          if (presentationSlide && presentationSlide.pageElements) {
            presentationSlide.pageElements.forEach(element => {
              if (element.objectId) {
                if (element.shape?.placeholder?.type === 'TITLE' || element.shape?.placeholder?.type === 'CENTERED_TITLE') {
                  contentRequests.push({
                    insertText: {
                      objectId: element.objectId,
                      text: slide.title,
                      insertionIndex: 0
                    }
                  });
                } else if (element.shape?.placeholder?.type === 'BODY' || element.shape?.placeholder?.type === 'SUBTITLE') {
                  contentRequests.push({
                    insertText: {
                      objectId: element.objectId,
                      text: slide.content,
                      insertionIndex: 0
                    }
                  });
                }
              }
            });
          }
        }

        if (contentRequests.length > 0) {
          await slidesService.presentations.batchUpdate({
            presentationId: a.presentationId,
            requestBody: { requests: contentRequests }
          });
        }
      }

      return {
        content: [{
          type: 'text',
          text: `Updated Google Slides presentation with ${a.slides.length} slide(s)\nLink: https://docs.google.com/presentation/d/${a.presentationId}`,
        }],
        isError: false,
      };
    }

    case "getGoogleSlidesContent": {
      const validation = GetGoogleSlidesContentSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      const slidesService = ctx.google.slides({ version: 'v1', auth: ctx.authClient });
      const presentation = await slidesService.presentations.get({
        presentationId: a.presentationId
      });

      if (!presentation.data.slides) {
        return errorResponse("No slides found in presentation");
      }

      let content = 'Presentation content with element IDs:\n\n';
      const slides = a.slideIndex !== undefined
        ? [presentation.data.slides[a.slideIndex]]
        : presentation.data.slides;

      slides.forEach((slide, index) => {
        if (!slide || !slide.objectId) return;

        content += `\nSlide ${a.slideIndex ?? index} (ID: ${slide.objectId}):\n`;
        content += '----------------------------\n';

        if (slide.pageElements) {
          slide.pageElements.forEach((element) => {
            if (!element.objectId) return;

            if (element.shape?.text) {
              content += `  Text Box (ID: ${element.objectId}):\n`;
              const textElements = element.shape.text.textElements || [];
              let text = '';
              textElements.forEach((textElement) => {
                if (textElement.textRun?.content) {
                  text += textElement.textRun.content;
                }
              });
              content += `    "${text.trim()}"\n`;
            } else if (element.shape) {
              content += `  Shape (ID: ${element.objectId}): ${element.shape.shapeType || 'Unknown'}\n`;
            } else if (element.image) {
              content += `  Image (ID: ${element.objectId})\n`;
            } else if (element.video) {
              content += `  Video (ID: ${element.objectId})\n`;
            } else if (element.table) {
              content += `  Table (ID: ${element.objectId})\n`;
            }
          });
        }
      });

      return {
        content: [{ type: "text", text: content }],
        isError: false
      };
    }

    case "formatGoogleSlidesText": {
      const validation = FormatGoogleSlidesTextSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      const slidesService = ctx.google.slides({ version: 'v1', auth: ctx.authClient });
      const textStyle: any = {};
      const fields: string[] = [];

      if (a.bold !== undefined) {
        textStyle.bold = a.bold;
        fields.push('bold');
      }

      if (a.italic !== undefined) {
        textStyle.italic = a.italic;
        fields.push('italic');
      }

      if (a.underline !== undefined) {
        textStyle.underline = a.underline;
        fields.push('underline');
      }

      if (a.strikethrough !== undefined) {
        textStyle.strikethrough = a.strikethrough;
        fields.push('strikethrough');
      }

      if (a.fontSize !== undefined) {
        textStyle.fontSize = {
          magnitude: a.fontSize,
          unit: 'PT'
        };
        fields.push('fontSize');
      }

      if (a.fontFamily !== undefined) {
        textStyle.fontFamily = a.fontFamily;
        fields.push('fontFamily');
      }

      if (a.foregroundColor) {
        textStyle.foregroundColor = {
          opaqueColor: {
            rgbColor: {
              red: a.foregroundColor.red || 0,
              green: a.foregroundColor.green || 0,
              blue: a.foregroundColor.blue || 0
            }
          }
        };
        fields.push('foregroundColor');
      }

      if (fields.length === 0) {
        return errorResponse("No formatting options specified");
      }

      const updateRequest: any = {
        updateTextStyle: {
          objectId: a.objectId,
          style: textStyle,
          fields: fields.join(',')
        }
      };

      // Add text range if specified
      if (a.startIndex !== undefined && a.endIndex !== undefined) {
        updateRequest.updateTextStyle.textRange = {
          type: 'FIXED_RANGE',
          startIndex: a.startIndex,
          endIndex: a.endIndex
        };
      } else {
        updateRequest.updateTextStyle.textRange = { type: 'ALL' };
      }

      await slidesService.presentations.batchUpdate({
        presentationId: a.presentationId,
        requestBody: { requests: [updateRequest] }
      });

      return {
        content: [{ type: "text", text: `Applied text formatting to object ${a.objectId}` }],
        isError: false
      };
    }

    case "formatGoogleSlidesParagraph": {
      const validation = FormatGoogleSlidesParagraphSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      const slidesService = ctx.google.slides({ version: 'v1', auth: ctx.authClient });
      const requests: any[] = [];

      if (a.alignment) {
        requests.push({
          updateParagraphStyle: {
            objectId: a.objectId,
            style: { alignment: a.alignment },
            fields: 'alignment'
          }
        });
      }

      if (a.lineSpacing !== undefined) {
        requests.push({
          updateParagraphStyle: {
            objectId: a.objectId,
            style: { lineSpacing: a.lineSpacing },
            fields: 'lineSpacing'
          }
        });
      }

      if (a.bulletStyle) {
        if (a.bulletStyle === 'NONE') {
          requests.push({
            deleteParagraphBullets: {
              objectId: a.objectId
            }
          });
        } else if (a.bulletStyle === 'NUMBERED') {
          requests.push({
            createParagraphBullets: {
              objectId: a.objectId,
              bulletPreset: 'NUMBERED_DIGIT_ALPHA_ROMAN'
            }
          });
        } else {
          requests.push({
            createParagraphBullets: {
              objectId: a.objectId,
              bulletPreset: `BULLET_${a.bulletStyle}_CIRCLE_SQUARE`
            }
          });
        }
      }

      if (requests.length === 0) {
        return errorResponse("No formatting options specified");
      }

      await slidesService.presentations.batchUpdate({
        presentationId: a.presentationId,
        requestBody: { requests }
      });

      return {
        content: [{ type: "text", text: `Applied paragraph formatting to object ${a.objectId}` }],
        isError: false
      };
    }

    case "styleGoogleSlidesShape": {
      const validation = StyleGoogleSlidesShapeSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      const slidesService = ctx.google.slides({ version: 'v1', auth: ctx.authClient });
      const shapeProperties: any = {};
      const fields: string[] = [];

      if (a.backgroundColor) {
        shapeProperties.shapeBackgroundFill = {
          solidFill: {
            color: {
              rgbColor: {
                red: a.backgroundColor.red || 0,
                green: a.backgroundColor.green || 0,
                blue: a.backgroundColor.blue || 0
              }
            },
            alpha: a.backgroundColor.alpha || 1
          }
        };
        fields.push('shapeBackgroundFill');
      }

      const outline: any = {};
      let hasOutlineChanges = false;

      if (a.outlineColor) {
        outline.outlineFill = {
          solidFill: {
            color: {
              rgbColor: {
                red: a.outlineColor.red || 0,
                green: a.outlineColor.green || 0,
                blue: a.outlineColor.blue || 0
              }
            }
          }
        };
        hasOutlineChanges = true;
      }

      if (a.outlineWeight !== undefined) {
        outline.weight = {
          magnitude: a.outlineWeight,
          unit: 'PT'
        };
        hasOutlineChanges = true;
      }

      if (a.outlineDashStyle !== undefined) {
        outline.dashStyle = a.outlineDashStyle;
        hasOutlineChanges = true;
      }

      if (hasOutlineChanges) {
        shapeProperties.outline = outline;
        fields.push('outline');
      }

      if (fields.length === 0) {
        return errorResponse("No styling options specified");
      }

      await slidesService.presentations.batchUpdate({
        presentationId: a.presentationId,
        requestBody: {
          requests: [{
            updateShapeProperties: {
              objectId: a.objectId,
              shapeProperties,
              fields: fields.join(',')
            }
          }]
        }
      });

      return {
        content: [{ type: "text", text: `Applied styling to shape ${a.objectId}` }],
        isError: false
      };
    }

    case "setGoogleSlidesBackground": {
      const validation = SetGoogleSlidesBackgroundSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      const slidesService = ctx.google.slides({ version: 'v1', auth: ctx.authClient });
      const requests = a.pageObjectIds.map(pageObjectId => ({
        updatePageProperties: {
          objectId: pageObjectId,
          pageProperties: {
            pageBackgroundFill: {
              solidFill: {
                color: {
                  rgbColor: {
                    red: a.backgroundColor.red || 0,
                    green: a.backgroundColor.green || 0,
                    blue: a.backgroundColor.blue || 0
                  }
                },
                alpha: a.backgroundColor.alpha || 1
              }
            }
          },
          fields: 'pageBackgroundFill'
        }
      }));

      await slidesService.presentations.batchUpdate({
        presentationId: a.presentationId,
        requestBody: { requests }
      });

      return {
        content: [{ type: "text", text: `Set background color for ${a.pageObjectIds.length} slide(s)` }],
        isError: false
      };
    }

    case "createGoogleSlidesTextBox": {
      const validation = CreateGoogleSlidesTextBoxSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      const slidesService = ctx.google.slides({ version: 'v1', auth: ctx.authClient });
      const elementId = `textBox_${uuidv4().substring(0, 8)}`;

      const requests: any[] = [
        {
          createShape: {
            objectId: elementId,
            shapeType: 'TEXT_BOX',
            elementProperties: {
              pageObjectId: a.pageObjectId,
              size: {
                width: { magnitude: a.width, unit: 'EMU' },
                height: { magnitude: a.height, unit: 'EMU' }
              },
              transform: {
                scaleX: 1,
                scaleY: 1,
                translateX: a.x,
                translateY: a.y,
                unit: 'EMU'
              }
            }
          }
        },
        {
          insertText: {
            objectId: elementId,
            text: a.text,
            insertionIndex: 0
          }
        }
      ];

      // Apply optional formatting
      if (a.fontSize || a.bold || a.italic) {
        const textStyle: any = {};
        const fields: string[] = [];

        if (a.fontSize) {
          textStyle.fontSize = {
            magnitude: a.fontSize,
            unit: 'PT'
          };
          fields.push('fontSize');
        }

        if (a.bold !== undefined) {
          textStyle.bold = a.bold;
          fields.push('bold');
        }

        if (a.italic !== undefined) {
          textStyle.italic = a.italic;
          fields.push('italic');
        }

        if (fields.length > 0) {
          requests.push({
            updateTextStyle: {
              objectId: elementId,
              style: textStyle,
              fields: fields.join(','),
              textRange: { type: 'ALL' }
            }
          });
        }
      }

      await slidesService.presentations.batchUpdate({
        presentationId: a.presentationId,
        requestBody: { requests }
      });

      return {
        content: [{ type: "text", text: `Created text box with ID: ${elementId}` }],
        isError: false
      };
    }

    case "createGoogleSlidesShape": {
      const validation = CreateGoogleSlidesShapeSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      const slidesService = ctx.google.slides({ version: 'v1', auth: ctx.authClient });
      const elementId = `shape_${uuidv4().substring(0, 8)}`;

      const createRequest: any = {
        createShape: {
          objectId: elementId,
          shapeType: a.shapeType,
          elementProperties: {
            pageObjectId: a.pageObjectId,
            size: {
              width: { magnitude: a.width, unit: 'EMU' },
              height: { magnitude: a.height, unit: 'EMU' }
            },
            transform: {
              scaleX: 1,
              scaleY: 1,
              translateX: a.x,
              translateY: a.y,
              unit: 'EMU'
            }
          }
        }
      };

      const requests = [createRequest];

      // Apply background color if specified
      if (a.backgroundColor) {
        requests.push({
          updateShapeProperties: {
            objectId: elementId,
            shapeProperties: {
              shapeBackgroundFill: {
                solidFill: {
                  color: {
                    rgbColor: {
                      red: a.backgroundColor.red || 0,
                      green: a.backgroundColor.green || 0,
                      blue: a.backgroundColor.blue || 0
                    }
                  },
                  alpha: a.backgroundColor.alpha || 1
                }
              }
            },
            fields: 'shapeBackgroundFill'
          }
        });
      }

      await slidesService.presentations.batchUpdate({
        presentationId: a.presentationId,
        requestBody: { requests }
      });

      return {
        content: [{ type: "text", text: `Created ${a.shapeType} shape with ID: ${elementId}` }],
        isError: false
      };
    }

    case "getGoogleSlidesSpeakerNotes": {
      const validation = GetGoogleSlidesSpeakerNotesSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      const slidesService = ctx.google.slides({ version: 'v1', auth: ctx.authClient });

      // Get the presentation to access the slide
      const presentation = await slidesService.presentations.get({
        presentationId: a.presentationId
      });

      if (!presentation.data.slides || a.slideIndex >= presentation.data.slides.length) {
        return errorResponse(`Slide index ${a.slideIndex} not found in presentation (has ${presentation.data.slides?.length ?? 0} slides)`);
      }

      const slide = presentation.data.slides[a.slideIndex];

      // Get the notes page object ID from the slide properties
      const notesObjectId = slide.slideProperties?.notesPage?.notesProperties?.speakerNotesObjectId;

      if (!notesObjectId) {
        return {
          content: [{ type: "text", text: "No speaker notes found for this slide" }],
          isError: false
        };
      }

      // Get the notes page to read the speaker notes text
      const notesPageObjectId = slide.slideProperties?.notesPage?.objectId;
      if (!notesPageObjectId) {
        return {
          content: [{ type: "text", text: "No speaker notes found for this slide" }],
          isError: false
        };
      }

      // Find the notes page for this slide
      const notesPage = presentation.data.slides?.[a.slideIndex]?.slideProperties?.notesPage;

      if (!notesPage || !notesPage.pageElements) {
        return {
          content: [{ type: "text", text: "No speaker notes found for this slide" }],
          isError: false
        };
      }

      // Find the speaker notes shape
      const speakerNotesElement = notesPage.pageElements.find(
        element => element.objectId === notesObjectId
      );

      if (!speakerNotesElement || !speakerNotesElement.shape?.text) {
        return {
          content: [{ type: "text", text: "No speaker notes found for this slide" }],
          isError: false
        };
      }

      // Extract the text from the speaker notes
      let notesText = '';
      const textElements = speakerNotesElement.shape.text.textElements || [];
      textElements.forEach((textElement) => {
        if (textElement.textRun?.content) {
          notesText += textElement.textRun.content;
        }
      });

      return {
        content: [{ type: "text", text: notesText.trim() || "No speaker notes found for this slide" }],
        isError: false
      };
    }

    case "updateGoogleSlidesSpeakerNotes": {
      const validation = UpdateGoogleSlidesSpeakerNotesSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const a = validation.data;

      const slidesService = ctx.google.slides({ version: 'v1', auth: ctx.authClient });

      // Get the presentation to access the slide
      const presentation = await slidesService.presentations.get({
        presentationId: a.presentationId
      });

      if (!presentation.data.slides || a.slideIndex >= presentation.data.slides.length) {
        return errorResponse(`Slide index ${a.slideIndex} not found in presentation (has ${presentation.data.slides?.length ?? 0} slides)`);
      }

      const slide = presentation.data.slides[a.slideIndex];

      // Get the notes page object ID from the slide properties
      const notesObjectId = slide.slideProperties?.notesPage?.notesProperties?.speakerNotesObjectId;

      if (!notesObjectId) {
        return errorResponse("This slide does not have a speaker notes object. Speaker notes may need to be initialized manually in Google Slides first.");
      }

      // Create the batchUpdate request to replace the speaker notes text
      // Only delete existing text if there is any — deleteText with type:'ALL' fails on empty notes
      const notesPage = slide.slideProperties?.notesPage;
      const speakerNotesShape = notesPage?.pageElements?.find(
        (el: any) => el.objectId === notesObjectId
      );
      const existingTextElements = speakerNotesShape?.shape?.text?.textElements || [];
      const hasExistingText = existingTextElements.some(
        (el: any) => el.textRun?.content
      );

      const requests: any[] = [];

      if (hasExistingText) {
        requests.push({ deleteText: { objectId: notesObjectId, textRange: { type: 'ALL' } } });
      }

      requests.push({ insertText: { objectId: notesObjectId, text: a.notes, insertionIndex: 0 } });

      await slidesService.presentations.batchUpdate({
        presentationId: a.presentationId,
        requestBody: { requests }
      });

      return {
        content: [{ type: "text", text: `Successfully updated speaker notes for slide ${a.slideIndex}` }],
        isError: false
      };
    }

    case "deleteGoogleSlide": {
      const validation = DeleteGoogleSlideSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const a = validation.data;

      const slidesService = ctx.google.slides({ version: 'v1', auth: ctx.authClient });
      await slidesService.presentations.batchUpdate({
        presentationId: a.presentationId,
        requestBody: {
          requests: [{ deleteObject: { objectId: a.slideObjectId } }]
        }
      });

      return {
        content: [{ type: 'text', text: `Deleted slide ${a.slideObjectId}` }],
        isError: false,
      };
    }

    case "duplicateSlide": {
      const validation = DuplicateSlideSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const a = validation.data;

      const slidesService = ctx.google.slides({ version: 'v1', auth: ctx.authClient });
      const response = await slidesService.presentations.batchUpdate({
        presentationId: a.presentationId,
        requestBody: {
          requests: [{ duplicateObject: { objectId: a.slideObjectId } }]
        }
      });

      const dupId = response.data.replies?.[0]?.duplicateObject?.objectId;
      return {
        content: [{ type: 'text', text: `Duplicated slide ${a.slideObjectId}${dupId ? ` -> ${dupId}` : ''}` }],
        isError: false,
      };
    }

    case "reorderSlides": {
      const validation = ReorderSlidesSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const a = validation.data;

      const slidesService = ctx.google.slides({ version: 'v1', auth: ctx.authClient });
      await slidesService.presentations.batchUpdate({
        presentationId: a.presentationId,
        requestBody: {
          requests: [{
            updateSlidesPosition: {
              slideObjectIds: a.slideObjectIds,
              insertionIndex: a.insertionIndex,
            }
          }]
        }
      });

      return {
        content: [{ type: 'text', text: `Reordered ${a.slideObjectIds.length} slide(s) to index ${a.insertionIndex}` }],
        isError: false,
      };
    }

    case "replaceAllTextInSlides": {
      const validation = ReplaceAllTextInSlidesSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const a = validation.data;

      const slidesService = ctx.google.slides({ version: 'v1', auth: ctx.authClient });
      const response = await slidesService.presentations.batchUpdate({
        presentationId: a.presentationId,
        requestBody: {
          requests: [{
            replaceAllText: {
              containsText: {
                text: a.containsText,
                matchCase: a.matchCase,
              },
              replaceText: a.replaceText,
            }
          }]
        }
      });

      const count = response.data.replies?.[0]?.replaceAllText?.occurrencesChanged ?? 0;
      return {
        content: [{ type: 'text', text: `Replaced ${count} occurrence(s) of "${a.containsText}" in slides.` }],
        isError: false,
      };
    }

    case "exportSlideThumbnail": {
      const validation = ExportSlideThumbnailSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const a = validation.data;

      const slidesService = ctx.google.slides({ version: 'v1', auth: ctx.authClient });
      const response = await slidesService.presentations.pages.getThumbnail({
        presentationId: a.presentationId,
        pageObjectId: a.slideObjectId,
        'thumbnailProperties.mimeType': a.mimeType,
        'thumbnailProperties.thumbnailSize': a.size,
      });

      const url = response.data?.contentUrl;
      if (!url) return errorResponse('No thumbnail URL returned by Google Slides API.');

      return {
        content: [{ type: 'text', text: `Slide thumbnail URL (${a.mimeType}, ${a.size}): ${url}` }],
        isError: false,
      };
    }

    case "getSlideElementInfo": {
      const validation = GetSlideElementInfoSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const a = validation.data;

      const slidesService = ctx.google.slides({ version: 'v1', auth: ctx.authClient });

      // Page size lives on the presentation; grab just that field.
      const sizeOnly = await slidesService.presentations.get({
        presentationId: a.presentationId,
        fields: 'pageSize',
      });
      const slideWidth = sizeOnly.data.pageSize?.width?.magnitude || 9144000;
      const slideHeight = sizeOnly.data.pageSize?.height?.magnitude || 6858000;

      // If a specific slide is requested, fetch just that page. Otherwise fetch
      // only the fields of the slides we actually render.
      let slides: slides_v1.Schema$Page[] = [];
      if (a.slideObjectId) {
        const page = await slidesService.presentations.pages.get({
          presentationId: a.presentationId,
          pageObjectId: a.slideObjectId,
          fields: 'objectId,pageElements(objectId,transform,size,shape/shapeType,image)',
        });
        slides = [page.data];
      } else {
        const withSlides = await slidesService.presentations.get({
          presentationId: a.presentationId,
          fields: 'slides(objectId,pageElements(objectId,transform,size,shape/shapeType,image))',
        });
        slides = withSlides.data.slides || [];
      }

      const lines: string[] = [`Slide dimensions: ${slideWidth} x ${slideHeight} EMU`];
      for (const slide of slides) {
        lines.push(`\n--- Slide: ${slide.objectId} ---`);
        for (const el of slide.pageElements || []) {
          const t = el.transform || {};
          const s = el.size || {};
          const intrW = s.width?.magnitude || 0;
          const intrH = s.height?.magnitude || 0;
          const scX = t.scaleX || 1;
          const scY = t.scaleY || 1;
          const tx = t.translateX || 0;
          const ty = t.translateY || 0;
          const renderedW = intrW * scX;
          const renderedH = intrH * scY;
          const right = tx + renderedW;
          const bottom = ty + renderedH;
          const offPage = (tx < 0 || ty < 0 || right > slideWidth || bottom > slideHeight)
            ? ' *** OFF PAGE ***' : '';
          lines.push(`  ${el.objectId} (${el.shape ? 'shape:' + el.shape.shapeType : el.image ? 'image' : 'other'})`);
          lines.push(`    intrinsic: ${intrW} x ${intrH}, scale: ${scX} x ${scY}`);
          lines.push(`    rendered: ${Math.round(renderedW)} x ${Math.round(renderedH)} at (${tx}, ${ty})`);
          lines.push(`    bounds: right=${Math.round(right)}, bottom=${Math.round(bottom)}${offPage}`);
        }
      }

      return { content: [{ type: 'text', text: lines.join('\n') }], isError: false };
    }

    case "moveSlideElement": {
      const validation = MoveSlideElementSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const a = validation.data;

      const slidesService = ctx.google.slides({ version: 'v1', auth: ctx.authClient });

      // Field-masked fetch: we only need each element's objectId/transform/size.
      const pres = await slidesService.presentations.get({
        presentationId: a.presentationId,
        fields: 'slides(pageElements(objectId,transform,size))',
      });

      let currentTransform: slides_v1.Schema$AffineTransform | null = null;
      let currentSize: slides_v1.Schema$Size | null = null;
      for (const slide of pres.data.slides || []) {
        for (const el of slide.pageElements || []) {
          if (el.objectId === a.objectId) {
            currentTransform = el.transform || null;
            currentSize = el.size || null;
            break;
          }
        }
        if (currentTransform) break;
      }

      if (!currentTransform) {
        return errorResponse(`Element ${a.objectId} not found in presentation`);
      }

      const origWidth = currentSize?.width?.magnitude || 3000000;
      const origHeight = currentSize?.height?.magnitude || 3000000;
      const curScaleX = currentTransform.scaleX || 1;
      const curScaleY = currentTransform.scaleY || 1;

      const newScaleX = a.width !== undefined ? a.width / origWidth : curScaleX;
      const newScaleY = a.height !== undefined ? a.height / origHeight : curScaleY;
      const newX = a.x ?? (currentTransform.translateX || 0);
      const newY = a.y ?? (currentTransform.translateY || 0);

      const newTransform: slides_v1.Schema$AffineTransform = {
        scaleX: newScaleX,
        scaleY: newScaleY,
        translateX: newX,
        translateY: newY,
        shearX: currentTransform.shearX || 0,
        shearY: currentTransform.shearY || 0,
        unit: 'EMU',
      };

      const requests: slides_v1.Schema$Request[] = [{
        updatePageElementTransform: {
          objectId: a.objectId,
          applyMode: 'ABSOLUTE',
          transform: newTransform,
        },
      }];

      await slidesService.presentations.batchUpdate({
        presentationId: a.presentationId,
        requestBody: { requests },
      });

      const didResize = a.width !== undefined || a.height !== undefined;
      const action = didResize ? 'Moved/resized' : 'Moved';
      return {
        content: [{ type: 'text', text: `${action} element ${a.objectId} to (${newX}, ${newY})` }],
        isError: false,
      };
    }

    case "deleteSlideElement": {
      const validation = DeleteSlideElementSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const a = validation.data;

      const slidesService = ctx.google.slides({ version: 'v1', auth: ctx.authClient });
      await slidesService.presentations.batchUpdate({
        presentationId: a.presentationId,
        requestBody: {
          requests: [{ deleteObject: { objectId: a.objectId } }],
        },
      });

      return {
        content: [{ type: 'text', text: `Deleted element ${a.objectId}` }],
        isError: false,
      };
    }

    case "insertSlidesImageFromUrl": {
      const validation = InsertSlidesImageFromUrlSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const a = validation.data;
      return insertImageIntoSlide(ctx, a.presentationId, a.pageObjectId, a.imageUrl, a.x, a.y, a.width, a.height);
    }

    case "insertSlidesLocalImage": {
      const validation = InsertSlidesLocalImageSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const a = validation.data;

      // Upload briefly as public so the Slides API can fetch the bytes.
      const { fileId, webContentLink } = await uploadImageToDrive(ctx, a.localImagePath, {
        makePublic: true,
      });
      try {
        const result = await insertImageIntoSlide(
          ctx, a.presentationId, a.pageObjectId, webContentLink,
          a.x, a.y, a.width, a.height,
        );
        // Slides stores its own copy once createImage returns, so the intermediary
        // Drive file is no longer referenced. Delete it (which also removes the
        // public permission we just granted).
        await deleteDriveFile(ctx, fileId).catch((err) =>
          ctx.log(`insertSlidesLocalImage: failed to delete intermediary Drive file ${fileId}`, err),
        );
        return result;
      } catch (err) {
        // On embed failure, still try to clean up the uploaded file.
        await deleteDriveFile(ctx, fileId).catch(() => {});
        throw err;
      }
    }

    case "setSlideVisibility": {
      const validation = SetSlideVisibilitySchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const a = validation.data;

      const slidesService = ctx.google.slides({ version: 'v1', auth: ctx.authClient });
      await slidesService.presentations.batchUpdate({
        presentationId: a.presentationId,
        requestBody: {
          requests: a.slideObjectIds.map((objectId) => ({
            updateSlideProperties: {
              objectId,
              slideProperties: { isSkipped: a.skipped },
              fields: 'isSkipped',
            },
          })),
        },
      });

      return {
        content: [{ type: 'text', text: `${a.skipped ? 'Hid' : 'Unhid'} ${a.slideObjectIds.length} slide(s): ${a.slideObjectIds.join(', ')}` }],
        isError: false,
      };
    }

    case "replaceSlideImage": {
      const validation = ReplaceSlideImageSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const a = validation.data;

      const slidesService = ctx.google.slides({ version: 'v1', auth: ctx.authClient });
      const doReplace = async (url: string) => slidesService.presentations.batchUpdate({
        presentationId: a.presentationId,
        requestBody: {
          requests: [{
            replaceImage: {
              imageObjectId: a.imageObjectId,
              url,
              imageReplaceMethod: a.replaceMethod,
            },
          }],
        },
      });

      if (a.localImagePath) {
        // Upload briefly as public so the Slides API can fetch the bytes, then
        // clean up — Slides stores its own copy once the request returns.
        const { fileId, webContentLink } = await uploadImageToDrive(ctx, a.localImagePath, {
          makePublic: true,
        });
        try {
          await doReplace(webContentLink);
        } finally {
          await deleteDriveFile(ctx, fileId).catch((err) =>
            ctx.log(`replaceSlideImage: failed to delete intermediary Drive file ${fileId}`, err),
          );
        }
      } else {
        await doReplace(a.imageUrl!);
      }

      return {
        content: [{ type: 'text', text: `Replaced image ${a.imageObjectId} in place (${a.replaceMethod}) — layering, position, size, and crop preserved` }],
        isError: false,
      };
    }

    case "setElementZOrder": {
      const validation = SetElementZOrderSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const a = validation.data;

      const slidesService = ctx.google.slides({ version: 'v1', auth: ctx.authClient });
      await slidesService.presentations.batchUpdate({
        presentationId: a.presentationId,
        requestBody: {
          requests: [{
            updatePageElementsZOrder: {
              pageElementObjectIds: a.pageElementObjectIds,
              operation: a.operation,
            },
          }],
        },
      });

      return {
        content: [{ type: 'text', text: `Applied ${a.operation} to ${a.pageElementObjectIds.length} element(s): ${a.pageElementObjectIds.join(', ')}` }],
        isError: false,
      };
    }

    case "setElementText": {
      const validation = SetElementTextSchema.safeParse(args);
      if (!validation.success) return errorResponse(validation.error.errors[0].message);
      const a = validation.data;

      const slidesService = ctx.google.slides({ version: 'v1', auth: ctx.authClient });

      // deleteText on an element with no existing text is an API error, so
      // check whether the shape currently has any. Search slides, layouts, and
      // masters (recursing into groups) — the setter works on any of them.
      const pres = await slidesService.presentations.get({
        presentationId: a.presentationId,
        fields: 'slides(pageElements(objectId,shape/text,elementGroup)),layouts(pageElements(objectId,shape/text,elementGroup)),masters(pageElements(objectId,shape/text,elementGroup))',
      });

      let found = false;
      let hasText = false;
      const visit = (elements: any[] | undefined) => {
        for (const el of elements || []) {
          if (el.objectId === a.objectId) {
            found = true;
            // autoText elements (slide numbers, dates) carry no textRun but are
            // still text: counting only textRuns would read the shape as empty,
            // skip deleteText, and insert the new copy in front of the
            // auto-text rather than replacing it.
            const elements: any[] = el.shape?.text?.textElements || [];
            const content = elements.map((te: any) => te.textRun?.content || '').join('');
            hasText = content.replace(/\n$/, '').length > 0 || elements.some((te: any) => te.autoText);
            return;
          }
          if (el.elementGroup?.children) visit(el.elementGroup.children);
          if (found) return;
        }
      };
      for (const page of [...(pres.data.slides || []), ...(pres.data.layouts || []), ...(pres.data.masters || [])]) {
        visit(page.pageElements as any[]);
        if (found) break;
      }

      if (!found) {
        return errorResponse(`Element ${a.objectId} not found in presentation (checked slides, layouts, and masters)`);
      }

      const requests: any[] = [];
      if (hasText) {
        requests.push({ deleteText: { objectId: a.objectId, textRange: { type: 'ALL' } } });
      }
      if (a.text.length > 0) {
        requests.push({ insertText: { objectId: a.objectId, insertionIndex: 0, text: a.text } });
      }
      if (requests.length === 0) {
        return {
          content: [{ type: 'text', text: `Element ${a.objectId} already has no text — nothing to do` }],
          isError: false,
        };
      }

      await slidesService.presentations.batchUpdate({
        presentationId: a.presentationId,
        requestBody: { requests },
      });

      const action = a.text.length === 0 ? 'Cleared text of' : `Set text of`;
      return {
        content: [{ type: 'text', text: `${action} element ${a.objectId}${a.text.length ? ` (${a.text.length} chars)` : ''}` }],
        isError: false,
      };
    }

    default:
      return null;
  }
}
