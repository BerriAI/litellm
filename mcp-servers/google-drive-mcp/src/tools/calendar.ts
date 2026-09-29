import { z } from 'zod';
import { buildCalendarEventUpdate } from '../utils.js';
import { errorResponse } from '../types.js';
import type { ToolDefinition, ToolResult, ToolContext } from '../types.js';

// ---------------------------------------------------------------------------
// Interfaces
// ---------------------------------------------------------------------------

interface CalendarEventInfo {
  id: string;
  summary?: string;
  description?: string;
  location?: string;
  start?: { dateTime?: string; date?: string; timeZone?: string };
  end?: { dateTime?: string; date?: string; timeZone?: string };
  status?: string;
  htmlLink?: string;
  hangoutLink?: string;
  meetingLink?: string;
  attendees?: { email: string; displayName?: string; responseStatus?: string }[];
  organizer?: { email?: string; displayName?: string };
  recurrence?: string[];
  attachments?: { fileUrl?: string; title?: string; mimeType?: string; fileId?: string; iconLink?: string }[];
  created?: string;
  updated?: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatCalendarEvent(event: any): CalendarEventInfo {
  const result: CalendarEventInfo = {
    id: event.id || '',
    summary: event.summary,
    description: event.description,
    location: event.location,
    status: event.status,
    htmlLink: event.htmlLink,
    created: event.created,
    updated: event.updated,
  };

  if (event.start) {
    result.start = {
      dateTime: event.start.dateTime,
      date: event.start.date,
      timeZone: event.start.timeZone,
    };
  }

  if (event.end) {
    result.end = {
      dateTime: event.end.dateTime,
      date: event.end.date,
      timeZone: event.end.timeZone,
    };
  }

  if (event.hangoutLink) {
    result.hangoutLink = event.hangoutLink;
  }

  if (event.conferenceData?.entryPoints) {
    const videoEntry = event.conferenceData.entryPoints.find((ep: any) => ep.entryPointType === 'video');
    if (videoEntry?.uri) {
      result.meetingLink = videoEntry.uri;
    }
  }

  if (event.attendees) {
    result.attendees = event.attendees.map((a: any) => ({
      email: a.email || '',
      displayName: a.displayName,
      responseStatus: a.responseStatus,
    }));
  }

  if (event.organizer) {
    result.organizer = {
      email: event.organizer.email,
      displayName: event.organizer.displayName,
    };
  }

  if (event.recurrence) {
    result.recurrence = event.recurrence;
  }

  if (event.attachments) {
    result.attachments = event.attachments.map((a: any) => ({
      fileUrl: a.fileUrl,
      title: a.title,
      mimeType: a.mimeType,
      fileId: a.fileId,
      iconLink: a.iconLink,
    }));
  }

  return result;
}

function formatEventForDisplay(event: CalendarEventInfo): string {
  const lines: string[] = [];
  lines.push(`**${event.summary || '(No title)'}**`);

  if (event.start) {
    const startStr = event.start.dateTime || event.start.date || '';
    const endStr = event.end?.dateTime || event.end?.date || '';
    if (event.start.date) {
      // All-day event
      lines.push(`Date: ${startStr}${endStr && endStr !== startStr ? ` - ${endStr}` : ''}`);
    } else {
      lines.push(`Time: ${startStr} - ${endStr}`);
    }
  }

  if (event.location) lines.push(`Location: ${event.location}`);
  if (event.description) lines.push(`Description: ${event.description}`);
  if (event.hangoutLink || event.meetingLink) {
    lines.push(`Meeting: ${event.meetingLink || event.hangoutLink}`);
  }
  if (event.attendees && event.attendees.length > 0) {
    lines.push(`Attendees: ${event.attendees.map(a => a.email).join(', ')}`);
  }
  if (event.attachments && event.attachments.length > 0) {
    const items = event.attachments.map(a => {
      if (a.title && a.fileUrl) return `${a.title} (${a.fileUrl})`;
      return a.title || a.fileUrl || '(untitled)';
    });
    lines.push(`Attachments: ${items.join(', ')}`);
  }
  if (event.htmlLink) lines.push(`Link: ${event.htmlLink}`);
  lines.push(`Event ID: ${event.id}`);

  return lines.join('\n');
}

// ---------------------------------------------------------------------------
// Zod Schemas
// ---------------------------------------------------------------------------

const ListCalendarsSchema = z.object({
  showHidden: z.boolean().optional().default(false).describe("Include hidden calendars")
});

const GetCalendarEventsSchema = z.object({
  calendarId: z.string().optional().default("primary").describe("Calendar ID (default: primary)"),
  timeMin: z.string().optional().describe("Start of time range (RFC3339, e.g., '2024-01-01T00:00:00Z')"),
  timeMax: z.string().optional().describe("End of time range (RFC3339)"),
  query: z.string().optional().describe("Free text search in events"),
  maxResults: z.number().int().min(1).max(250).optional().default(50).describe("Maximum events to return (1-250)"),
  singleEvents: z.boolean().optional().default(true).describe("Expand recurring events into instances"),
  orderBy: z.enum(["startTime", "updated"]).optional().default("startTime").describe("Sort order")
});

const GetCalendarEventSchema = z.object({
  eventId: z.string().min(1, "Event ID is required"),
  calendarId: z.string().optional().default("primary").describe("Calendar ID (default: primary)")
});

// Google Calendar allows at most 25 attachments per event. `fileUrl` is the
// only required input; `fileId`/`mimeType` are output-only and derived by the
// API (Drive files: use the Drive file's share URL as `fileUrl`).
const AttachmentInputSchema = z
  .array(
    z.object({
      fileUrl: z.string().min(1, "Attachment fileUrl is required").describe("URL of the file to attach; for Drive files use the file's share URL"),
      title: z.string().optional().describe("Attachment title"),
      mimeType: z.string().optional().describe("MIME type (optional; ignored for Drive files, which the API resolves)"),
    }),
  )
  .max(25, "A calendar event can have at most 25 attachments")
  .optional()
  .describe("File attachments for the event (max 25). On update this replaces existing attachments; pass an empty array to remove all of them, or omit to keep them.");

const CreateCalendarEventSchema = z.object({
  summary: z.string().min(1, "Event title is required"),
  calendarId: z.string().optional().default("primary").describe("Calendar ID (default: primary)"),
  description: z.string().optional().describe("Event description"),
  location: z.string().optional().describe("Event location"),
  start: z.object({
    dateTime: z.string().optional().describe("RFC3339 timestamp for timed events"),
    date: z.string().optional().describe("Date for all-day events (YYYY-MM-DD)"),
    timeZone: z.string().optional().describe("Time zone (e.g., 'America/Los_Angeles')")
  }).describe("Start time"),
  end: z.object({
    dateTime: z.string().optional().describe("RFC3339 timestamp for timed events"),
    date: z.string().optional().describe("Date for all-day events (YYYY-MM-DD)"),
    timeZone: z.string().optional().describe("Time zone (e.g., 'America/Los_Angeles')")
  }).describe("End time"),
  attendees: z.array(z.string()).optional().describe("Email addresses of attendees"),
  sendUpdates: z.enum(["all", "externalOnly", "none"]).optional().default("none").describe("Send notifications to attendees (default: none)"),
  conferenceType: z.enum(["hangoutsMeet"]).optional().describe("Add Google Meet link"),
  recurrence: z.array(z.string()).optional().describe("RRULE strings for recurring events"),
  visibility: z.enum(["default", "public", "private", "confidential"]).optional().describe("Event visibility"),
  attachments: AttachmentInputSchema
});

const UpdateCalendarEventSchema = z.object({
  eventId: z.string().min(1, "Event ID is required"),
  calendarId: z.string().optional().default("primary").describe("Calendar ID (default: primary)"),
  summary: z.string().optional().describe("New event title"),
  description: z.string().optional().describe("New event description"),
  location: z.string().optional().describe("New event location"),
  start: z.object({
    dateTime: z.string().optional(),
    date: z.string().optional(),
    timeZone: z.string().optional()
  }).optional().describe("New start time"),
  end: z.object({
    dateTime: z.string().optional(),
    date: z.string().optional(),
    timeZone: z.string().optional()
  }).optional().describe("New end time"),
  attendees: z.array(z.string()).optional().describe("Updated attendee emails (replaces existing)"),
  attachments: AttachmentInputSchema,
  sendUpdates: z.enum(["all", "externalOnly", "none"]).optional().default("none").describe("Send notifications about the update (default: none)")
});

const DeleteCalendarEventSchema = z.object({
  eventId: z.string().min(1, "Event ID is required"),
  calendarId: z.string().optional().default("primary").describe("Calendar ID (default: primary)"),
  sendUpdates: z.enum(["all", "externalOnly", "none"]).optional().default("none").describe("Send cancellation notifications to attendees (default: none)")
});

// ---------------------------------------------------------------------------
// Tool Definitions
// ---------------------------------------------------------------------------

export const toolDefinitions: ToolDefinition[] = [
  {
    name: "listCalendars",
    description: "List all accessible Google Calendars for the authenticated user",
    inputSchema: {
      type: "object",
      properties: {
        showHidden: { type: "boolean", description: "Include hidden calendars (default: false)" }
      }
    }
  },
  {
    name: "getCalendarEvents",
    description: "Get events from a Google Calendar with optional filtering",
    inputSchema: {
      type: "object",
      properties: {
        calendarId: { type: "string", description: "Calendar ID (default: primary)" },
        timeMin: { type: "string", description: "Start of time range (RFC3339, e.g., '2024-01-01T00:00:00Z')" },
        timeMax: { type: "string", description: "End of time range (RFC3339)" },
        query: { type: "string", description: "Free text search in events" },
        maxResults: { type: "number", description: "Maximum events to return (1-250, default: 50)" },
        singleEvents: { type: "boolean", description: "Expand recurring events into instances (default: true)" },
        orderBy: { type: "string", enum: ["startTime", "updated"], description: "Sort order (default: startTime)" }
      }
    }
  },
  {
    name: "getCalendarEvent",
    description: "Get a single calendar event by ID",
    inputSchema: {
      type: "object",
      properties: {
        eventId: { type: "string", description: "The event ID to retrieve" },
        calendarId: { type: "string", description: "Calendar ID (default: primary)" }
      },
      required: ["eventId"]
    }
  },
  {
    name: "createCalendarEvent",
    description: "Create a new calendar event. Supports timed events, all-day events, and Google Meet integration.",
    inputSchema: {
      type: "object",
      properties: {
        summary: { type: "string", description: "Event title" },
        calendarId: { type: "string", description: "Calendar ID (default: primary)" },
        description: { type: "string", description: "Event description" },
        location: { type: "string", description: "Event location" },
        start: {
          type: "object",
          description: "Start time (use dateTime for timed events, date for all-day)",
          properties: {
            dateTime: { type: "string", description: "RFC3339 timestamp (e.g., '2024-01-15T09:00:00-08:00')" },
            date: { type: "string", description: "Date for all-day events (YYYY-MM-DD)" },
            timeZone: { type: "string", description: "Time zone (e.g., 'America/Los_Angeles')" }
          }
        },
        end: {
          type: "object",
          description: "End time",
          properties: {
            dateTime: { type: "string" },
            date: { type: "string" },
            timeZone: { type: "string" }
          }
        },
        attendees: { type: "array", items: { type: "string" }, description: "Email addresses of attendees" },
        sendUpdates: { type: "string", enum: ["all", "externalOnly", "none"], description: "Send notifications (default: none)" },
        conferenceType: { type: "string", enum: ["hangoutsMeet"], description: "Add Google Meet link" },
        recurrence: { type: "array", items: { type: "string" }, description: "RRULE strings for recurring events" },
        visibility: { type: "string", enum: ["default", "public", "private", "confidential"], description: "Event visibility" },
        attachments: {
          type: "array",
          maxItems: 25,
          description: "File attachments (max 25). For Drive files use the file's share URL as fileUrl.",
          items: {
            type: "object",
            properties: {
              fileUrl: { type: "string", description: "URL of the file to attach (required)" },
              title: { type: "string", description: "Attachment title" },
              mimeType: { type: "string", description: "MIME type (optional; ignored for Drive files)" }
            },
            required: ["fileUrl"]
          }
        }
      },
      required: ["summary", "start", "end"]
    }
  },
  {
    name: "updateCalendarEvent",
    description: "Update an existing calendar event",
    inputSchema: {
      type: "object",
      properties: {
        eventId: { type: "string", description: "The event ID to update" },
        calendarId: { type: "string", description: "Calendar ID (default: primary)" },
        summary: { type: "string", description: "New event title" },
        description: { type: "string", description: "New event description" },
        location: { type: "string", description: "New event location" },
        start: {
          type: "object",
          properties: {
            dateTime: { type: "string" },
            date: { type: "string" },
            timeZone: { type: "string" }
          }
        },
        end: {
          type: "object",
          properties: {
            dateTime: { type: "string" },
            date: { type: "string" },
            timeZone: { type: "string" }
          }
        },
        attendees: { type: "array", items: { type: "string" }, description: "Updated attendee emails (replaces existing)" },
        attachments: {
          type: "array",
          maxItems: 25,
          description: "File attachments (max 25). Replaces existing attachments; pass an empty array to remove all, or omit to keep them. For Drive files use the file's share URL as fileUrl.",
          items: {
            type: "object",
            properties: {
              fileUrl: { type: "string", description: "URL of the file to attach (required)" },
              title: { type: "string", description: "Attachment title" },
              mimeType: { type: "string", description: "MIME type (optional; ignored for Drive files)" }
            },
            required: ["fileUrl"]
          }
        },
        sendUpdates: { type: "string", enum: ["all", "externalOnly", "none"], description: "Send notifications (default: none)" }
      },
      required: ["eventId"]
    }
  },
  {
    name: "deleteCalendarEvent",
    description: "Delete a calendar event",
    inputSchema: {
      type: "object",
      properties: {
        eventId: { type: "string", description: "The event ID to delete" },
        calendarId: { type: "string", description: "Calendar ID (default: primary)" },
        sendUpdates: { type: "string", enum: ["all", "externalOnly", "none"], description: "Send cancellation notifications (default: none)" }
      },
      required: ["eventId"]
    }
  }
];

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

export async function handleTool(
  toolName: string,
  args: Record<string, unknown>,
  ctx: ToolContext,
): Promise<ToolResult | null> {
  switch (toolName) {
    case "listCalendars": {
      const validation = ListCalendarsSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const parsed = validation.data;

      const response = await ctx.getCalendar().calendarList.list({
        showHidden: parsed.showHidden,
        maxResults: 250
      });

      const calendars = response.data.items || [];
      if (calendars.length === 0) {
        return { content: [{ type: "text", text: "No calendars found." }], isError: false };
      }

      const lines = calendars.map((cal: any) => {
        const primary = cal.primary ? ' (PRIMARY)' : '';
        const role = cal.accessRole ? ` [${cal.accessRole}]` : '';
        return `- ${cal.summary}${primary}${role}\n  ID: ${cal.id}`;
      });

      return {
        content: [{ type: "text", text: `Found ${calendars.length} calendar(s):\n\n${lines.join('\n\n')}` }],
        isError: false
      };
    }

    case "getCalendarEvents": {
      const validation = GetCalendarEventsSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const parsed = validation.data;

      const params: any = {
        calendarId: parsed.calendarId || 'primary',
        maxResults: parsed.maxResults || 50,
        singleEvents: parsed.singleEvents !== false,
        orderBy: parsed.orderBy || 'startTime'
      };

      if (parsed.timeMin) params.timeMin = parsed.timeMin;
      if (parsed.timeMax) params.timeMax = parsed.timeMax;
      if (parsed.query) params.q = parsed.query;

      const response = await ctx.getCalendar().events.list(params);

      const events = response.data.items || [];
      if (events.length === 0) {
        return { content: [{ type: "text", text: "No events found." }], isError: false };
      }

      const formattedEvents = events.map((e: any) => formatEventForDisplay(formatCalendarEvent(e)));

      return {
        content: [{ type: "text", text: `Found ${events.length} event(s):\n\n${formattedEvents.join('\n\n---\n\n')}` }],
        isError: false
      };
    }

    case "getCalendarEvent": {
      const validation = GetCalendarEventSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const parsed = validation.data;

      const response = await ctx.getCalendar().events.get({
        calendarId: parsed.calendarId || 'primary',
        eventId: parsed.eventId
      });

      const formatted = formatEventForDisplay(formatCalendarEvent(response.data));
      return {
        content: [{ type: "text", text: formatted }],
        isError: false
      };
    }

    case "createCalendarEvent": {
      const validation = CreateCalendarEventSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const parsed = validation.data;

      const eventResource: any = {
        summary: parsed.summary,
        description: parsed.description,
        location: parsed.location,
        start: parsed.start,
        end: parsed.end,
        visibility: parsed.visibility
      };

      if (parsed.attendees && parsed.attendees.length > 0) {
        eventResource.attendees = parsed.attendees.map((email: string) => ({ email }));
      }

      if (parsed.recurrence) {
        eventResource.recurrence = parsed.recurrence;
      }

      if (parsed.attachments && parsed.attachments.length > 0) {
        eventResource.attachments = parsed.attachments;
      }

      let conferenceDataVersion = 0;
      if (parsed.conferenceType === 'hangoutsMeet') {
        eventResource.conferenceData = {
          createRequest: {
            requestId: `meet-${Date.now()}`,
            conferenceSolutionKey: { type: 'hangoutsMeet' }
          }
        };
        conferenceDataVersion = 1;
      }

      const insertParams: any = {
        calendarId: parsed.calendarId || 'primary',
        requestBody: eventResource,
        sendUpdates: parsed.sendUpdates,
        // Required by the Calendar API to persist `attachments`; harmless otherwise.
        supportsAttachments: true
      };

      if (conferenceDataVersion > 0) {
        insertParams.conferenceDataVersion = conferenceDataVersion;
      }

      const response = await ctx.getCalendar().events.insert(insertParams);
      const created = formatCalendarEvent(response.data);

      return {
        content: [{ type: "text", text: `Event created successfully!\n\n${formatEventForDisplay(created)}` }],
        isError: false
      };
    }

    case "updateCalendarEvent": {
      const validation = UpdateCalendarEventSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const parsed = validation.data;

      // First get the existing event
      const existingResponse = await ctx.getCalendar().events.get({
        calendarId: parsed.calendarId || 'primary',
        eventId: parsed.eventId
      });

      const existing = existingResponse.data;
      const eventResource = buildCalendarEventUpdate(existing, parsed);

      const response = await ctx.getCalendar().events.update({
        calendarId: parsed.calendarId || 'primary',
        eventId: parsed.eventId,
        requestBody: eventResource,
        sendUpdates: parsed.sendUpdates,
        // Required so forwarded/overridden `attachments` are persisted rather
        // than wiped; without it the API ignores attachment changes.
        supportsAttachments: true
      });

      const updated = formatCalendarEvent(response.data);

      return {
        content: [{ type: "text", text: `Event updated successfully!\n\n${formatEventForDisplay(updated)}` }],
        isError: false
      };
    }

    case "deleteCalendarEvent": {
      const validation = DeleteCalendarEventSchema.safeParse(args);
      if (!validation.success) {
        return errorResponse(validation.error.errors[0].message);
      }
      const parsed = validation.data;

      await ctx.getCalendar().events.delete({
        calendarId: parsed.calendarId || 'primary',
        eventId: parsed.eventId,
        sendUpdates: parsed.sendUpdates
      });

      return {
        content: [{ type: "text", text: `Event ${parsed.eventId} has been deleted.` }],
        isError: false
      };
    }

    default:
      return null;
  }
}
