import assert from 'node:assert/strict';
import { describe, it, before, after, beforeEach } from 'node:test';
import { setupTestServer, callTool, type TestContext } from '../helpers/setup-server.js';

describe('Calendar tools', () => {
  let ctx: TestContext;

  before(async () => { ctx = await setupTestServer(); });
  after(async () => { await ctx.cleanup(); });
  beforeEach(() => {
    ctx.mocks.calendar.tracker.reset();
  });

  // --- listCalendars ---
  describe('listCalendars', () => {
    it('happy path', async () => {
      const res = await callTool(ctx.client, 'listCalendars', {});
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('My Calendar'));
    });

    it('propagates API error', async () => {
      ctx.mocks.calendar.service.calendarList.list._setImpl(async () => { throw new Error('Cal API down'); });
      const res = await callTool(ctx.client, 'listCalendars', {});
      assert.equal(res.isError, true);
      assert.ok(res.content[0].text!.includes('Cal API down'));
      ctx.mocks.calendar.service.calendarList.list._resetImpl();
    });
  });

  // --- getCalendarEvents ---
  describe('getCalendarEvents', () => {
    it('happy path', async () => {
      const res = await callTool(ctx.client, 'getCalendarEvents', {});
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Test Event'));
    });

    it('surfaces attachments for listed events', async () => {
      const res = await callTool(ctx.client, 'getCalendarEvents', {});
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Attachments:'));
      assert.ok(res.content[0].text!.includes('Agenda.pdf'));
      assert.ok(res.content[0].text!.includes('https://drive.google.com/file/d/file-1/view'));
    });
  });

  // --- getCalendarEvent ---
  describe('getCalendarEvent', () => {
    it('happy path', async () => {
      const res = await callTool(ctx.client, 'getCalendarEvent', { eventId: 'event-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Test Event'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'getCalendarEvent', {});
      assert.equal(res.isError, true);
    });

    it('surfaces attachments in the response', async () => {
      const res = await callTool(ctx.client, 'getCalendarEvent', { eventId: 'event-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Attachments:'));
      assert.ok(res.content[0].text!.includes('Agenda.pdf'));
      assert.ok(res.content[0].text!.includes('https://drive.google.com/file/d/file-1/view'));
    });

    it('renders a title-less attachment without duplicating the URL', async () => {
      ctx.mocks.calendar.service.events.get._setImpl(async () => ({
        data: {
          id: 'event-1',
          summary: 'No-title attachment',
          start: { dateTime: '2025-01-01T10:00:00Z' },
          end: { dateTime: '2025-01-01T11:00:00Z' },
          status: 'confirmed',
          attachments: [{ fileUrl: 'https://example.com/f' }],
        },
      }));
      const res = await callTool(ctx.client, 'getCalendarEvent', { eventId: 'event-1' });
      ctx.mocks.calendar.service.events.get._resetImpl();
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('Attachments: https://example.com/f'));
      assert.ok(!res.content[0].text!.includes('https://example.com/f (https://example.com/f)'));
    });
  });

  // --- createCalendarEvent ---
  describe('createCalendarEvent', () => {
    it('happy path', async () => {
      const res = await callTool(ctx.client, 'createCalendarEvent', {
        summary: 'New Event',
        start: { dateTime: '2025-06-01T10:00:00Z' },
        end: { dateTime: '2025-06-01T11:00:00Z' },
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('created'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'createCalendarEvent', {});
      assert.equal(res.isError, true);
    });

    it('forwards attachments and supportsAttachments to events.insert', async () => {
      const res = await callTool(ctx.client, 'createCalendarEvent', {
        summary: 'Event with attachment',
        start: { dateTime: '2025-06-01T10:00:00Z' },
        end: { dateTime: '2025-06-01T11:00:00Z' },
        attachments: [{ fileUrl: 'https://drive.google.com/file/d/abc/view', title: 'Spec' }],
      });
      assert.equal(res.isError, false);
      const call = ctx.mocks.calendar.tracker.getCalls('events.insert').at(-1);
      assert.ok(call, 'events.insert should have been called');
      assert.equal(call!.args[0].supportsAttachments, true);
      assert.deepEqual(call!.args[0].requestBody.attachments, [
        { fileUrl: 'https://drive.google.com/file/d/abc/view', title: 'Spec' },
      ]);
    });

    it('rejects an attachment without fileUrl', async () => {
      const res = await callTool(ctx.client, 'createCalendarEvent', {
        summary: 'Bad attachment',
        start: { dateTime: '2025-06-01T10:00:00Z' },
        end: { dateTime: '2025-06-01T11:00:00Z' },
        attachments: [{ title: 'no url' }],
      });
      assert.equal(res.isError, true);
    });
  });

  // --- updateCalendarEvent ---
  describe('updateCalendarEvent', () => {
    it('happy path', async () => {
      const res = await callTool(ctx.client, 'updateCalendarEvent', {
        eventId: 'event-1', summary: 'Updated Event',
      });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('updated'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'updateCalendarEvent', {});
      assert.equal(res.isError, true);
    });

    it('preserves existing attachments when not overridden', async () => {
      const res = await callTool(ctx.client, 'updateCalendarEvent', {
        eventId: 'event-1', summary: 'Renamed, attachments untouched',
      });
      assert.equal(res.isError, false);
      const call = ctx.mocks.calendar.tracker.getCalls('events.update').at(-1);
      assert.ok(call, 'events.update should have been called');
      assert.equal(call!.args[0].supportsAttachments, true);
      // The existing event (from events.get) carries one attachment — it must
      // be forwarded so the update does not wipe it.
      assert.equal(call!.args[0].requestBody.attachments.length, 1);
      assert.equal(call!.args[0].requestBody.attachments[0].fileId, 'file-1');
    });

    it('replaces attachments when explicitly provided', async () => {
      const res = await callTool(ctx.client, 'updateCalendarEvent', {
        eventId: 'event-1',
        attachments: [{ fileUrl: 'https://drive.google.com/file/d/new/view', title: 'New' }],
      });
      assert.equal(res.isError, false);
      const call = ctx.mocks.calendar.tracker.getCalls('events.update').at(-1);
      assert.deepEqual(call!.args[0].requestBody.attachments, [
        { fileUrl: 'https://drive.google.com/file/d/new/view', title: 'New' },
      ]);
    });

    it('removes all attachments when given an empty array', async () => {
      const res = await callTool(ctx.client, 'updateCalendarEvent', {
        eventId: 'event-1', attachments: [],
      });
      assert.equal(res.isError, false);
      const call = ctx.mocks.calendar.tracker.getCalls('events.update').at(-1);
      assert.equal(call!.args[0].supportsAttachments, true);
      assert.deepEqual(call!.args[0].requestBody.attachments, []);
    });
  });

  // --- deleteCalendarEvent ---
  describe('deleteCalendarEvent', () => {
    it('happy path', async () => {
      const res = await callTool(ctx.client, 'deleteCalendarEvent', { eventId: 'event-1' });
      assert.equal(res.isError, false);
      assert.ok(res.content[0].text!.includes('deleted'));
    });

    it('validation error', async () => {
      const res = await callTool(ctx.client, 'deleteCalendarEvent', {});
      assert.equal(res.isError, true);
    });
  });
});
