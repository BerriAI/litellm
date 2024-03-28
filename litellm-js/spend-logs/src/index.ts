import { serve } from '@hono/node-server'
import { Hono } from 'hono'
import { PrismaClient } from '@prisma/client'
import {LiteLLM_SpendLogs, LiteLLM_IncrementSpend, LiteLLM_IncrementObject} from './_types'

const app = new Hono()
const prisma = new PrismaClient()
// In-memory storage for logs
let spend_logs: LiteLLM_SpendLogs[] = [];
const key_logs: LiteLLM_IncrementObject[] = [];
const user_logs: LiteLLM_IncrementObject[] = [];
const transaction_logs: LiteLLM_IncrementObject[] = [];


app.get('/', (c) => {
  return c.text('Hello Hono!')
})

const MIN_LOGS = 1; // Minimum number of logs needed to initiate a flush
const FLUSH_INTERVAL = 5000; // Time in ms to wait before trying to flush again
const BATCH_SIZE = 100; // Preferred size of each batch to write to the database
const MAX_LOGS_PER_INTERVAL = 1000; // Maximum number of logs to flush in a single interval

const flushLogsToDb = async () => {
  if (spend_logs.length >= MIN_LOGS) {
    // Limit the logs to process in this interval to MAX_LOGS_PER_INTERVAL or less
    const logsToProcess = spend_logs.slice(0, MAX_LOGS_PER_INTERVAL);
  
    for (let i = 0; i < logsToProcess.length; i += BATCH_SIZE) {
      // Create subarray for current batch, ensuring it doesn't exceed the BATCH_SIZE
      const batch = logsToProcess.slice(i, i + BATCH_SIZE);

      // Convert datetime strings to Date objects
      const batchWithDates = batch.map(entry => ({
        ...entry,
        startTime: new Date(entry.startTime),
        endTime: new Date(entry.endTime),
        // Repeat for any other DateTime fields you may have
      }));

      await prisma.liteLLM_SpendLogs.createMany({
        data: batchWithDates,
      });

      console.log(`Flushed ${batch.length} logs to the DB.`);
    }

    // Remove the processed logs from spend_logs
    spend_logs = spend_logs.slice(logsToProcess.length);
    
    console.log(`${logsToProcess.length} logs processed. Remaining in queue: ${spend_logs.length}`);
  } else {
    // This will ensure it doesn't falsely claim "No logs to flush." when it's merely below the MIN_LOGS threshold.
    if(spend_logs.length > 0) {
      console.log(`Accumulating logs. Currently at ${spend_logs.length}, waiting for at least ${MIN_LOGS}.`);
    } else {
      console.log("No logs to flush.");
    }
  }
};

// Setup interval for attempting to flush the logs
setInterval(flushLogsToDb, FLUSH_INTERVAL);

// Route to receive log messages
app.post('/spend/update', async (c) => {
  const incomingLogs = await c.req.json<LiteLLM_SpendLogs[]>();
  
  spend_logs.push(...incomingLogs);

  console.log(`Received and stored ${incomingLogs.length} logs. Total logs in memory: ${spend_logs.length}`);
  
  return c.json({ message: `Successfully stored ${incomingLogs.length} logs` });
});



const port = 3000
console.log(`Server is running on port ${port}`)

serve({
  fetch: app.fetch,
  port
})
