import { Hono } from 'hono'
import { Context } from 'hono';
import { bearerAuth } from 'hono/bearer-auth'
import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: "sk-1234",
  baseURL: "https://openai-endpoint.ishaanjaffer0324.workers.dev"
});

async function call_proxy() {
  const completion = await openai.chat.completions.create({
    messages: [{ role: "system", content: "You are a helpful assistant." }],
    model: "gpt-3.5-turbo",
  });

  return completion
}

const app = new Hono()

// Middleware for API Key Authentication
const apiKeyAuth = async (c: Context, next: Function) => {
  const apiKey = c.req.header('Authorization');
  if (!apiKey || apiKey !== 'Bearer sk-1234') {
    return c.text('Unauthorized', 401);
  }
  await next();
};


app.use('/*', apiKeyAuth)


app.get('/', (c) => {
  return c.text('Hello Hono!')
})




// Handler for chat completions
const chatCompletionHandler = async (c: Context) => {
  // Assuming your logic for handling chat completion goes here
  // For demonstration, just returning a simple JSON response
  const response = await call_proxy()
  return c.json(response);
};

// Register the above handler for different POST routes with the apiKeyAuth middleware
app.post('/v1/chat/completions', chatCompletionHandler);
app.post('/chat/completions', chatCompletionHandler);

// Example showing how you might handle dynamic segments within the URL
// Here, using ':model*' to capture the rest of the path as a parameter 'model'
app.post('/openai/deployments/:model*/chat/completions', chatCompletionHandler);


export default app
