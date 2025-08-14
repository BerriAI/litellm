// test.js
const WebSocket = require("ws");

const url = "ws://localhost:4000/v1/realtime?model=gpt-4o-mini-realtime-preview";

const ws = new WebSocket(url, {
    headers: {
        "api-key": `sk-1234`,
        "OpenAI-Beta": "realtime=v1",
    },
});


ws.on("open", function open() {
    console.log("Connected to server.");
    ws.send(JSON.stringify({
        type: "response.create",
        response: {
            modalities: ["text"],
            instructions: "Please assist the user.",
        }
    }));
});

ws.on("message", function incoming(message) {
    console.log(JSON.parse(message.toString()));
});

ws.on("error", function handleError(error) {
    console.error("Error: ", error);
});