// const { GoogleGenerativeAI } = require("@google/generative-ai");

// const genAI = new GoogleGenerativeAI("sk-1234");
// const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

// const prompt = "Explain how AI works in 2 pages";

// async function run() {
//     try {
//         const result = await model.generateContentStream(prompt, { baseUrl: "http://localhost:4000/gemini" });
//         const response = await result.response;
//         console.log(response.text());
//         for await (const chunk of result.stream) {
//             const chunkText = chunk.text();
//             console.log(chunkText);
//             process.stdout.write(chunkText);
//         }
//     } catch (error) {
//         console.error("Error:", error);
//     }
// }

// run();