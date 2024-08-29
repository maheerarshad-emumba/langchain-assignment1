import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import {ConversationChain} from 'langchain/chains';
import readline, { createInterface } from "readline";
import {HumanMessage, AIMessage} from '@langchain/core/messages';
import {TavilySearchResults} from "@langchain/community/tools/tavily_search";
import { createOpenAIFunctionsAgent, AgentExecutor } from "langchain/agents";

import * as dotenv from "dotenv"
dotenv.config();

//initialize model
const initializeModel = () => {
    const model = new ChatOpenAI({
        modelName: "gpt-3.5-turbo",
        temperature: 0.7,
    });
    return model;
};

//create prompt
const createToolandPrompt = async (model) => {
    const prompt = ChatPromptTemplate.fromMessages([
        {role:"system", content:"You are an AI specialized in ansering questions about {subject}. If the question is not related to the specified subject, respond with 'I do not have information about it. '"},
        new MessagesPlaceholder("chat_history"),
        {role:"human", content:"{input}"},
        new MessagesPlaceholder("agent_scratchpad"),
    ]);

    const searchTool = new TavilySearchResults();
    const tools = [searchTool];

    const agent = await createOpenAIFunctionsAgent({
        llm: model,
        prompt,
        tools,
    });

    const agentExecutor = new AgentExecutor({
        agent,
        tools,
    });

    return agentExecutor;
};

//readline intialization for user input
const initializeReadline = () => {
    const rl = createInterface({
        input: process.stdin,
        output: process.stdout,
    });
    return rl;
}; 

const askQuestion = async (rl, subject, agentExecutor, chatHistory) => {
    rl.question("User: ", async(input) => {
        if (input.toLowerCase() === "exit"){
            rl.close();
            return;
        }

        const response = await agentExecutor.invoke({
            input: input,
            chat_history: chatHistory,
            agent_scratchpad: "",
            subject: subject,
        });
        
        console.log("Agent: ", response.output);
        chatHistory.push(new HumanMessage(input));
        chatHistory.push(new AIMessage(response.output));

        askQuestion(rl, subject, agentExecutor, chatHistory);
    });   
};

const runChatbot = async () => {
    const model = initializeModel();
    
    const agentExecutor = await createToolandPrompt(model);

    const rl = initializeReadline();
    const chatHistory = [];
    
    rl.question("Please enter the subject you want to discuss: ", (subject) => {
        askQuestion(rl, subject, agentExecutor, chatHistory);
    });};

runChatbot();
