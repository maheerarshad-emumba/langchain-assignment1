import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { createInterface } from "readline";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";  
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrievalChain } from "langchain/chains/retrieval";
import path from 'path';

import * as dotenv from "dotenv";
dotenv.config();

//initialize model
const initializeModel = () => {
    const model = new ChatOpenAI({
        modelName: "gpt-3.5-turbo",
        temperature: 0.7,
    });
    return model;
};

//load and process PDF
const loadAndProcessPDF = async (pdfPath) => {
    const pdfLoader = new PDFLoader(pdfPath);
    const docs = await pdfLoader.load();

    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 200,
        chunkOverlap: 20,
    });

    const splitDocs = await splitter.splitDocuments(docs);
    return splitDocs;
};

//load and process webpage content
const loadAndProcessWebPage = async (url) => {
    const webpageLoader = new CheerioWebBaseLoader(url);
    const docs = await webpageLoader.load();

    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 200,
        chunkOverlap: 20,
    });

    const splitDocs = await splitter.splitDocuments(docs);
    return splitDocs;
};

//create a VectorStore
const createVectorStore = async (splitDocs) => {
    const embeddings = new OpenAIEmbeddings();
    const vectorstore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);
    return vectorstore;
};

//create a Retrieval Chain from Documents
const createRetrievalChainFromDocs = async (model, vectorstore) => {
    const retriever = vectorstore.asRetriever();

    const prompt = ChatPromptTemplate.fromMessages([
        { role: "system", content: "Context: {context}. Answer the user's question based on the context provided. If the question is not related to the context, respond with 'I do not have information about it.'" },
        new MessagesPlaceholder("chat_history"), //placeholder for chat history
        { role: "human", content: "{input}" },   //user's input/question
    ]);

    const chain = await createStuffDocumentsChain({
        llm: model,
        prompt,
    });

    const retrievalChain = await createRetrievalChain({
        combineDocsChain: chain,
        retriever,
    });

    return retrievalChain;
};

//readline for user input
const setupReadlineInterface = () => {
    const rl = createInterface({
        input: process.stdin,
        output: process.stdout,
    });
    return rl;
};

const askQuestion = (rl, retrievalChain, chatHistory) => {
    rl.question("User: ", async(input) => {
        if (input.toLowerCase() === "exit"){
            rl.close();
            return;
        }

        const response = await retrievalChain.invoke({
            input: input,
            chat_history: chatHistory,
        });
        
        console.log("Agent: ", response.answer);
        chatHistory.push(new HumanMessage(input));
        chatHistory.push(new AIMessage(response.answer));

        askQuestion(rl, retrievalChain, chatHistory);
    });
};

const runChatbot = async () => {
    const model = initializeModel();
    
    const pdfPath = path.resolve("C:/Users/Emumba/Documents/qa pathways/assignment1 langchain/Loan Policy of Emumba Inc.pdf");
    const pdfDocs = await loadAndProcessPDF(pdfPath);
    
    const webDocs = await loadAndProcessWebPage("https://emumba.com/");
    
    const allDocs = [...pdfDocs, ...webDocs];
    
    const vectorstore = await createVectorStore(allDocs);
    const retrievalChain = await createRetrievalChainFromDocs(model, vectorstore);
    
    const rl = setupReadlineInterface();
    const chatHistory = [];
    
    askQuestion(rl, retrievalChain, chatHistory);
};

runChatbot();
