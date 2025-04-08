require('dotenv').config();
const express = require('express');
const axios = require('axios');
const { v4: uuidv4 } = require('uuid');

////////////////////////////////////////////////////////////////////////////////////////////////////
// Init Express App
////////////////////////////////////////////////////////////////////////////////////////////////////
const app = express();

// to handle JSON requests in standard OpenAI-like format: { model, messages, ... }
app.use(express.json());

// to handle large request bodies if needed - app.use(express.json({ limit: '50mb' }));

////////////////////////////////////////////////////////////////////////////////////////////////////
// Environment / Configuration
//    Provide all the environment variables for different LLM providers
////////////////////////////////////////////////////////////////////////////////////////////////////
const PORT = process.env.PORT || 4000;

// OpenAI
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || null;

// Azure OpenAI
const AZURE_OPENAI_KEY = process.env.AZURE_OPENAI_KEY || null;
const AZURE_OPENAI_ENDPOINT = process.env.AZURE_OPENAI_ENDPOINT || null;

// Anthropic
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY || null;

// Hugging Face
const HUGGINGFACE_API_KEY = process.env.HUGGINGFACE_API_KEY || null;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Simple Helper: Extract provider from 'model' field
//    Example: "openai/gpt-3.5" => provider = "openai", rawModel = "gpt-3.5"
//             "anthropic/claude-2" => provider = "anthropic", rawModel = "claude-2"
//             "azure/gpt-35-deployment" => provider = "azure", rawModel = "gpt-35-deployment"
////////////////////////////////////////////////////////////////////////////////////////////////////
function parseProviderAndModel(modelString) {
    // If model is "openai/gpt-3.5", split by "/"
    const [maybeProvider, ...rest] = modelString.split('/');
    if (!rest || rest.length === 0) {
        // If we don't find a slash, default to openai
        return { provider: 'openai', rawModel: modelString };
    }
    const provider = maybeProvider.toLowerCase();
    const rawModel = rest.join('/'); // like "gpt-3.5"
    return { provider, rawModel };
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Route: Chat Completions (POST /v1/chat/completions)
//    Accepts an OpenAI-style request body: {
//      model: "openai/gpt-3.5-turbo" or "anthropic/claude-2" or "azure/..." or "huggingface/...",
//      messages: [ { role: "user", content: "Hello" }, ...],
//      stream: Boolean (optional),
//      temperature, top_p, max_tokens, etc...
//    }
//    return an OpenAI-like JSON response or a stream of tokens if "stream" = true
////////////////////////////////////////////////////////////////////////////////////////////////////
app.post('/v1/chat/completions', async (req, res) => {
    // Read entire request body
    const {
        model,
        messages,
        stream,
        temperature,
        top_p,
        max_tokens,
        // can pass any other fields from the OpenAI request: presence_penalty, frequency_penalty, etc.
        // We'll just store them and selectively pass them on to the providers if needed
        ...extraParams
    } = req.body;

    if (!model || !messages) {
        return res.status(400).json({
            error: {
                message: 'Please provide both "model" and "messages" in the request body.',
                code: 'INVALID_REQUEST',
            },
        });
    }

    // Determine which provider is needed
    const { provider, rawModel } = parseProviderAndModel(model);

    // define a handler for each possible provider.
    // If we don't know the provider, we can return an error.
    try {
        if (provider === 'openai') {
            await handleOpenAIChat({
                res,
                rawModel,
                messages,
                stream,
                temperature,
                top_p,
                max_tokens,
                extraParams,
            });
        } else if (provider === 'anthropic') {
            await handleAnthropicChat({
                res,
                rawModel,
                messages,
                stream,
                temperature,
                max_tokens,
                extraParams,
            });
        } else if (provider === 'azure') {
            await handleAzureOpenAIChat({
                res,
                rawModel,
                messages,
                stream,
                temperature,
                top_p,
                max_tokens,
                extraParams,
            });
        } else if (provider === 'huggingface') {
            await handleHuggingFaceChat({
                res,
                rawModel,
                messages,
                stream,
                temperature,
                max_tokens,
                extraParams,
            });
        } else {
            return res.status(400).json({
                error: {
                    message: `Unsupported provider prefix: '${provider}'`,
                    code: 'UNSUPPORTED_PROVIDER',
                },
            });
        }
    } catch (err) {
        console.error('Chat Completion Error:', err?.response?.data || err.message || err);
        res
            .status(500)
            .json({ error: { message: 'Error processing request', details: err.message || err } });
    }
});

////////////////////////////////////////////////////////
// Route: Embeddings (POST /v1/embeddings)
//    Accepts an OpenAI-style request body: { model, input, user?, etc. }
//    We respond with an OpenAI-like embeddings JSON
////////////////////////////////////////////////////////////////
app.post('/v1/embeddings', async (req, res) => {
    const { model, input, user, ...extraParams } = req.body;
    if (!model || !input) {
        return res.status(400).json({
            error: {
                message: 'Please provide "model" and "input" fields in the request body.',
                code: 'INVALID_REQUEST',
            },
        });
    }

    const { provider, rawModel } = parseProviderAndModel(model);

    try {
        if (provider === 'openai') {
            await handleOpenAIEmbeddings({ res, rawModel, input });
        } else if (provider === 'azure') {
            await handleAzureOpenAIEmbeddings({ res, rawModel, input });
        } else if (provider === 'huggingface') {
            await handleHuggingFaceEmbeddings({ res, rawModel, input });
        } else {
            return res.status(400).json({
                error: {
                    message: `Unsupported provider for embeddings: '${provider}'`,
                    code: 'UNSUPPORTED_PROVIDER',
                },
            });
        }
    } catch (err) {
        console.error('Embeddings Error:', err?.response?.data || err.message || err);
        res
            .status(500)
            .json({ error: { message: 'Error processing embeddings request', details: err.message } });
    }
});

////////////////////////////////////////////////////////////////////////////////////
// Route: Image Generation (POST /v1/images/generations)
//    Accepts an OpenAI-style request body: { prompt, n, size, model? ... }
//    We respond with an OpenAI-like JSON containing generated image URLs or base64
////////////////////////////////////////////////////////////////////////////////////
app.post('/v1/images/generations', async (req, res) => {
    const { prompt, n, size, model, ...extraParams } = req.body;
    if (!prompt) {
        return res.status(400).json({
            error: {
                message: 'Please provide "prompt" field in the request body.',
                code: 'INVALID_REQUEST',
            },
        });
    }

    // route to provider based on 'model' if provided, else default to openai if not specified
    let provider = 'openai';
    let rawModel = 'dall-e'; // or something
    if (model) {
        const parsed = parseProviderAndModel(model);
        provider = parsed.provider;
        rawModel = parsed.rawModel;
    }

    try {
        if (provider === 'openai') {
            await handleOpenAIImageGeneration({ res, prompt, n, size });
        } else if (provider === 'azure') {
            // Azure supports DALL-E if your resource is configured for it. Not guaranteed.
            await handleAzureOpenAIImageGeneration({ res, prompt, n, size });
        } else {
            return res.status(400).json({
                error: {
                    message: `Unsupported provider for image generation: '${provider}'`,
                    code: 'UNSUPPORTED_PROVIDER',
                },
            });
        }
    } catch (err) {
        console.error('Image Generation Error:', err?.response?.data || err.message || err);
        res
            .status(500)
            .json({ error: { message: 'Error generating images', details: err.message || err } });
    }
});

////////////////////////////////////////////////////////////////////////////////////////////////////
// Provider Handlers - Chat Completions
//
//    We’ll define handleOpenAIChat, handleAzureOpenAIChat, handleAnthropicChat, handleHuggingFaceChat, etc.
//
//    Each function receives {res, rawModel, messages, stream, temperature, ...} and performs:
//      - transforms the request into that provider’s format
//      - calls the provider’s API
//      - returns an OpenAI-like response
//      - if streaming, streams partial data
////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////
// OpenAI Chat
//////////////////////////////////////
async function handleOpenAIChat({
    res,
    rawModel,
    messages,
    stream,
    temperature,
    top_p,
    max_tokens,
    extraParams,
}) {
    if (!OPENAI_API_KEY) {
        throw new Error('OPENAI_API_KEY not set in .env');
    }

    // Prepare the request body for OpenAI
    const body = {
        model: rawModel,
        messages,
        temperature: temperature == null ? 1.0 : temperature,
        top_p: top_p == null ? 1.0 : top_p,
        max_tokens: max_tokens == null ? 256 : max_tokens,
        stream: !!stream,
        ...extraParams,
        // any additional OpenAI-compatible parameters (like presence_penalty, frequency_penalty, etc.)
    };

    if (!stream) {
        // Non-streaming
        const response = await axios.post(
            'https://api.openai.com/v1/chat/completions',
            body,
            {
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${OPENAI_API_KEY}`,
                },
            }
        );
        // Return as JSON
        return res.json(response.data);
    } else {
        // Streaming
        const streamingResponse = await axios({
            method: 'post',
            url: 'https://api.openai.com/v1/chat/completions',
            data: body,
            responseType: 'stream',
            headers: {
                'Content-Type': 'application/json',
                Authorization: `Bearer ${OPENAI_API_KEY}`,
            },
        });

        // Set appropriate headers
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache, no-transform');
        res.setHeader('Connection', 'keep-alive');

        streamingResponse.data.on('data', (chunk) => {
            // Directly pipe the chunk to client
            res.write(chunk);
        });

        streamingResponse.data.on('end', () => {
            res.end();
        });

        streamingResponse.data.on('error', (err) => {
            console.error('OpenAI Streaming Error:', err);
            res.end();
        });
    }
}

//////////////////////////////////////
// Azure OpenAI Chat
//////////////////////////////////////
async function handleAzureOpenAIChat({
    res,
    rawModel,
    messages,
    stream,
    temperature,
    top_p,
    max_tokens,
    extraParams,
}) {
    // If there is a custom Azure deployment name (rawModel = <deployment-name>),
    // or if rawModel is "gpt-35-turbo" and deployment name is "gpt-35-turbo-deployment"
    if (!AZURE_OPENAI_KEY || !AZURE_OPENAI_ENDPOINT) {
        throw new Error('Azure OpenAI credentials not set in .env');
    }

    const AZURE_API_VERSION = '2023-03-15-preview'; // might differ
    const deploymentName = rawModel; // assume user gave "my-gpt35-deployment"

    const url = `${AZURE_OPENAI_ENDPOINT}/openai/deployments/${deploymentName}/chat/completions?api-version=${AZURE_API_VERSION}`;

    const body = {
        messages,
        temperature: temperature == null ? 1.0 : temperature,
        top_p: top_p == null ? 1.0 : top_p,
        max_tokens: max_tokens == null ? 256 : max_tokens,
        stream: !!stream,
        ...extraParams,
    };

    if (!stream) {
        const response = await axios.post(url, body, {
            headers: {
                'Content-Type': 'application/json',
                'api-key': AZURE_OPENAI_KEY,
            },
        });
        // transform Azure’s response to OpenAI-like response if needed
        // typically Azure’s response is already in an almost identical structure
        return res.json(response.data);
    } else {
        // Streaming
        const streamingResponse = await axios({
            method: 'post',
            url,
            data: body,
            responseType: 'stream',
            headers: {
                'Content-Type': 'application/json',
                'api-key': AZURE_OPENAI_KEY,
            },
        });

        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache, no-transform');
        res.setHeader('Connection', 'keep-alive');

        streamingResponse.data.on('data', (chunk) => {
            res.write(chunk);
        });
        streamingResponse.data.on('end', () => {
            res.end();
        });
        streamingResponse.data.on('error', (err) => {
            console.error('Azure OpenAI Streaming Error:', err);
            res.end();
        });
    }
}

//////////////////////////////////////
// Anthropic Chat (Claude)
//////////////////////////////////////
async function handleAnthropicChat({
    res,
    rawModel,
    messages,
    stream,
    temperature,
    max_tokens,
    extraParams,
}) {
    // Anthropic’s chat is typically done by building a single prompt with “Human:” / “Assistant:”
    // Then POST to https://api.anthropic.com/v1/complete
    // The model is something like "claude-2", "claude-instant-1", etc.

    if (!ANTHROPIC_API_KEY) {
        throw new Error('ANTHROPIC_API_KEY not set in .env');
    }

    // Build the conversation prompt from the OpenAI-like messages
    // For each user message => "Human: <content>\n\n"
    // For each assistant message => "Assistant: <content>\n\n"
    // Then ending with "Assistant: "
    let prompt = '';
    messages.forEach((m) => {
        if (m.role === 'user') {
            prompt += `\n\nHuman: ${m.content}`;
        } else if (m.role === 'assistant') {
            prompt += `\n\nAssistant: ${m.content}`;
        } else if (m.role === 'system') {
            prompt += `\n\nAssistant: ${m.content}`;
        } else {
            // default fallback
            prompt += `\n\nHuman: ${m.content}`;
        }
    });
    prompt += '\n\nAssistant:';

    const body = {
        prompt: prompt,
        model: rawModel,
        max_tokens_to_sample: max_tokens == null ? 256 : max_tokens,
        // Anthropic calls it "temperature", "top_p" is "top_p", etc. 
        temperature: temperature == null ? 1.0 : temperature,
        stream: !!stream,
        // can pass additional Anthropic parameters like "stop_sequences", "top_k", etc.
        ...extraParams,
    };

    const url = 'https://api.anthropic.com/v1/complete';
    if (!stream) {
        // Non-streaming
        const response = await axios.post(url, body, {
            headers: {
                'Content-Type': 'application/json',
                'x-api-key': ANTHROPIC_API_KEY,
            },
        });
        // Convert Anthropic response to OpenAI-like structure
        const openAIStyleResponse = {
            id: `chatcmpl-${uuidv4()}`,
            object: 'chat.completion',
            created: Date.now(),
            model: rawModel,
            choices: [
                {
                    index: 0,
                    finish_reason: 'stop', // or you can parse from response
                    message: {
                        role: 'assistant',
                        content: response.data.completion,
                    },
                },
            ],
        };
        return res.json(openAIStyleResponse);
    } else {
        // Streaming
        const streamingResponse = await axios({
            method: 'post',
            url,
            data: body,
            responseType: 'stream',
            headers: {
                'Content-Type': 'application/json',
                'x-api-key': ANTHROPIC_API_KEY,
            },
        });

        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache, no-transform');
        res.setHeader('Connection', 'keep-alive');

        // Anthropic streams in an SSE format => better parse out the data events 
        // and convert them into OpenAI chunk eventsl. Here we are passing them along directly.
        streamingResponse.data.on('data', (chunk) => {
            res.write(chunk);
        });
        streamingResponse.data.on('end', () => {
            res.end();
        });
        streamingResponse.data.on('error', (err) => {
            console.error('Anthropic Streaming Error:', err);
            res.end();
        });
    }
}

//////////////////////////////////////
// Hugging Face Chat
//////////////////////////////////////
async function handleHuggingFaceChat({
    res,
    rawModel,
    messages,
    stream,
    temperature,
    max_tokens,
    extraParams,
}) {
    // For Hugging Face Inference API - can call "https://api-inference.huggingface.co/models/<model>"  with the HUGGINGFACE_API_KEY in the headers.
    if (!HUGGINGFACE_API_KEY) {
        throw new Error('HUGGINGFACE_API_KEY not set in .env');
    }

    // need to build a single prompt from messages
    // HF pipeline might expect the conversation in a certain format depending on the model.
    // For a generic approach, let's just combine user messages 
    let conversation = '';
    messages.forEach((m) => {
        if (m.role === 'user') {
            conversation += `User: ${m.content}\n`;
        } else if (m.role === 'assistant') {
            conversation += `Assistant: ${m.content}\n`;
        } else {
            conversation += `${m.role}: ${m.content}\n`;
        }
    });
    conversation += 'Assistant:'; // we want the model to continue from here

    const url = `https://api-inference.huggingface.co/models/${rawModel}`;
    const requestBody = {
        inputs: conversation,
        parameters: {
            temperature: temperature == null ? 1.0 : temperature,
            max_new_tokens: max_tokens == null ? 256 : max_tokens,
            // HF has "top_p" or "top_k" and can pass them here as needed
            ...extraParams,
        },
    };

    // Hugging Face Inference API does not always provide streaming. 
    // If the model is set up with `--use_stream`, or if there is a special endpoint, we can do SSE streaming. 
    // But the default HF Inference endpoints typically don’t stream. So let's ignore "stream" here.
    try {
        const response = await axios.post(url, requestBody, {
            headers: {
                Authorization: `Bearer ${HUGGINGFACE_API_KEY}`,
                'Content-Type': 'application/json',
            },
        });
        // The response format depends on the model. Often it might be { "generated_text": "..." }
        // or an array [ { "generated_text": "..." } ] -> Assuming we can parse it in a standard way:
        let hfContent = '';
        if (Array.isArray(response.data)) {
            if (response.data[0] && response.data[0].generated_text) {
                hfContent = response.data[0].generated_text;
            } else {
                hfContent = JSON.stringify(response.data);
            }
        } else if (response.data.generated_text) {
            hfContent = response.data.generated_text;
        } else {
            hfContent = JSON.stringify(response.data);
        }

        // Convert to an OpenAI-like response
        const openAIStyleResp = {
            id: `chatcmpl-${uuidv4()}`,
            object: 'chat.completion',
            created: Date.now(),
            model: rawModel,
            choices: [
                {
                    index: 0,
                    finish_reason: 'stop',
                    message: {
                        role: 'assistant',
                        content: hfContent,
                    },
                },
            ],
        };

        return res.json(openAIStyleResp);
    } catch (err) {
        throw err;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Provider Handlers - Embeddings
//
//    We define handleOpenAIEmbeddings, handleAzureOpenAIEmbeddings, handleHuggingFaceEmbeddings, etc.
////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////
// OpenAI Embeddings
//////////////////////////////////////
async function handleOpenAIEmbeddings({ res, rawModel, input }) {
    if (!OPENAI_API_KEY) {
        throw new Error('OPENAI_API_KEY not set in .env');
    }

    const requestBody = {
        model: rawModel,
        input: input,
    };

    const response = await axios.post(
        'https://api.openai.com/v1/embeddings',
        requestBody,
        {
            headers: {
                Authorization: `Bearer ${OPENAI_API_KEY}`,
            },
        }
    );
    return res.json(response.data);
}

//////////////////////////////////////
// Azure OpenAI Embeddings
//////////////////////////////////////
async function handleAzureOpenAIEmbeddings({ res, rawModel, input }) {
    if (!AZURE_OPENAI_KEY || !AZURE_OPENAI_ENDPOINT) {
        throw new Error('Azure OpenAI credentials not set.');
    }
    // assumption - rawModel is the deployment name for embeddings
    // The endpoint is typically:
    // https://YOUR-RESOURCE-NAME.openai.azure.com/openai/deployments/<DEPLOYMENT>/embeddings?api-version=2023-03-15-preview
    const AZURE_API_VERSION = '2023-03-15-preview';
    const url = `${AZURE_OPENAI_ENDPOINT}/openai/deployments/${rawModel}/embeddings?api-version=${AZURE_API_VERSION}`;

    const requestBody = {
        input,
    };

    const response = await axios.post(url, requestBody, {
        headers: {
            'Content-Type': 'application/json',
            'api-key': AZURE_OPENAI_KEY,
        },
    });
    return res.json(response.data);
}

//////////////////////////////////////
// HuggingFace Embeddings
//////////////////////////////////////
async function handleHuggingFaceEmbeddings({ res, rawModel, input }) {
    // Some HF embedding models will accept a JSON body: { inputs: "your text" }
    // Then respond with an array like [[0.123, 0.456, ...]].
    // We can adapt it to the OpenAI embeddings format.

    if (!HUGGINGFACE_API_KEY) {
        throw new Error('HUGGINGFACE_API_KEY not set in .env');
    }

    // If input is an array might have to make multiple requests or batch them. 
    // If it's an array, you can do something like input.join("\n") or handle each item.
    const isArray = Array.isArray(input);
    const textInput = isArray ? input.join("\n") : input;

    const url = `https://api-inference.huggingface.co/models/${rawModel}`;
    const requestBody = {
        inputs: textInput,
    };

    const hfResponse = await axios.post(url, requestBody, {
        headers: {
            Authorization: `Bearer ${HUGGINGFACE_API_KEY}`,
            'Content-Type': 'application/json',
        },
    });

    // Suppose the response is [ [0.123, 0.456, ...] ]. We'll store that in an array
    const vector = hfResponse.data;
    if (!Array.isArray(vector) || !Array.isArray(vector[0])) {
        throw new Error(`Unexpected HuggingFace embeddings format: ${JSON.stringify(vector)}`);
    }

    // Convert to OpenAI embedding format
    const openAIStyleResp = {
        object: 'list',
        data: [
            {
                object: 'embedding',
                embedding: vector[0],
                index: 0,
            },
        ],
        model: rawModel,
        usage: {
            prompt_tokens: 0,
            total_tokens: 0,
        },
    };

    return res.json(openAIStyleResp);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Provider Handlers - Image Generation
//
//    handleOpenAIImageGeneration, etc.
//    Accept prompt, n, size, etc. and return an OpenAI-like result with "data" containing URLs.
////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////
// OpenAI Image Generation
//////////////////////////////////////
async function handleOpenAIImageGeneration({ res, prompt, n, size }) {
    if (!OPENAI_API_KEY) {
        throw new Error('OPENAI_API_KEY not set in .env');
    }
    const requestBody = {
        prompt,
        n: n || 1,
        size: size || '1024x1024',
    };

    const response = await axios.post(
        'https://api.openai.com/v1/images/generations',
        requestBody,
        {
            headers: {
                'Content-Type': 'application/json',
                Authorization: `Bearer ${OPENAI_API_KEY}`,
            },
        }
    );
    return res.json(response.data);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// Basic Health Endpoint
////////////////////////////////////////////////////////////////////////////////////////////////////
app.get('/health', (req, res) => {
    return res.json({
        status: 'ok',
        timestamp: new Date().toISOString(),
    });
});

////////////////////////////////////////////////////////////////////////////////////////////////////
// Start the Express Server
////////////////////////////////////////////////////////////////////////////////////////////////////
app.listen(PORT, () => {
    console.log(`LLM core proxy server is running on http://localhost:${PORT}`);
});
