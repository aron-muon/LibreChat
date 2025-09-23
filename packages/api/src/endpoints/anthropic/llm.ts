import { Dispatcher, ProxyAgent } from 'undici';
import { AnthropicVertex } from '@anthropic-ai/vertex-sdk';
import { GoogleAuth } from 'google-auth-library';
import { AnthropicClientOptions } from '@librechat/agents';
import { anthropicSettings, removeNullishValues, AuthKeys } from 'librechat-data-provider';
import type {
  AnthropicLLMConfigResult,
  AnthropicConfigOptions,
  AnthropicCredentials,
} from '~/types/anthropic';
import { checkPromptCacheSupport, getClaudeHeaders, configureReasoning } from './helpers';

/**
 * Generates configuration options for creating an Anthropic language model (LLM) instance.
 * @param credentials - The API key for authentication with Anthropic, or credentials object for Vertex AI.
 * @param options={} - Additional options for configuring the LLM.
 * @returns Configuration options for creating an Anthropic LLM instance, with null and undefined values removed.
 */
function getLLMConfig(
  credentials: string | AnthropicCredentials | undefined,
  options: AnthropicConfigOptions = {},
): AnthropicLLMConfigResult {
  const systemOptions = {
    thinking: options.modelOptions?.thinking ?? anthropicSettings.thinking.default,
    promptCache: options.modelOptions?.promptCache ?? anthropicSettings.promptCache.default,
    thinkingBudget:
      options.modelOptions?.thinkingBudget ?? anthropicSettings.thinkingBudget.default,
  };

  /** Couldn't figure out a way to still loop through the object while deleting the overlapping keys when porting this
   * over from javascript, so for now they are being deleted manually until a better way presents itself.
   */
  if (options.modelOptions) {
    delete options.modelOptions.thinking;
    delete options.modelOptions.promptCache;
    delete options.modelOptions.thinkingBudget;
  } else {
    throw new Error('No modelOptions provided');
  }

  const defaultOptions = {
    model: anthropicSettings.model.default,
    maxOutputTokens: anthropicSettings.maxOutputTokens.default,
    stream: true,
  };

  const mergedOptions = Object.assign(defaultOptions, options.modelOptions);

  let creds: AnthropicCredentials = {};
  if (typeof credentials === 'string') {
    try {
      creds = JSON.parse(credentials);
    } catch (err: unknown) {
      throw new Error(
        `Error parsing string credentials: ${err instanceof Error ? err.message : 'Unknown error'}`,
      );
    }
  } else if (credentials && typeof credentials === 'object') {
    creds = credentials;
  }

  const vertexServiceKeyRaw = creds[AuthKeys.GOOGLE_SERVICE_KEY] ?? {};
  const vertexServiceKey =
    typeof vertexServiceKeyRaw === 'string'
      ? JSON.parse(vertexServiceKeyRaw)
      : (vertexServiceKeyRaw ?? {});

  const apiKey = creds[AuthKeys.ANTHROPIC_API_KEY] ?? null;
  const vertexProjectId = !apiKey ? (vertexServiceKey?.project_id ?? null) : null;

  let requestOptions: AnthropicClientOptions & { stream?: boolean } = {
    model: mergedOptions.model,
    stream: mergedOptions.stream,
    temperature: mergedOptions.temperature,
    stopSequences: mergedOptions.stop,
    maxTokens:
      mergedOptions.maxOutputTokens || anthropicSettings.maxOutputTokens.reset(mergedOptions.model),
    clientOptions: {},
    invocationKwargs: {
      metadata: {
        user_id: mergedOptions.user,
      },
    },
  };

  if (vertexProjectId) {
    // Vertex AI configuration - use custom client
    const location = process.env.ANTHROPIC_VERTEX_REGION || 'global';

    try {
      const googleAuth = new GoogleAuth({
        credentials: vertexServiceKey,
        scopes: 'https://www.googleapis.com/auth/cloud-platform',
      });

      const vertexClient = new AnthropicVertex({
        projectId: vertexProjectId,
        region: location,
        googleAuth: googleAuth,
      });

      requestOptions.createClient = () => vertexClient;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`Failed to create Vertex AI client: ${message}`);
    }
  } else if (apiKey) {
    // Direct API configuration
    requestOptions.apiKey = apiKey;
  } else {
    throw new Error(
      'Invalid credentials provided. Please provide either a valid Anthropic API key or service account credentials for Vertex AI.',
    );
  }

  requestOptions = configureReasoning(requestOptions, systemOptions);

  if (!/claude-3[-.]7/.test(mergedOptions.model)) {
    requestOptions.topP = mergedOptions.topP;
    requestOptions.topK = mergedOptions.topK;
  } else if (requestOptions.thinking == null) {
    requestOptions.topP = mergedOptions.topP;
    requestOptions.topK = mergedOptions.topK;
  }

  const supportsCacheControl =
    systemOptions.promptCache === true && checkPromptCacheSupport(requestOptions.model ?? '');
  const headers = getClaudeHeaders(requestOptions.model ?? '', supportsCacheControl);
  if (headers && requestOptions.clientOptions) {
    requestOptions.clientOptions.defaultHeaders = headers;
  }

  if (options.proxy && requestOptions.clientOptions) {
    const proxyAgent = new ProxyAgent(options.proxy);
    requestOptions.clientOptions.fetchOptions = {
      dispatcher: proxyAgent,
    };
  }

  if (options.reverseProxyUrl && requestOptions.clientOptions) {
    requestOptions.clientOptions.baseURL = options.reverseProxyUrl;
    requestOptions.anthropicApiUrl = options.reverseProxyUrl;
  }

  const tools = [];

  if (mergedOptions.web_search) {
    tools.push({
      type: 'web_search_20250305',
      name: 'web_search',
    });
  }

  return {
    tools,
    llmConfig: removeNullishValues(
      requestOptions as Record<string, unknown>,
    ) as AnthropicClientOptions & { clientOptions?: { fetchOptions?: { dispatcher: Dispatcher } } },
  };
}

export { getLLMConfig };
