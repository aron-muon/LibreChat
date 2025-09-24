const path = require('path');
const { getLLMConfig, loadServiceKey, isEnabled } = require('@librechat/api');
const { EModelEndpoint, AuthKeys } = require('librechat-data-provider');
const { getUserKey, checkUserKeyExpiry } = require('~/server/services/UserService');
const AnthropicClient = require('~/app/clients/AnthropicClient');

const initializeClient = async ({ req, res, endpointOption, overrideModel, optionsOnly }) => {
  const appConfig = req.config;
  const { ANTHROPIC_API_KEY, ANTHROPIC_REVERSE_PROXY, PROXY } = process.env;
  const expiresAt = req.body.key;

  let credentials = {};
  let anthropicApiKey = null;

  if (isEnabled(process.env.ANTHROPIC_USE_VERTEX)) {
    let serviceKey = {};
    try {
      const serviceKeyPath =
        process.env.GOOGLE_SERVICE_KEY_FILE ||
        path.join(__dirname, '../../../..', 'data', 'auth.json');
      serviceKey = await loadServiceKey(serviceKeyPath);
      if (!serviceKey) {
        serviceKey = {};
      }
    } catch (_e) {
      // Service key loading failed, but that's okay if not required
      serviceKey = {};
    }
    credentials[AuthKeys.GOOGLE_SERVICE_KEY] = serviceKey;
  } else {
    const isUserProvided = ANTHROPIC_API_KEY === 'user_provided';
    anthropicApiKey = isUserProvided
      ? await getUserKey({ userId: req.user.id, name: EModelEndpoint.anthropic })
      : ANTHROPIC_API_KEY;

    if (!anthropicApiKey) {
      throw new Error('Anthropic API key not provided. Please provide it again.');
    }

    if (expiresAt && isUserProvided) {
      checkUserKeyExpiry(expiresAt, EModelEndpoint.anthropic);
    }
    credentials[AuthKeys.ANTHROPIC_API_KEY] = anthropicApiKey;
  }

  let clientOptions = {};

  /** @type {undefined | TBaseEndpoint} */
  const anthropicConfig = appConfig.endpoints?.[EModelEndpoint.anthropic];

  if (anthropicConfig) {
    clientOptions.streamRate = anthropicConfig.streamRate;
    clientOptions.titleModel = anthropicConfig.titleModel;
  }

  const allConfig = appConfig.endpoints?.all;
  if (allConfig) {
    clientOptions.streamRate = allConfig.streamRate;
  }

  if (optionsOnly) {
    clientOptions = Object.assign(
      {
        proxy: PROXY ?? null,
        reverseProxyUrl: ANTHROPIC_REVERSE_PROXY ?? null,
        modelOptions: endpointOption?.model_parameters ?? {},
      },
      clientOptions,
    );
    if (overrideModel) {
      clientOptions.modelOptions.model = overrideModel;
    }
    clientOptions.modelOptions.user = req.user.id;
    return getLLMConfig(credentials, clientOptions);
  }

  const client = new AnthropicClient(credentials, {
    req,
    res,
    reverseProxyUrl: ANTHROPIC_REVERSE_PROXY ?? null,
    proxy: PROXY ?? null,
    ...clientOptions,
    ...endpointOption,
  });

  return {
    client,
    anthropicApiKey,
  };
};

module.exports = initializeClient;
