/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import {GoogleGenAI} from '@google/genai';

import {logger} from '../utils/logger.js';

import {BaseLlmConnection} from './base_llm_connection.js';
import {Gemini} from './google_llm.js';
import {LlmRequest} from './llm_request.js';
import {LlmResponse} from './llm_response.js';

const APIGEE_PROXY_URL_ENV_VARIABLE_NAME = 'APIGEE_PROXY_URL';
const GOOGLE_GENAI_USE_VERTEXAI_ENV_VARIABLE_NAME = 'GOOGLE_GENAI_USE_VERTEXAI';
const PROJECT_ENV_VARIABLE_NAME = 'GOOGLE_CLOUD_PROJECT';
const LOCATION_ENV_VARIABLE_NAME = 'GOOGLE_CLOUD_LOCATION';
const GOOGLE_GENAI_API_KEY_ENV_VARIABLE_NAME = 'GOOGLE_GENAI_API_KEY';
const GEMINI_API_KEY_ENV_VARIABLE_NAME = 'GEMINI_API_KEY';

export interface ApigeeLlmParams {
  /**
   * The name of the model to use. The model string specifies the LLM provider
   * (e.g., Vertex AI, Gemini), API version, and the model ID. Supported format:
   *     `apigee/[<provider>/][<version>/]<model_id>`
   *     Components:
   *       `provider` (optional): `vertex_ai` or `gemini`.
   *       `version` (optional): The API version (e.g., `v1`, `v1beta`). If not
   *         provided, a default version will selected based on the provider.
   *       `model_id` (required): The model identifier (e.g.,
   *         `gemini-2.5-flash`).
   *     Examples:
   *       - `apigee/gemini-2.5-flash`
   *       - `apigee/v1/gemini-2.5-flash`
   *       - `apigee/vertex_ai/gemini-2.5-flash`
   *       - `apigee/gemini/v1/gemini-2.5-flash`
   *       - `apigee/vertex_ai/v1beta/gemini-2.5-flash`
   */
  model: string;
  /**
   * The proxy URL for the provider API. If not provided, it will look for
   * the APIGEE_PROXY_URL environment variable.
   */
  proxyUrl?: string;
  /**
   * API key to use. If not provided, it will look for
   * the GOOGLE_GENAI_API_KEY or GEMINI_API_KEY environment variable. If gemini
   * provider is selected and no key is provided, the fake key "-" will be
   * used for the "x-goog-api-key" header.
   */
  apiKey?: string;
  /**
   * Headers to merge with internally crafted headers.
   */
  headers?: Record<string, string>;
}

export class ApigeeLlm extends Gemini {
  private readonly proxyUrl: string;


  constructor({
    model,
    proxyUrl,
    apiKey,
    headers,
  }: ApigeeLlmParams) {
    if (!ApigeeLlm.validateModel(model)) {
      throw new Error(`Model ${
          model} is not a valid Apigee model, expected apigee/[<provider>/][<version>/]<model_id>`);
    }

    const vertexai = ApigeeLlm.isVertexAi(model);
    let project = '';
    let location = '';
    const canReadEnv = typeof process === 'object';

    if (vertexai) {
      if (!canReadEnv) {
        throw new Error(`Environment variables ${
            PROJECT_ENV_VARIABLE_NAME} and ${
            LOCATION_ENV_VARIABLE_NAME} must be provided when using Vertex AI.`);
      }
      if (!project) {
        project = process.env[PROJECT_ENV_VARIABLE_NAME] || '';
      }
      if (!project) {
        throw new Error(`The ${
            PROJECT_ENV_VARIABLE_NAME} environment variable must be set when using Vertex AI.`);
      }
      if (!location) {
        location = process.env[LOCATION_ENV_VARIABLE_NAME] || '';
      }
      if (!location) {
        throw new Error(`The ${
            LOCATION_ENV_VARIABLE_NAME} environment variable must be set when using Vertex AI.`);
      }
    } else {
      if (canReadEnv && !apiKey) {
        // First check env vars then add a fake key if no key is provided.
        apiKey = process.env[GOOGLE_GENAI_API_KEY_ENV_VARIABLE_NAME] ||
            process.env[GEMINI_API_KEY_ENV_VARIABLE_NAME];
      }
      if (!apiKey) {
        logger.warn(
            `No API key provided when using a Gemini model, using a fake key "-".`);
        apiKey = '-';
      }
    }
    super({
      model: model,
      apiKey: apiKey,
      vertexai: vertexai,
      project: project,
      location: location,
      headers: headers,
    });

    this.proxyUrl = proxyUrl ?? '';
    if (canReadEnv && !this.proxyUrl) {
      this.proxyUrl = process.env[APIGEE_PROXY_URL_ENV_VARIABLE_NAME] ?? '';
    }
    if (!this.proxyUrl) {
      throw new Error(`Proxy URL must be provided via the constructor or ${
          APIGEE_PROXY_URL_ENV_VARIABLE_NAME} environment variable.`);
    }
  }

  /**
   * A list of model name patterns that are supported by this LLM.
   *
   * @returns A list of supported models.
   */
  static override readonly supportedModels: Array<string|RegExp> = [
    /apigee\/.*/,
  ];

  private _apigeeApiClient?: GoogleGenAI;
  private _apigeeLiveApiClient?: GoogleGenAI;
  private _apigeeLiveApiVersion?: string;

  override get apiClient(): GoogleGenAI {
    if (this._apigeeApiClient) {
      return this._apigeeApiClient;
    }

    const combinedHeaders = {
      ...this.trackingHeaders,
      ...this.headers,
    }

    if (this.vertexai) {
      this._apigeeApiClient = new GoogleGenAI({
        vertexai: this.vertexai,
        project: this.project,
        location: this.location,
        httpOptions: {
          headers: combinedHeaders,
          baseUrl: this.proxyUrl,
        },
      });
    }
    else {
      this._apigeeApiClient = new GoogleGenAI({
        apiKey: this.apiKey,
        httpOptions: {
          headers: combinedHeaders,
          baseUrl: this.proxyUrl,
        },
      });
    }
    return this._apigeeApiClient;
  }


  private identifyApiVersion(): string {
    const modelTrimmed = this.model.startsWith('apigee/') ?
        this.model.substring('apigee/'.length) :
        this.model;
    const components = modelTrimmed.split('/');
    if (components.length === 3) {
      // Format: <provider>/<version>/<model_id>
      return components[1];
    }
    if (components.length === 2) {
      // Format: <version>/<model_id> but not <provider>/<model_id>
      if ((components[0] != 'vertex_ai') && (components[0] != 'gemini') &&
          components[0].startsWith('v')) {
        return components[0];
      }
    }
    // Default to v1beta1 for vertex AI and v1alpha for Gemini.
    return this.vertexai ? 'v1beta1' : 'v1alpha';
  }

  override get liveApiVersion(): string {
    if (!this._apigeeLiveApiVersion) {
      this._apigeeLiveApiVersion = this.identifyApiVersion();
    }
    return this._apigeeLiveApiVersion;
  }

  override get liveApiClient(): GoogleGenAI {
    if (!this._apigeeLiveApiClient) {
      this._apigeeLiveApiClient = new GoogleGenAI({
        apiKey: this.apiKey,
        httpOptions: {
          headers: this.trackingHeaders,
          apiVersion: this.liveApiVersion,
          baseUrl: this.proxyUrl,
        },
      });
    }
    return this._apigeeLiveApiClient;
  }

  private static isVertexAi(model: string): boolean {
    return !model.startsWith('apigee/gemini/') &&
        (model.startsWith('apigee/vertex_ai/') ||
         ApigeeLlm.isEnvEnabled(GOOGLE_GENAI_USE_VERTEXAI_ENV_VARIABLE_NAME));
  }

  private static isEnvEnabled(envVariableName: string): boolean {
    const canReadEnv = typeof process === 'object';
    if (!canReadEnv) {
      return false;
    }
    const envValue = process.env[envVariableName];
    if (envValue) {
      return envValue.toLowerCase() === 'true' || envValue === '1';
    }
    return false;
  }


  private static validateModel(model: string): boolean {
    const validProviders = ['vertex_ai', 'gemini'];
    if (!model.startsWith('apigee/')) {
      return false;
    }
    const modelPart = model.substring('apigee/'.length);
    if (modelPart.length === 0) {
      return false;
    }
    const components = modelPart.split('/', -1);
    if (components[components.length - 1].length === 0) {
      return false;
    }
    // If the model string has exactly 1 component, it means only the model_id
    // is present. This is a valid format (e.g. "apigee/my-model").
    if (components.length == 1) {
      return true;
    }
    if (components.length == 2) {
      // allowed format: apigee/<provider>/<model_id>
      // (e.g. apigee/vertex_ai/my-model)
      if (validProviders.includes(components[0])) {
        return true;
      }
      // allowed format: apigee/<version>/<model_id>
      // (e.g.apigee/v1beta1/my-model)
      return components[0].startsWith('v')
    }
    if (components.length == 3) {
      // allowed format: apigee/<provider>/<version>/<model_id>
      // (e.g. apigee/vertex_ai/v1beta1/my-model)
      if (!validProviders.includes(components[0])) {
        return false;
      }
      return components[1].startsWith('v');
    }
    return false;
  }

  private static getModelId(model: string): string {
    if (!ApigeeLlm.validateModel(model)) {
      throw new Error(`Model ${
          model} is not a valid Apigee model, expected apigee/[<provider>/][<version>/]<model_id>`);
    }
    const components = model.split('/');
    return components[components.length - 1];
  }

  override async *
      generateContentAsync(
          llmRequest: LlmRequest,
          stream = false,
          ): AsyncGenerator<LlmResponse, void> {
    const modelToUse = llmRequest.model ?? this.model;
    llmRequest.model = ApigeeLlm.getModelId(modelToUse);
    yield* super.generateContentAsync(llmRequest, stream);
  }

  override async connect(llmRequest: LlmRequest): Promise<BaseLlmConnection> {
    const modelToUse = llmRequest.model ?? this.model;
    llmRequest.model = ApigeeLlm.getModelId(modelToUse);
    return super.connect(llmRequest);
  }
}
