/**
 * [CUSTOM] Model Override for Task Tool
 *
 * This module provides runtime model override capability for subagents.
 * Allows the lead agent to specify which model each subagent should use.
 *
 * Features:
 * - Parse model string in "providerID/modelID" format
 * - Validate model exists in provider catalog
 * - Validate model supports tool calls
 * - Permission check via "model_override" permission
 *
 * Usage:
 *   Task tool accepts optional `model` parameter
 *   e.g., task(model="anthropic/claude-sonnet-4", ...)
 *
 * Fallback chain:
 *   1. Explicit task model parameter
 *   2. Agent configured model (from config)
 *   3. Parent session model
 *   4. Default model (opencode/big-pickle)
 */

import type { Provider } from "@/provider/provider"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { ModelV2 } from "@opencode-ai/core/model"
import { Effect } from "effect"

// Default fallback model when no model is specified anywhere
export const DEFAULT_MODEL = {
  providerID: "opencode",
  modelID: "big-pickle",
}

export type ModelOverride = {
  providerID: ProviderV2.ID
  modelID: ModelV2.ID
}

export type ModelOverrideResult =
  | { success: true; model: ModelOverride }
  | { success: false; error: string }

export type OrchestraRole = "orchestra-implementer" | "orchestra-reviewer" | "orchestra-tester"

// Bound specialist fan-out while allowing independent reviewer/tester work.
export const ORCHESTRA_MAX_CONCURRENT_TASKS = 3

export type OrchestraTaskCounts = Partial<Record<OrchestraRole, number>>

const orchestraContracts = {
  "orchestra-implementer": {
    statuses: ["complete", "blocked"],
    fields: ["CHANGES", "VALIDATION", "RISKS", "NEXT_ACTION"],
  },
  "orchestra-reviewer": {
    statuses: ["approved", "needs_revision", "blocked"],
    fields: ["FINDINGS", "VALIDATION", "RISKS", "NEXT_ACTION"],
  },
  "orchestra-tester": {
    statuses: ["passed", "failed", "blocked"],
    fields: ["CHANGES", "VALIDATION", "FAILURES", "NEXT_ACTION"],
  },
} satisfies Record<OrchestraRole, { statuses: string[]; fields: string[] }>

export function orchestraConcurrencyError(counts: OrchestraTaskCounts, role: string): string | undefined {
  if (!(role in { "orchestra-implementer": true, "orchestra-reviewer": true, "orchestra-tester": true })) {
    return undefined
  }

  const total = Object.values(counts).reduce((sum, value) => sum + (value ?? 0), 0)
  if (total >= ORCHESTRA_MAX_CONCURRENT_TASKS) {
    return `The orchestra Lead already has ${total} active specialist tasks (maximum ${ORCHESTRA_MAX_CONCURRENT_TASKS}). Wait for one to finish before delegating another.`
  }

  if ((counts[role as OrchestraRole] ?? 0) > 0) {
    return `The orchestra role ${role} already has an active task. Reuse its task_id or wait for it to finish instead of starting a concurrent duplicate.`
  }

  return undefined
}

/**
 * Require the structured status contract for configured orchestra specialists.
 * Generic subagents keep the existing free-form output behavior.
 */
export function validateOrchestraRoleOutput(role: string, output: string): string | undefined {
  const contract = orchestraContracts[role as OrchestraRole]
  if (!contract) return undefined

  // Use the final STATUS line so examples or earlier discussion cannot satisfy
  // the contract. Markdown decoration around field names remains tolerated.
  const statuses = [...output.matchAll(/(?:^|\r?\n)\s*(?:[#>*-]+\s*)?(?:[*_`~]+\s*)?STATUS\s*(?:[*_`~]+\s*)?:\s*([^\r\n;]+)/gi)]
  const match = statuses.at(-1)
  const status = match?.[1]
    ?.replace(/^[*_`~]+|[*_`~]+$/g, "")
    .trim()
    .toLowerCase()
  const handoff = match ? output.slice((match.index ?? 0) + match[0].length) : ""
  const complete = contract.fields.every((field) =>
    new RegExp(`(?:^|\\r?\\n)\\s*(?:[#>*-]+\\s*)?(?:[*_\`~]+\\s*)?${field}\\s*(?:[*_\`~]+\\s*)?:`, "i").test(
      handoff,
    ),
  )
  if (status && contract.statuses.includes(status) && complete) return undefined

  return `${role} must return a complete handoff with STATUS (${contract.statuses.join(", ")}) and fields ${contract.fields.join(", ")}.`
}

export function orchestraHandoffSchema(role: string): Record<string, unknown> | undefined {
  const contract = orchestraContracts[role as OrchestraRole]
  if (!contract) return undefined

  return {
    type: "object",
    properties: {
      status: { type: "string", enum: contract.statuses },
      ...Object.fromEntries(contract.fields.map((field) => [field.toLowerCase(), { type: "string" }])),
    },
    required: ["status", ...contract.fields.map((field) => field.toLowerCase())],
    additionalProperties: false,
  }
}

export function renderOrchestraHandoff(role: string, value: unknown): string | undefined {
  const contract = orchestraContracts[role as OrchestraRole]
  if (!contract || typeof value !== "object" || value === null) return undefined
  const output = value as Record<string, unknown>
  if (typeof output.status !== "string" || !contract.statuses.includes(output.status)) return undefined
  if (!contract.fields.every((field) => typeof output[field.toLowerCase()] === "string")) return undefined

  return [
    `STATUS: ${output.status}`,
    ...contract.fields.map((field) => `${field}: ${output[field.toLowerCase()]}`),
  ].join("\n")
}

/**
 * Keep the parent task protocol total when a specialist stops without a handoff.
 * The raw transcript is preserved for diagnosis, but never escapes as a tool error.
 */
export function normalizeOrchestraRoleOutput(role: string, output: string): string {
  if (!validateOrchestraRoleOutput(role, output)) return output

  const summary = `${role} did not return a valid structured handoff.`
  const raw = (output || "(no output)")
    .split(/\r?\n/)
    .map((line) => `RAW> ${line}`)
    .join("\n")
  if (role === "orchestra-reviewer")
    return [
      "STATUS: blocked",
      `FINDINGS: ${summary}`,
      "REASON: invalid_handoff",
      "VALIDATION: The specialist task ended without a complete recognized handoff.",
      `RISKS: Raw specialist output follows:\n${raw}`,
      "NEXT_ACTION: The Lead must inspect the raw output and resume the specialist once if useful.",
    ].join("\n")
  if (role === "orchestra-tester")
    return [
      "STATUS: blocked",
      `CHANGES: ${summary}`,
      "REASON: invalid_handoff",
      "VALIDATION: The specialist task ended without a complete recognized handoff.",
      `FAILURES: Raw specialist output follows:\n${raw}`,
      "NEXT_ACTION: The Lead must inspect the raw output and resume the specialist once if useful.",
    ].join("\n")

  return [
    "STATUS: blocked",
    `CHANGES: ${summary}`,
    "REASON: invalid_handoff",
    "VALIDATION: The specialist task ended without a complete recognized handoff.",
    `RISKS: Raw specialist output follows:\n${raw}`,
    "NEXT_ACTION: The Lead must inspect the raw output and resume the specialist once if useful.",
  ].join("\n")
}

/**
 * Parse model string in "providerID/modelID" format
 * Returns null if format is invalid
 */
export function parseModelString(modelStr: string): { providerID: string; modelID: string } | null {
  const slashIndex = modelStr.indexOf("/")
  if (slashIndex <= 0 || slashIndex === modelStr.length - 1) {
    return null
  }
  return {
    providerID: modelStr.slice(0, slashIndex),
    modelID: modelStr.slice(slashIndex + 1),
  }
}

/**
 * Validate and resolve model override
 * Returns ModelOverrideResult with either the resolved model or error message
 */
export function resolveModelOverride(
  modelStr: string,
  provider: Provider.Interface,
): Effect.Effect<ModelOverrideResult, never, never> {
  return Effect.gen(function* () {
    // Parse model string
    const parsed = parseModelString(modelStr)
    if (!parsed) {
      return {
        success: false,
        error: `Invalid model format: "${modelStr}". Expected "providerID/modelID" (e.g. "anthropic/claude-sonnet-4").`,
      }
    }

    const modelOverride: ModelOverride = {
      providerID: ProviderV2.ID.make(parsed.providerID),
      modelID: ModelV2.ID.make(parsed.modelID),
    }

    // Check if model exists in provider
    const modelInfo = yield* provider.getModel(modelOverride.providerID, modelOverride.modelID).pipe(
      Effect.catchTag("ProviderModelNotFoundError", () => Effect.succeed(undefined)),
    )

    if (!modelInfo) {
      return {
        success: false,
        error: `Model "${modelStr}" not found. Check if the provider is configured and the model ID is correct.`,
      }
    }

    // Check if model supports tool calls
    if (!modelInfo.capabilities.toolcall) {
      return {
        success: false,
        error: `Model "${modelStr}" does not support tool calls (tool_call: false). Subagents require a model with tool_call capability.`,
      }
    }

    return { success: true, model: modelOverride }
  })
}

/**
 * Resolve final model for subagent
 * Priority: explicit override > agent configured > parent session > default
 */
export function resolveFinalModel(
  override: ModelOverride | undefined,
  agentModel: { providerID: string; modelID: string } | undefined,
  parentModel: { providerID: string; modelID: string } | undefined,
): ModelOverride {
  if (override) {
    return override
  }

  if (agentModel) {
    return {
      providerID: ProviderV2.ID.make(agentModel.providerID),
      modelID: ModelV2.ID.make(agentModel.modelID),
    }
  }

  if (parentModel) {
    return {
      providerID: ProviderV2.ID.make(parentModel.providerID),
      modelID: ModelV2.ID.make(parentModel.modelID),
    }
  }

  return {
    providerID: ProviderV2.ID.make(DEFAULT_MODEL.providerID),
    modelID: ModelV2.ID.make(DEFAULT_MODEL.modelID),
  }
}
