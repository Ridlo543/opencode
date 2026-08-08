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
export const ORCHESTRA_ASSISTANT = "orchestra-assistant-specialist"
export const ORCHESTRA_LEAD = "orchestra"
export const ORCHESTRA_CUSTOM_LEAD = "orchestra-custom"
export type OrchestraDelegate = OrchestraRole | typeof ORCHESTRA_ASSISTANT
const orchestraRoles = new Set<string>(["orchestra-implementer", "orchestra-reviewer", "orchestra-tester"])
export const RESEARCH_ROLES = [
  "research-methodologist",
  "research-scout",
  "research-analyst",
  "research-critic",
  "research-writer",
  "research-reviewer",
  "research-editor",
] as const
export type ResearchRole = (typeof RESEARCH_ROLES)[number]
const researchRoles = new Set<string>(RESEARCH_ROLES)

export function isOrchestraRole(agent: string): agent is OrchestraRole {
  return orchestraRoles.has(agent)
}

export function isOrchestraPrivateAgent(agent: string): agent is OrchestraDelegate {
  return isOrchestraRole(agent) || agent === ORCHESTRA_ASSISTANT
}

export function isOrchestraLead(agent: string) {
  return agent === ORCHESTRA_LEAD || agent === ORCHESTRA_CUSTOM_LEAD
}

export function isResearchRole(agent: string): agent is ResearchRole {
  return researchRoles.has(agent)
}

export function isPrivateWorkflowAgent(agent: string): agent is OrchestraDelegate | ResearchRole {
  return isOrchestraPrivateAgent(agent) || isResearchRole(agent)
}

export function workflowTaskAccessError(caller: string, target: string): string | undefined {
  if (isOrchestraLead(caller)) {
    if (isOrchestraPrivateAgent(target)) return undefined
    return `The orchestra Lead must delegate using orchestra-assistant-specialist, orchestra-implementer, orchestra-reviewer, or orchestra-tester; received ${target}.`
  }
  if (caller === "research") {
    if (isResearchRole(target)) return undefined
    return `The Research Lead must delegate using ${RESEARCH_ROLES.join(", ")}; received ${target}.`
  }
  if (isOrchestraPrivateAgent(caller)) {
    return `${caller} cannot delegate tasks; only the orchestra Lead owns delegation and workflow routing.`
  }
  if (isResearchRole(caller)) {
    return `${caller} cannot delegate tasks; only the Research Lead owns delegation and workflow routing.`
  }
  if (isOrchestraPrivateAgent(target)) {
    return `${target} is a private Orchestra specialist and can only be delegated by the orchestra Lead.`
  }
  if (isResearchRole(target)) {
    return `${target} is a private Research specialist and can only be delegated by the Research Lead.`
  }
  return undefined
}

// Match the native task fan-out while allowing any mix of specialist roles.
export const ORCHESTRA_MAX_CONCURRENT_TASKS = 4
export const ORCHESTRA_MAX_BLOCKED_ATTEMPTS = 5

export type OrchestraTaskCounts = Partial<Record<OrchestraDelegate, number>>

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

const orchestraSuccessStatuses = new Set(["complete", "approved", "passed"])

const evidenceHeadingPattern =
  /^\s*(?:[#*-]+\s*)?(?:(?:VALIDATION|VALIDATION_RESULTS|VERIFICATION|VERIFICATION_RESULTS|TESTS|TESTS_RUN|TEST_RESULTS|TESTING|CHECKS|CHECKS_RUN|CHECK_RESULTS|COMMANDS_RUN|EVIDENCE|PROOF_OF_CHECKS|BUKTI_VALIDASI|CONFIDENCE_EVIDENCE|QUALITY_GATES|RESULT|RESULTS)\s*:|(?:checks|review findings|quality gates|validation|verification|evidence|proof)\s*$)/i

function hasGroundedEvidence(body: string) {
  const lines = body.split(/\r?\n/)
  let sawHeading = false
  for (const raw of lines) {
    const line = raw.replace(/^\s*[-*+]\s*/, "").trim()
    if (!line) continue
    if (evidenceHeadingPattern.test(line)) {
      sawHeading = true
      const value = line.replace(evidenceHeadingPattern, "").trim()
      if (hasMeaningfulHandoffValue([value])) return true
      continue
    }
    if (sawHeading && /passed|succeeded|success|ok\b|fail(?:ed|ure)?|error|command|test|check|result|exit/i.test(line)) return true
  }
  return false
}

function finalHandoffStatus(output: string) {
  let result: { status: string; line: number } | undefined
  let fence: string | undefined
  const lines = output.split(/\r?\n/)
  for (let line = 0; line < lines.length; line++) {
    const value = lines[line] ?? ""
    const fenceMatch = value.match(/^\s*(```+|~~~+)/)
    if (fenceMatch) {
      fence = fence ? undefined : fenceMatch[1]?.[0]
      continue
    }
    if (fence || /^\s*>/.test(value)) continue
    const match = value.match(/^\s*(?:[#[*-]+\s*)?(?:[*_`~]+\s*)?STATUS\s*(?:[*_`~]+\s*)?:\s*([^;]+?)\s*$/i)
    if (!match) continue
    result = {
      status: (match[1] ?? "")
        .replace(/^[*_`~]+|[*_`~]+$/g, "")
        .trim()
        .toLowerCase(),
      line,
    }
  }
  return result ? { status: result.status, handoff: lines.slice(result.line + 1).join("\n") } : undefined
}

function hasMeaningfulHandoffValue(lines: string[] | undefined) {
  if (!lines) return false
  const value = lines
    .join("\n")
    .replace(/^\s*[-*+]\s*/gm, "")
    .trim()
  if (!value || !/[\p{L}\p{N}]/u.test(value)) return false
  return !/^(?:<\s*(?:text|value|todo)\s*>|\.{2,}|tbd|todo)$/i.test(value)
}

export function orchestraConcurrencyError(counts: OrchestraTaskCounts, role: string): string | undefined {
  if (!isOrchestraPrivateAgent(role)) return undefined

  const total = Object.values(counts).reduce((sum, value) => sum + (value ?? 0), 0)
  if (total >= ORCHESTRA_MAX_CONCURRENT_TASKS) {
    return `The orchestra Lead already has ${total} active specialist tasks (maximum ${ORCHESTRA_MAX_CONCURRENT_TASKS}). Wait for one to finish before delegating another.`
  }

  return undefined
}

/**
 * Require a valid role status and substantive handoff body. Suggested headings
 * improve interoperability but must not override readable specialist evidence.
 */
export function validateOrchestraRoleOutput(role: string, output: string): string | undefined {
  const contract = orchestraContracts[role as OrchestraRole]
  if (!contract) return undefined

  // Use the final unquoted STATUS line outside code fences so examples and raw
  // transcripts cannot satisfy the contract. Parse only the handoff after it.
  const final = finalHandoffStatus(output)
  const status = final?.status
  const handoff = final?.handoff ?? ""
  const meaningful = hasMeaningfulHandoffValue([handoff]) && handoff.replace(/^\s*[-*+]\s*/gm, "").trim().length >= 40
  if (!status || !contract.statuses.includes(status) || !meaningful) {
    return `${role} must return STATUS (${contract.statuses.join(", ")}) followed by a substantive handoff. Suggested fields: ${contract.fields.join(", ")}.`
  }
  if (orchestraSuccessStatuses.has(status) && !hasGroundedEvidence(handoff)) {
    return `${role} must ground a success status (${status}) in concrete validation or observed execution evidence; a human-readable result is sufficient, but a status alone is not.`
  }
  return undefined
}

export function orchestraRoleStatus(role: string, output: string) {
  const contract = orchestraContracts[role as OrchestraRole]
  if (!contract || validateOrchestraRoleOutput(role, output)) return undefined
  const status = finalHandoffStatus(output)?.status
  return status && contract.statuses.includes(status) ? status : undefined
}

export function orchestraBlockedRecovery(attempt: number) {
  const current = Math.min(Math.max(1, attempt), ORCHESTRA_MAX_BLOCKED_ATTEMPTS)
  const strategies = [
    "Inspect the actual diff and validation evidence, identify the concrete blocker, and make the next attempt address that blocker rather than redoing completed work.",
    "Narrow the unresolved scope, preserve valid existing changes, and delegate a root-cause fix with exact acceptance criteria and validation commands.",
    "Audit assumptions and interfaces around the blocker, then try a materially different implementation path instead of repeating the prior approach.",
    "Use the strongest available evidence to choose the lowest-risk alternative, explicitly covering prior failure reasons and remaining validation gaps.",
    "Stop implementation retries for this chat turn. Inspect the retained diff and report the root cause, viable implementation options, recommended option with rationale, exact unresolved work, and evidence needed to proceed.",
  ]
  return [
    `ORCHESTRA_RECOVERY: blocked attempt ${current} of ${ORCHESTRA_MAX_BLOCKED_ATTEMPTS}`,
    `RECOVERY_STRATEGY: ${strategies[current - 1]}`,
    current === ORCHESTRA_MAX_BLOCKED_ATTEMPTS
      ? "RETRY_POLICY: Terminal for this chat turn; do not start a sixth implementer attempt. A new user message starts a fresh attempt budget in the same session."
      : `RETRY_POLICY: At most ${ORCHESTRA_MAX_BLOCKED_ATTEMPTS - current} further implementer attempts remain for this chat turn.`,
  ].join("\n")
}

export function orchestraHandoffInstruction(role: string): string | undefined {
  const contract = orchestraContracts[role as OrchestraRole]
  if (!contract) return undefined

  return [
    `First line: STATUS: one of ${contract.statuses.join(", ")}`,
    "Then provide a substantive, readable handoff. The headings below are recommended but not required when equivalent information is clear:",
    ...contract.fields.map((field) => `Suggested field: ${field}: <text>`),
    `Success statuses (${[...orchestraSuccessStatuses].join(", ")}) require a concrete validation or execution result, such as the exact command and observed outcome. Never claim success from an intended patch, an unexecuted test, or a summary alone.`,
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
      "NEXT_ACTION: The Lead must inspect the raw output and follow the task runtime's ORCHESTRA_RECOVERY strategy and retry policy.",
    ].join("\n")
  if (role === "orchestra-tester")
    return [
      "STATUS: blocked",
      `CHANGES: ${summary}`,
      "REASON: invalid_handoff",
      "VALIDATION: The specialist task ended without a complete recognized handoff.",
      `FAILURES: Raw specialist output follows:\n${raw}`,
      "NEXT_ACTION: The Lead must inspect the raw output and follow the task runtime's ORCHESTRA_RECOVERY strategy and retry policy.",
    ].join("\n")

  return [
    "STATUS: blocked",
    `CHANGES: ${summary}`,
    "REASON: invalid_handoff",
    "VALIDATION: The specialist task ended without a complete recognized handoff.",
    `RISKS: Raw specialist output follows:\n${raw}`,
    "NEXT_ACTION: The Lead must inspect the raw output and follow the task runtime's ORCHESTRA_RECOVERY strategy and retry policy.",
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
