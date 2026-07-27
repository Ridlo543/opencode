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
const orchestraRoles = new Set<string>(["orchestra-implementer", "orchestra-reviewer", "orchestra-tester"])

export function isOrchestraRole(agent: string): agent is OrchestraRole {
  return orchestraRoles.has(agent)
}

export function orchestraTaskAccessError(caller: string, target: string): string | undefined {
  if (caller === "orchestra") {
    if (isOrchestraRole(target)) return undefined
    return `The orchestra Lead must delegate using orchestra-implementer, orchestra-reviewer, or orchestra-tester; received ${target}.`
  }
  if (isOrchestraRole(target)) {
    return `${target} is a private Orchestra specialist and can only be delegated by the orchestra Lead.`
  }
  return undefined
}

// Match the native task fan-out while allowing any mix of specialist roles.
export const ORCHESTRA_MAX_CONCURRENT_TASKS = 4

export type OrchestraTaskCounts = Partial<Record<OrchestraRole, number>>

const handoffAliases = {
  CHANGES: [
    "FILES_CHANGED",
    "CHANGED_FILES",
    "FILES_MODIFIED",
    "MODIFIED_FILES",
    "IMPLEMENTATION",
    "IMPLEMENTATION_SUMMARY",
    "MODIFICATIONS",
    "WORK_COMPLETED",
  ],
  VALIDATION: [
    "TESTS",
    "TESTS_RUN",
    "TEST_RESULTS",
    "TESTING",
    "CHECKS",
    "CHECKS_RUN",
    "CHECK_RESULTS",
    "COMMANDS_RUN",
    "VERIFICATION",
    "VERIFICATION_RESULTS",
    "VALIDATION_RESULTS",
  ],
  RISKS: ["KNOWN_RISKS", "REMAINING_RISKS", "RESIDUAL_RISKS"],
  NEXT_ACTION: ["NEXT_ACTIONS", "NEXT_STEP", "NEXT_STEPS", "FOLLOW_UP", "FOLLOW_UP_ACTIONS"],
  FINDINGS: ["ISSUES", "ISSUES_FOUND", "REVIEW_FINDINGS", "REVIEW_RESULTS"],
  FAILURES: ["FAILED_TESTS", "FAILING_TESTS", "TEST_FAILURES", "TEST_ISSUES"],
} as const

const orchestraContracts = {
  "orchestra-implementer": {
    statuses: ["complete", "blocked"],
    fields: ["CHANGES", "VALIDATION", "RISKS", "NEXT_ACTION"],
    aliases: {
      CHANGES: handoffAliases.CHANGES,
      VALIDATION: handoffAliases.VALIDATION,
      RISKS: handoffAliases.RISKS,
      NEXT_ACTION: handoffAliases.NEXT_ACTION,
    },
  },
  "orchestra-reviewer": {
    statuses: ["approved", "needs_revision", "blocked"],
    fields: ["FINDINGS", "VALIDATION", "RISKS", "NEXT_ACTION"],
    aliases: {
      FINDINGS: handoffAliases.FINDINGS,
      VALIDATION: handoffAliases.VALIDATION,
      RISKS: handoffAliases.RISKS,
      NEXT_ACTION: handoffAliases.NEXT_ACTION,
    },
  },
  "orchestra-tester": {
    statuses: ["passed", "failed", "blocked"],
    fields: ["CHANGES", "VALIDATION", "FAILURES", "NEXT_ACTION"],
    aliases: {
      CHANGES: [...handoffAliases.CHANGES, "TEST_CHANGES", "TESTS_ADDED"],
      VALIDATION: handoffAliases.VALIDATION,
      FAILURES: handoffAliases.FAILURES,
      NEXT_ACTION: handoffAliases.NEXT_ACTION,
    },
  },
} satisfies Record<
  OrchestraRole,
  { statuses: string[]; fields: string[]; aliases: Record<string, readonly string[]> }
>

function normalizeHandoffHeading(value: string) {
  return value
    .replace(/^[\s#>*-]+/, "")
    .replace(/[*`~]/g, "")
    .trim()
    .replace(/[\s-]+/g, "_")
    .toUpperCase()
}

function parseHandoffSections(output: string, canonical: Map<string, string>) {
  const sections = new Map<string, string[]>()
  const structural = new Map<string, string[]>()
  let current: string | undefined
  let currentStructural: string | undefined
  let fence: string | undefined
  for (const line of output.split(/\r?\n/)) {
    const fenceMatch = line.match(/^\s*(```+|~~~+)/)
    if (fenceMatch) {
      fence = fence ? undefined : fenceMatch[1]?.[0]
      continue
    }
    if (fence || /^\s*>/.test(line)) continue
    const match = line.match(/^\s*(?:[#[*-]+\s*)?(?:[*_`~]+\s*)?([A-Za-z][A-Za-z _-]*?)(?:\s*[*_`~]+)?\s*(?::\s*(.*))?$/)
    const rawHeading = match?.[1]?.trim() ?? ""
    const heading = normalizeHandoffHeading(rawHeading)
    const field = canonical.get(heading)
    const markdownHeading = /^\s*#{1,6}\s+/.test(line)
    const bullet = /^\s*[-*+]\s+/.test(line)
    const explicit = line.includes(":") || markdownHeading
    const protocolHeading = rawHeading === rawHeading.toUpperCase()
    const structuralHeading = explicit && (markdownHeading || (!bullet && protocolHeading))
    if (match && field && explicit) {
      current = field
      currentStructural = heading
      const value = match[2]?.trim()
      // Canonical and alias duplicates are one section; the final declaration wins.
      sections.set(field, value ? [value] : [])
      structural.set(heading, value ? [value] : [])
      continue
    }
    if (match && structuralHeading) {
      current = undefined
      currentStructural = heading
      const value = match[2]?.trim()
      structural.set(heading, value ? [value] : [])
      continue
    }
    // An explicit unknown heading must not become content for the preceding field.
    if (match && (markdownHeading || (line.includes(":") && !bullet && protocolHeading))) {
      current = undefined
      currentStructural = undefined
      continue
    }
    if (current && line.trim()) sections.get(current)?.push(line.trim())
    if (currentStructural && line.trim()) structural.get(currentStructural)?.push(line.trim())
  }
  return { canonical: sections, structural }
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

const supplementaryHandoffSections = new Set([
  "SUMMARY",
  "NOTES",
  "SCOPE",
  "CONTEXT",
  "DESIGN_INVARIANTS",
  "ASSUMPTIONS",
  "DETAILS",
])

function hasStructuralHandoff(sections: Map<string, string[]>) {
  const evidence = [...sections.entries()].filter(
    ([heading, lines]) => !supplementaryHandoffSections.has(heading) && hasMeaningfulHandoffValue(lines),
  )
  if (evidence.length < 4) return false
  const content = evidence
    .flatMap(([, lines]) => lines)
    .join("\n")
    .replace(/^\s*[-*+]\s*/gm, "")
    .trim()
  return content.length >= 40
}

export function orchestraConcurrencyError(counts: OrchestraTaskCounts, role: string): string | undefined {
  if (!isOrchestraRole(role)) return undefined

  const total = Object.values(counts).reduce((sum, value) => sum + (value ?? 0), 0)
  if (total >= ORCHESTRA_MAX_CONCURRENT_TASKS) {
    return `The orchestra Lead already has ${total} active specialist tasks (maximum ${ORCHESTRA_MAX_CONCURRENT_TASKS}). Wait for one to finish before delegating another.`
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

  // Use the final unquoted STATUS line outside code fences so examples and raw
  // transcripts cannot satisfy the contract. Parse only the handoff after it.
  const final = finalHandoffStatus(output)
  const status = final?.status
  const handoff = final?.handoff ?? ""
  const canonical = new Map<string, string>()
  const aliases = contract.aliases as Record<string, readonly string[]>
  for (const field of contract.fields) {
    canonical.set(field, field)
    for (const alias of aliases[field] ?? []) canonical.set(alias, field)
  }
  // SUMMARY is useful context but deliberately not a substitute for role evidence.
  canonical.set("SUMMARY", "SUMMARY")
  const sections = parseHandoffSections(handoff, canonical)
  const complete = contract.fields.every((field) => hasMeaningfulHandoffValue(sections.canonical.get(field)))
  const canonicalConsistent = contract.fields
    .filter((field) => sections.canonical.has(field))
    .every((field) => hasMeaningfulHandoffValue(sections.canonical.get(field)))
  const structural = canonicalConsistent && hasStructuralHandoff(sections.structural)
  if (status && contract.statuses.includes(status) && (complete || structural)) return undefined

  return `${role} must return a complete handoff with STATUS (${contract.statuses.join(", ")}) and fields ${contract.fields.join(", ")}.`
}

export function orchestraHandoffInstruction(role: string): string | undefined {
  const contract = orchestraContracts[role as OrchestraRole]
  if (!contract) return undefined

  return [
    `First line: STATUS: one of ${contract.statuses.join(", ")}`,
    ...contract.fields.map((field) => `Required field: ${field}: <text>`),
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
