import * as Tool from "./tool"
import DESCRIPTION from "./task.txt"
import { ToolJsonSchema } from "./json-schema"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { BackgroundJob } from "@/background/job"
import { Session } from "@/session/session"
import { SessionID, MessageID } from "../session/schema"
import { MessageV2 } from "../session/message-v2"
import { Agent } from "../agent/agent"
import { deriveSubagentSessionPermission } from "../agent/subagent-permissions"
import * as Delegation from "../agent/delegation"
import { Permission } from "../permission"
import type { SessionPrompt } from "../session/prompt"
import { Config } from "@/config/config"
import { Provider } from "@/provider/provider"
import { Cause, Effect, Exit, Schema, Scope, SynchronizedRef } from "effect"
import { EffectBridge } from "@/effect/bridge"
import { RuntimeFlags } from "@/effect/runtime-flags"
import { Database } from "@opencode-ai/core/database/database"
// [CUSTOM] Model override support
import {
  resolveFinalModel,
  orchestraConcurrencyError,
  workflowTaskAccessError,
  orchestraHandoffInstruction,
  orchestraBlockedRecovery,
  orchestraRoleStatus,
  isOrchestraLead,
  normalizeOrchestraRoleOutput,
  resolveModelOverride,
  validateOrchestraRoleOutput,
  type ModelOverride,
} from "./model-override"

export interface TaskPromptOps {
  cancel(sessionID: SessionID): Effect.Effect<void>
  resolvePromptParts(template: string): Effect.Effect<SessionPrompt.PromptInput["parts"]>
  prompt(input: SessionPrompt.PromptInput): Effect.Effect<SessionV1.WithParts>
}

const ORCHESTRA_WORKFLOW_METADATA = "opencode.orchestra.workflow"

type OrchestraWorkflowEntry = {
  taskID: string
  role: string
  status?: string
  requestedModel?: string
  resolvedModel: string
  variant?: string
  startedAt: number
  finishedAt?: number
}

type OrchestraWorkflowLedger = {
  version: 1
  phase: string
  delegations: Record<string, number>
  latest: Record<string, OrchestraWorkflowEntry>
  reviewApproved: boolean
  testsPassed: boolean
  releaseReady: boolean
  updatedAt: number
}

function orchestraLedger(value: unknown): OrchestraWorkflowLedger {
  if (value && typeof value === "object" && (value as OrchestraWorkflowLedger).version === 1) {
    return structuredClone(value as OrchestraWorkflowLedger)
  }
  return {
    version: 1,
    phase: "planning",
    delegations: {},
    latest: {},
    reviewApproved: false,
    testsPassed: false,
    releaseReady: false,
    updatedAt: Date.now(),
  }
}

function orchestraPhase(role: string, status?: string) {
  if (role === "orchestra-implementer") return status === "blocked" ? "implementation_blocked" : "implementation"
  if (role === "orchestra-reviewer") return status === "approved" ? "review_approved" : "review"
  if (role === "orchestra-tester") return status === "passed" ? "tests_passed" : "testing"
  return "analysis"
}

const id = "task"
const BACKGROUND_DESCRIPTION = [
  "Background mode: background=true launches the subagent asynchronously and returns immediately.",
  "Foreground is the default; use it when you need the result before continuing.",
  "Use background only for independent work that can run while you continue elsewhere.",
  "You will be notified automatically when it finishes.",
].join(" ")
const BACKGROUND_STARTED = [
  "The task is working in the background. You will be notified automatically when it finishes.",
  "DO NOT sleep, poll for progress, ask the task for status, or duplicate this task's work — avoid working with the same files or topics it is using.",
  "Work on non-overlapping tasks, or briefly tell the user what you launched and end your response.",
].join("\n")
const BACKGROUND_UPDATED = [
  "Additional context sent to the running background task.",
  "The task is still working in the background. You will be notified automatically when it finishes.",
  "DO NOT sleep, poll for progress, ask the task for status, or duplicate this task's work — avoid working with the same files or topics it is using.",
  "Work on non-overlapping tasks, or briefly tell the user what you sent and end your response.",
].join("\n")
const HANDOFF_TASK_MAX_CHARS = 8_000
const HANDOFF_EVIDENCE_MAX_CHARS = 12_000
const HANDOFF_TOOL_OUTPUT_MAX_CHARS = 2_000

function handoffRepairEvidence(messages: SessionV1.WithParts[]) {
  const rows = messages.flatMap((message) =>
    message.parts.flatMap((part) => {
      if (part.type === "text" && message.info.role === "assistant" && part.text.trim()) {
        return [`ASSISTANT:\n${part.text.trim()}`]
      }
      if (part.type !== "tool") return []
      const result =
        part.state.status === "completed"
          ? part.state.output
          : part.state.status === "error"
            ? part.state.error
            : "[interrupted]"
      return [
        [
          `TOOL: ${part.tool}`,
          `INPUT: ${JSON.stringify(part.state.input)}`,
          `RESULT: ${result.slice(0, HANDOFF_TOOL_OUTPUT_MAX_CHARS)}`,
        ].join("\n"),
      ]
    }),
  )
  return rows.join("\n\n").slice(-HANDOFF_EVIDENCE_MAX_CHARS) || "(no tool or text evidence recorded)"
}

const BaseParameterFields = {
  description: Schema.String.annotate({ description: "A short (3-5 words) description of the task" }),
  prompt: Schema.String.annotate({ description: "The task for the agent to perform" }),
  subagent_type: Schema.String.annotate({ description: "The type of specialized agent to use for this task" }),
  task_id: Schema.optional(Schema.String).annotate({
    description:
      "This should only be set if you mean to resume a previous task (you can pass a prior task_id and the task will continue the same subagent session as before instead of creating a fresh one)",
  }),
  command: Schema.optional(Schema.String).annotate({ description: "The command that triggered this task" }),
  model: Schema.optional(Schema.String).annotate({
    description:
      "Optional one-off model override for the subagent in provider/model format (e.g. 'anthropic/claude-sonnet-4', 'openai/gpt-5'). Prefer the subagent_type's configured model for stable role workflows. If omitted, the configured agent model or parent session model is used.",
  }),
}

const BaseParameters = Schema.Struct(BaseParameterFields)

export const Parameters = Schema.Struct({
  ...BaseParameterFields,
  background: Schema.optional(Schema.Boolean).annotate({
    description:
      "Run the agent in the background. You will be notified when it completes. DO NOT sleep, poll, or proactively check on its progress",
  }),
})

function renderOutput(input: {
  sessionID: SessionID
  state: "running" | "completed" | "error"
  summary?: string
  text: string
}) {
  const tag = input.state === "error" ? "task_error" : "task_result"
  return [
    `<task id="${input.sessionID}" state="${input.state}">`,
    ...(input.summary ? [`<summary>${input.summary}</summary>`] : []),
    `<${tag}>`,
    input.text,
    `</${tag}>`,
    "</task>",
  ].join("\n")
}

export const TaskTool = Tool.define(
  id,
  Effect.gen(function* () {
    const agent = yield* Agent.Service
    const background = yield* BackgroundJob.Service
    const config = yield* Config.Service
    const sessions = yield* Session.Service
    const scope = yield* Scope.Scope
    const flags = yield* RuntimeFlags.Service
    const database = yield* Database.Service
    const provider = yield* Provider.Service
    const orchestraSlots = yield* SynchronizedRef.make(new Map<string, Record<string, number>>())
    const orchestraBlockedAttempts = yield* SynchronizedRef.make(
      new Map<string, { messageID: MessageID; attempts: number }>(),
    )
    const orchestraLedgerUpdates = yield* SynchronizedRef.make(0)

    const updateOrchestraLedger = Effect.fn("TaskTool.updateOrchestraLedger")(function* (
      parentID: SessionID,
      entry: OrchestraWorkflowEntry,
    ) {
      yield* SynchronizedRef.updateEffect(orchestraLedgerUpdates, () =>
        Effect.gen(function* () {
          const parent = yield* sessions.get(parentID)
          const ledger = orchestraLedger(parent.metadata?.[ORCHESTRA_WORKFLOW_METADATA])
          const previous = ledger.latest[entry.role]
          ledger.delegations[entry.role] = (ledger.delegations[entry.role] ?? 0) + (entry.finishedAt ? 0 : 1)
          ledger.latest[entry.role] = entry.finishedAt ? { ...previous, ...entry } : entry
          ledger.phase = orchestraPhase(entry.role, entry.status)
          if (!entry.finishedAt) {
            if (entry.role === "orchestra-implementer") {
              ledger.reviewApproved = false
              ledger.testsPassed = false
            }
            if (entry.role === "orchestra-reviewer") {
              ledger.reviewApproved = false
              ledger.testsPassed = false
            }
            if (entry.role === "orchestra-tester") ledger.testsPassed = false
          } else {
            if (entry.role === "orchestra-reviewer") ledger.reviewApproved = entry.status === "approved"
            if (entry.role === "orchestra-tester") ledger.testsPassed = entry.status === "passed"
          }
          ledger.releaseReady = ledger.reviewApproved && ledger.testsPassed
          ledger.updatedAt = Date.now()
          yield* sessions.setMetadata({
            sessionID: parentID,
            metadata: { ...Delegation.userMetadata(parent.metadata), [ORCHESTRA_WORKFLOW_METADATA]: ledger },
          })
          return Date.now()
        }),
      )
    })

    const applyOrchestraRecovery = Effect.fn("TaskTool.applyOrchestraRecovery")(function* (
      parentID: SessionID,
      messageID: MessageID,
      role: string,
      output: string,
    ) {
      if (role === "orchestra-reviewer" || role === "orchestra-tester") {
        yield* SynchronizedRef.update(orchestraBlockedAttempts, (state) => {
          if (state.get(parentID)?.messageID !== messageID) return state
          const next = new Map(state)
          next.delete(parentID)
          return next
        })
        return output
      }
      if (role !== "orchestra-implementer") return output
      const status = orchestraRoleStatus(role, output)
      if (status === "complete") {
        yield* SynchronizedRef.update(orchestraBlockedAttempts, (state) => {
          if (state.get(parentID)?.messageID !== messageID) return state
          const next = new Map(state)
          next.delete(parentID)
          return next
        })
        return output
      }
      if (status !== "blocked") return output
      const attempt = yield* SynchronizedRef.modify(orchestraBlockedAttempts, (state) => {
        const current = state.get(parentID)
        const value = Math.min((current?.messageID === messageID ? current.attempts : 0) + 1, 5)
        return [value, new Map(state).set(parentID, { messageID, attempts: value })] as const
      })
      return `${output}\n\n${orchestraBlockedRecovery(attempt)}`
    })

    const reserveOrchestraSlot = Effect.fn("TaskTool.reserveOrchestraSlot")(function* (parentID: SessionID, role: string) {
      const error = yield* SynchronizedRef.modify(orchestraSlots, (slots) => {
        const counts = slots.get(parentID) ?? {}
        const failure = orchestraConcurrencyError(counts, role)
        if (failure) return [failure, slots] as const
        return [
          undefined,
          new Map(slots).set(parentID, { ...counts, [role]: (counts[role] ?? 0) + 1 }),
        ] as const
      })
      if (error) return yield* Effect.fail(new Error(error))
    })

    const releaseOrchestraSlot = Effect.fn("TaskTool.releaseOrchestraSlot")(function* (
      parentID: SessionID,
      role: string,
    ) {
      yield* SynchronizedRef.update(orchestraSlots, (slots) => {
        const counts = slots.get(parentID)
        if (!counts) return slots
        const nextCount = (counts[role] ?? 0) - 1
        const next = { ...counts }
        if (nextCount > 0) next[role] = nextCount
        else delete next[role]
        const updated = new Map(slots)
        if (Object.keys(next).length === 0) updated.delete(parentID)
        else updated.set(parentID, next)
        return updated
      })
    })

    const run = Effect.fn("TaskTool.execute")(function* (
      params: Schema.Schema.Type<typeof Parameters>,
      ctx: Tool.Context,
    ) {
      const cfg = yield* config.get()
      const runInBackground = params.background === true
      if (runInBackground && !flags.experimentalBackgroundSubagents) {
        return yield* Effect.fail(
          new Error("Background subagents require OPENCODE_EXPERIMENTAL_BACKGROUND_SUBAGENTS=true"),
        )
      }

      const parent = yield* sessions.get(ctx.sessionID)
      const accessError = workflowTaskAccessError(parent.agent ?? ctx.agent, params.subagent_type)
      if (accessError) return yield* Effect.fail(new Error(accessError))
      const caller = yield* agent.get(parent.agent ?? ctx.agent)
      if (!caller) return yield* Effect.fail(new Error(`Unknown caller agent: ${parent.agent ?? ctx.agent}`))
      const taskPermission = Permission.evaluate(
        "task",
        params.subagent_type,
        Permission.merge(caller.permission, parent.permission ?? []),
      )
      if (taskPermission.action === "deny") {
        return yield* Effect.fail(new Error(`${caller.name} is not permitted to delegate to ${params.subagent_type}.`))
      }
      let current = parent
      let depth = 0
      while (current.parentID) {
        depth++
        current = yield* sessions.get(current.parentID)
      }
      if (depth >= (cfg.subagent_depth ?? 1)) {
        return yield* Effect.fail(
          new Error(
            `Subagent depth limit reached (${cfg.subagent_depth ?? 1}). Increase "subagent_depth" to allow nested subagents.`,
          ),
        )
      }

      if (!ctx.extra?.bypassAgentCheck) {
        yield* ctx.ask({
          permission: id,
          patterns: [params.subagent_type],
          always: ["*"],
          metadata: {
            description: params.description,
            subagent_type: params.subagent_type,
          },
        })
      }

      const next = yield* agent.get(params.subagent_type)
      if (!next) {
        return yield* Effect.fail(new Error(`Unknown agent type: ${params.subagent_type} is not a valid agent type`))
      }

      // [CUSTOM] Model override support
      let modelOverride: ModelOverride | undefined
      if (params.model) {
        if (parent.agent === "orchestra" || parent.agent === "research") {
          const workflow = parent.agent === "orchestra" ? "orchestra" : "Research"
          return yield* Effect.fail(
            new Error(`The ${workflow} Lead cannot override specialist models; configure each role agent model instead.`),
          )
        }
        const result = yield* resolveModelOverride(params.model, provider)
        if (!result.success) {
          return yield* Effect.fail(new Error(result.error))
        }
        modelOverride = result.model

        // Permission check for model override
        if (!ctx.extra?.bypassAgentCheck) {
          yield* ctx.ask({
            permission: "model_override",
            patterns: [params.model],
            always: ["*"],
            metadata: {
              description: params.description,
              subagent_type: params.subagent_type,
              model: params.model,
            },
          })
        }
      }

      const session = params.task_id
        ? yield* sessions.get(SessionID.make(params.task_id)).pipe(Effect.catchCause(() => Effect.succeed(undefined)))
        : undefined
      if (
        session &&
        !Delegation.matches(session, { kind: "delegated-task", parentID: ctx.sessionID, agent: next.name })
      ) {
        return yield* Effect.fail(
          new Error(
            `Cannot resume task ${session.id}: it is not a delegated ${next.name} task owned by this parent session.`,
          ),
        )
      }
      const childPermission = deriveSubagentSessionPermission({
        parentSessionPermission: parent.permission ?? [],
        subagent: next,
      })
      const childToolDenies = [
        ...(next.permission.some((rule) => rule.permission === "todowrite")
          ? []
          : [{ permission: "todowrite" as const, pattern: "*" as const, action: "deny" as const }]),
        ...(next.permission.some((rule) => rule.permission === id)
          ? []
          : [{ permission: id, pattern: "*" as const, action: "deny" as const }]),
        ...(cfg.experimental?.primary_tools?.map((permission) => ({
          permission,
          pattern: "*" as const,
          action: "deny" as const,
        })) ?? []),
      ]
      const nextSession =
        session ??
        (yield* sessions.create({
          parentID: ctx.sessionID,
          title: params.description + ` (@${next.name} subagent)`,
          agent: next.name,
          internalMetadata: Delegation.metadata({ kind: "delegated-task", parentID: ctx.sessionID, agent: next.name }),
          permission: [
            ...childPermission,
            ...childToolDenies.filter(
              (deny) =>
                !childPermission.some(
                  (rule) =>
                    rule.permission === deny.permission && rule.pattern === deny.pattern && rule.action === deny.action,
                ),
            ),
          ],
        }))
      if (session) yield* sessions.setPermission({ sessionID: session.id, permission: [...childPermission, ...childToolDenies] })
      const existingJob = yield* background.get(nextSession.id)
      const isRunningContinuation = existingJob?.status === "running"
      if (isRunningContinuation && isOrchestraLead(parent.agent ?? ctx.agent)) {
        return yield* Effect.fail(
          new Error(`Cannot continue running Orchestra task ${nextSession.id}; wait for its handoff or cancel it first.`),
        )
      }

      const msg = yield* MessageV2.get({ sessionID: ctx.sessionID, messageID: ctx.messageID }).pipe(
        Effect.provideService(Database.Service, database),
        Effect.orDie,
      )
      if (msg.info.role !== "assistant") return yield* Effect.fail(new Error("Not an assistant message"))
      const variant = msg.info.variant

      // [CUSTOM] Resolve final model with fallback chain.
      const agentModel = parent.agent === "orchestra-custom" ? undefined : next.model
      const model = resolveFinalModel(modelOverride, agentModel, {
        providerID: msg.info.providerID,
        modelID: msg.info.modelID,
      })
      const metadata = {
        parentSessionId: ctx.sessionID,
        sessionId: nextSession.id,
        model,
        ...(runInBackground ? { background: true } : {}),
      }

      // [CUSTOM] Variant precedence: an explicit model override has no variant;
      // a specialist with its own configured model keeps that agent's configured
      // variant; inheriting the parent model keeps the parent's variant.
      const childVariant = modelOverride ? undefined : agentModel ? (next.variant ?? undefined) : variant

      const workflowEntry = {
        taskID: nextSession.id,
        role: next.name,
        requestedModel: params.model,
        resolvedModel: `${model.providerID}/${model.modelID}`,
        variant: childVariant,
        startedAt: Date.now(),
      } satisfies OrchestraWorkflowEntry
      if (isOrchestraLead(parent.agent ?? ctx.agent)) {
        yield* updateOrchestraLedger(ctx.sessionID, workflowEntry)
      }

      yield* ctx.metadata({
        title: params.description,
        metadata,
      })

      const ops = ctx.extra?.promptOps as TaskPromptOps
      if (!ops) return yield* Effect.fail(new Error("TaskTool requires promptOps in ctx.extra"))

      const runTask = Effect.fn("TaskTool.runTask")(function* () {
        const parts = yield* ops.resolvePromptParts(params.prompt)
        const result = yield* ops.prompt({
          messageID: MessageID.ascending(),
          sessionID: nextSession.id,
          model: {
            modelID: model.modelID,
            providerID: model.providerID,
          },
          // [CUSTOM] Discard variant when model override is set
          variant: childVariant,
          agent: next.name,
          parts,
        })
        if (result.info.role === "assistant" && result.info.error) {
          const message = "message" in result.info.error.data ? result.info.error.data.message : undefined
          return yield* Effect.fail(new Error(typeof message === "string" ? message : "Subagent failed"))
        }
        // Providers can split a specialist's final response across multiple
        // text parts around tool calls. Validate the complete text transcript,
        // not only the last fragment.
        const output = result.parts
          .filter((item): item is Extract<typeof item, { type: "text" }> => item.type === "text")
          .map((item) => item.text)
          .join("\n")
        const instruction = orchestraHandoffInstruction(next.name)
        if (!instruction || !validateOrchestraRoleOutput(next.name, output))
          return yield* applyOrchestraRecovery(ctx.sessionID, ctx.messageID, next.name, output)

        const evidence = handoffRepairEvidence(yield* sessions.messages({ sessionID: nextSession.id }))
        const repair = yield* ops
          .prompt({
            messageID: MessageID.ascending(),
            sessionID: nextSession.id,
            model: {
              modelID: model.modelID,
              providerID: model.providerID,
            },
            childVariant,
            agent: next.name,
            tools: { "*": false },
            parts: [
              {
                type: "text",
                text: [
                  "Return the final Orchestra handoff now based only on work and validation already completed.",
                  "Do not inspect, edit, run commands, or call tools. Respond with plain text only.",
                  "Do not claim success for incomplete or unvalidated work; choose the role's blocked or failed status and state what remains.",
                  instruction,
                  `Original task:\n${params.prompt.slice(0, HANDOFF_TASK_MAX_CHARS)}`,
                  `Recorded work and validation evidence:\n${evidence}`,
                  `The previous response was incomplete. Preserve this evidence in the appropriate field:\n${output || "(no text response)"}`,
                ].join("\n\n"),
              },
            ],
          })
          .pipe(Effect.exit)
        if (Exit.isSuccess(repair) && repair.value.info.role === "assistant" && !repair.value.info.error) {
          const rendered = repair.value.parts
            .filter((item): item is Extract<typeof item, { type: "text" }> => item.type === "text")
            .map((item) => item.text)
            .join("\n")
          if (!validateOrchestraRoleOutput(next.name, rendered))
            return yield* applyOrchestraRecovery(ctx.sessionID, ctx.messageID, next.name, rendered)
        }

        const repairError =
          Exit.isFailure(repair)
            ? String(Cause.squash(repair.cause))
            : repair.value.info.role === "assistant" &&
                repair.value.info.error &&
                "message" in repair.value.info.error.data
              ? repair.value.info.error.data.message
              : undefined
        const normalized = normalizeOrchestraRoleOutput(
          next.name,
          [output, repairError ? `Native handoff repair failed: ${repairError}` : "Native handoff repair returned no output."]
            .filter(Boolean)
            .join("\n"),
        )
        return yield* applyOrchestraRecovery(ctx.sessionID, ctx.messageID, next.name, normalized)
      })
      const trackedTask = () =>
        runTask().pipe(
          Effect.tap((output) => {
            if (!isOrchestraLead(parent.agent ?? ctx.agent)) return Effect.void
            return updateOrchestraLedger(ctx.sessionID, {
              ...workflowEntry,
              status: orchestraRoleStatus(next.name, output),
              finishedAt: Date.now(),
            })
          }),
        )

      const inject = Effect.fn("TaskTool.injectBackgroundResult")(function* (
        state: "completed" | "error",
        text: string,
      ) {
        const currentParent = yield* sessions.get(ctx.sessionID)
        yield* ops
          .prompt({
            sessionID: ctx.sessionID,
            agent: currentParent.agent ?? ctx.agent,
            variant,
            parts: [
              {
                type: "text",
                synthetic: true,
                text: renderOutput({
                  sessionID: nextSession.id,
                  state,
                  summary:
                    state === "completed"
                      ? `Background task completed: ${params.description}`
                      : `Background task failed: ${params.description}`,
                  text,
                }),
              },
            ],
          })
          .pipe(Effect.ignore, Effect.forkIn(scope, { startImmediately: true }))
      })

      const notify = Effect.fn("TaskTool.notifyBackgroundResult")(function* (jobID: string) {
        yield* background.wait({ id: jobID }).pipe(
          Effect.flatMap((result) => {
            if (result.info?.status === "completed") return inject("completed", result.info.output ?? "")
            if (result.info?.status === "error") return inject("error", result.info.error ?? "")
            return Effect.void
          }),
          Effect.forkIn(scope, { startImmediately: true }),
        )
      })

      if (yield* background.extend({ id: nextSession.id, run: trackedTask() })) {
        return {
          title: params.description,
          metadata: {
            ...metadata,
            background: true,
            jobId: nextSession.id,
          },
          output: renderOutput({
            sessionID: nextSession.id,
            state: "running",
            summary: "Background task updated",
            text: BACKGROUND_UPDATED,
          }),
        }
      }

      const reserved = isOrchestraLead(parent.agent ?? ctx.agent)
      if (reserved) yield* reserveOrchestraSlot(ctx.sessionID, next.name)

      const info = yield* background.start({
        id: nextSession.id,
        type: id,
        title: params.description,
        metadata,
        onPromote: Effect.all([
          ctx.metadata({
            title: params.description,
            metadata: { ...metadata, background: true, jobId: nextSession.id },
          }),
          notify(nextSession.id),
        ]),
        run: trackedTask().pipe(Effect.onInterrupt(() => ops.cancel(nextSession.id))),
      })
      if (reserved)
        yield* background
          .wait({ id: nextSession.id })
          .pipe(
            Effect.ensuring(releaseOrchestraSlot(ctx.sessionID, next.name)),
            Effect.forkIn(scope, { startImmediately: true }),
          )

      function backgroundResult() {
        return {
          title: params.description,
          metadata: {
            ...metadata,
            background: true,
            jobId: info.id,
          },
          output: renderOutput({
            sessionID: nextSession.id,
            state: "running",
            summary: "Background task started",
            text: BACKGROUND_STARTED,
          }),
        }
      }

      if (runInBackground) {
        yield* notify(info.id)
        return backgroundResult()
      }

      const runCancel = yield* EffectBridge.make()
      const cancel = ops.cancel(nextSession.id)

      function onAbort() {
        runCancel.fork(cancel)
      }

      return yield* Effect.acquireUseRelease(
        Effect.sync(() => {
          ctx.abort.addEventListener("abort", onAbort)
        }),
        () =>
          Effect.gen(function* () {
            const result = yield* Effect.raceFirst(
              background.wait({ id: nextSession.id }).pipe(Effect.map((waited) => waited.info)),
              background.waitForPromotion(nextSession.id),
            )
            if (result?.metadata?.background === true) return backgroundResult()
            if (result?.status === "error")
              return yield* Effect.fail(
                new Error(renderOutput({ sessionID: nextSession.id, state: "error", text: result.error ?? "Task failed" })),
              )
            if (result?.status === "cancelled")
              return yield* Effect.fail(
                new Error(
                  renderOutput({ sessionID: nextSession.id, state: "error", text: "Task was cancelled/aborted" }),
                ),
              )
            return {
              title: params.description,
              metadata,
              output: renderOutput({ sessionID: nextSession.id, state: "completed", text: result?.output ?? "" }),
            }
          }),
        (_, exit) =>
          Effect.gen(function* () {
            if (Exit.hasInterrupts(exit))
              yield* Effect.all([cancel, background.cancel(nextSession.id)], { discard: true })
          }).pipe(
            Effect.ensuring(
              Effect.sync(() => {
                ctx.abort.removeEventListener("abort", onAbort)
              }),
            ),
          ),
      )
    })

    return {
      description: flags.experimentalBackgroundSubagents
        ? [DESCRIPTION, BACKGROUND_DESCRIPTION].join("\n\n")
        : DESCRIPTION,
      parameters: Parameters,
      jsonSchema: flags.experimentalBackgroundSubagents ? undefined : ToolJsonSchema.fromSchema(BaseParameters),
      execute: (params: Schema.Schema.Type<typeof Parameters>, ctx: Tool.Context) =>
        run(params, ctx).pipe(Effect.orDie),
    }
  }),
)
