import { afterEach, describe, expect } from "bun:test"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { Database } from "@opencode-ai/core/database/database"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { SessionProjector } from "@opencode-ai/core/session/projector"
import { Cause, Deferred, Effect, Exit, Fiber, Layer } from "effect"
import { Agent } from "../../src/agent/agent"
import * as Delegation from "../../src/agent/delegation"
import { BackgroundJob } from "@/background/job"
import { EventV2Bridge } from "@/event-v2-bridge"
import { Config } from "@/config/config"
import { CrossSpawnSpawner } from "@opencode-ai/core/cross-spawn-spawner"
import { Ripgrep } from "@opencode-ai/core/ripgrep"
import { Session } from "@/session/session"
import type { SessionPrompt } from "../../src/session/prompt"
import { MessageID, PartID, SessionID } from "../../src/session/schema"
import { SessionRunState } from "@/session/run-state"
import { SessionStatus } from "@/session/status"

import { TaskTool, type TaskPromptOps } from "../../src/tool/task"
import { Truncate } from "@/tool/truncate"
import { ToolRegistry } from "@/tool/registry"
import { RuntimeFlags } from "@/effect/runtime-flags"
import { Provider } from "@/provider/provider"
import { disposeAllInstances } from "../fixture/fixture"
import { testEffect } from "../lib/effect"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { ModelV2 } from "@opencode-ai/core/model"

afterEach(async () => {
  await disposeAllInstances()
})

const ref = {
  providerID: ProviderV2.ID.make("test"),
  modelID: ModelV2.ID.make("test-model"),
}

const layer = (flags: Partial<RuntimeFlags.Info> = {}) =>
  LayerNode.compile(
    LayerNode.group([
      Agent.node,
      BackgroundJob.node,
      EventV2Bridge.node,
      Config.node,
      CrossSpawnSpawner.node,
      Session.node,
      SessionProjector.node,
      SessionRunState.node,
      SessionStatus.node,
      Truncate.node,
      ToolRegistry.node,
      Database.node,
      RuntimeFlags.node,
      Ripgrep.node,
      Provider.node,
    ]),
    [[RuntimeFlags.node, RuntimeFlags.layer(flags)]],
  )

const it = testEffect(layer())
const background = testEffect(layer({ experimentalBackgroundSubagents: true }))

function defer<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

const seed = Effect.fn("TaskToolTest.seed")(function* (title = "Pinned") {
  const session = yield* Session.Service
  const chat = yield* session.create({ title })
  const user = yield* session.updateMessage({
    id: MessageID.ascending(),
    role: "user",
    sessionID: chat.id,
    agent: "build",
    model: ref,
    time: { created: Date.now() },
  })
  const assistant: SessionV1.Assistant = {
    id: MessageID.ascending(),
    role: "assistant",
    parentID: user.id,
    sessionID: chat.id,
    mode: "build",
    agent: "build",
    cost: 0,
    path: { cwd: "/tmp", root: "/tmp" },
    tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
    modelID: ref.modelID,
    providerID: ref.providerID,
    variant: "xhigh",
    time: { created: Date.now() },
  }
  yield* session.updateMessage(assistant)
  return { chat, assistant }
})


const seedAgent = Effect.fn("TaskToolTest.seedAgent")(function* (agent: string) {
  const session = yield* Session.Service
  const chat = yield* session.create({ title: `Agent ${agent}`, agent })
  const user = yield* session.updateMessage({
    id: MessageID.ascending(),
    role: "user",
    sessionID: chat.id,
    agent,
    model: ref,
    time: { created: Date.now() },
  })
  const assistant: SessionV1.Assistant = {
    id: MessageID.ascending(),
    role: "assistant",
    parentID: user.id,
    sessionID: chat.id,
    mode: agent,
    agent,
    cost: 0,
    path: { cwd: "/tmp", root: "/tmp" },
    tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
    modelID: ref.modelID,
    providerID: ref.providerID,
    time: { created: Date.now() },
  }
  yield* session.updateMessage(assistant)
  return { chat, assistant }
})

function stubOps(opts?: {
  onPrompt?: (input: SessionPrompt.PromptInput) => void
  text?: string
  error?: NonNullable<SessionV1.Assistant["error"]>
  toolError?: string
}): TaskPromptOps {
  return {
    cancel: () => Effect.void,
    resolvePromptParts: (template) => Effect.succeed([{ type: "text" as const, text: template }]),
    prompt: (input) =>
      Effect.sync(() => {
        opts?.onPrompt?.(input)
        return reply(input, opts?.text ?? "done", opts?.error, opts?.toolError)
      }),
  }
}

function reply(
  input: SessionPrompt.PromptInput,
  text: string,
  error?: NonNullable<SessionV1.Assistant["error"]>,
  toolError?: string,
): SessionV1.WithParts {
  const id = MessageID.ascending()
  return {
    info: {
      id,
      role: "assistant",
      parentID: input.messageID ?? MessageID.ascending(),
      sessionID: input.sessionID,
      mode: input.agent ?? "general",
      agent: input.agent ?? "general",
      cost: 0,
      path: { cwd: "/tmp", root: "/tmp" },
      tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
      modelID: input.model?.modelID ?? ref.modelID,
      providerID: input.model?.providerID ?? ref.providerID,
      time: { created: Date.now() },
      finish: "stop",
      error,
    },
    parts: [
      {
        id: PartID.ascending(),
        messageID: id,
        sessionID: input.sessionID,
        type: "text",
        text,
      },
      ...(toolError
        ? [
            {
              id: PartID.ascending(),
              messageID: id,
              sessionID: input.sessionID,
              type: "tool" as const,
              tool: "read",
              callID: "call-1",
              state: {
                status: "error" as const,
                input: { filePath: "/external" },
                error: toolError,
                time: { start: Date.now(), end: Date.now() },
              },
            },
          ]
        : []),
    ],
  }
}

function failedReply(input: SessionPrompt.PromptInput, message: string): SessionV1.WithParts {
  const result = reply(input, "")
  if (result.info.role !== "assistant") return result
  result.info.error = new SessionV1.APIError({ message, isRetryable: true }).toObject()
  return result
}

describe("tool.task", () => {
  it.instance("uses the configured orchestra role model instead of the parent model", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seedAgent("orchestra")
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const seen: SessionPrompt.PromptInput[] = []

      yield* def.execute(
        {
          description: "implement scoped change",
          prompt: "implement the requested change",
          subagent_type: "orchestra-implementer",
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "orchestra",
          abort: new AbortController().signal,
          extra: {
            promptOps: stubOps({
              onPrompt: (input) => seen.push(input),
              text: "STATUS: complete\nCHANGES: none\nVALIDATION: pass\nRISKS: none\nNEXT_ACTION: review",
            }),
          },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      expect(seen).toHaveLength(1)
      expect(seen[0]?.agent).toBe("orchestra-implementer")
      expect(seen[0]?.model?.providerID as string).toBe("9router")
      expect(seen[0]?.model?.modelID as string).toBe("ag/gemini-3-flash-agent")
    }),
    {
      config: {
        agent: {
          orchestra: { mode: "primary" },
          "orchestra-implementer": {
            mode: "subagent",
            model: "9router/ag/gemini-3-flash-agent",
          },
        },
      },
    },
  )

  it.instance("allows only the Orchestra Lead to delegate to its assistant", () =>
    Effect.gen(function* () {
      const sessions = yield* Session.Service
      const lead = yield* seedAgent("orchestra")
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const result = yield* def.execute(
        {
          description: "scan risks",
          prompt: "summarize risks without editing",
          subagent_type: "orchestra-assistant-specialist",
        },
        {
          sessionID: lead.chat.id,
          messageID: lead.assistant.id,
          agent: "orchestra",
          abort: new AbortController().signal,
          extra: { promptOps: stubOps({ text: "ADVISORY: inspect the cache key" }) },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      expect((yield* sessions.get(result.metadata.sessionId)).agent).toBe("orchestra-assistant-specialist")
      expect(result.output).toContain("ADVISORY: inspect the cache key")

      const role = yield* seedAgent("orchestra-implementer")
      const denied = yield* def
        .execute(
          {
            description: "nested scan",
            prompt: "inspect implementation risks",
            subagent_type: "orchestra-assistant-specialist",
          },
          {
            sessionID: role.chat.id,
            messageID: role.assistant.id,
            agent: "orchestra-implementer",
            abort: new AbortController().signal,
            extra: { promptOps: stubOps() },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )
        .pipe(Effect.exit)

      expect(Exit.isFailure(denied)).toBe(true)
      if (Exit.isFailure(denied)) expect(String(Cause.squash(denied.cause))).toContain("cannot delegate tasks")
    }),
    {
      config: {
        subagent_depth: 2,
        agent: {
          orchestra: { mode: "primary" },
          "orchestra-assistant-specialist": { mode: "subagent" },
          "orchestra-implementer": {
            mode: "subagent",
            permission: { task: { "orchestra-assistant-specialist": "allow" } },
          },
        },
      },
    },
  )

  it.instance("allows only the Research Lead to delegate to Research specialists", () =>
    Effect.gen(function* () {
      const sessions = yield* Session.Service
      const lead = yield* seedAgent("research")
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const result = yield* def.execute(
        {
          description: "search primary literature",
          prompt: "build the source register for the assigned evidence lane",
          subagent_type: "research-scout",
        },
        {
          sessionID: lead.chat.id,
          messageID: lead.assistant.id,
          agent: "research",
          abort: new AbortController().signal,
          extra: { promptOps: stubOps({ text: "STATUS: complete\nSOURCES_ADDED: doi:10/example" }) },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      expect((yield* sessions.get(result.metadata.sessionId)).agent).toBe("research-scout")
      expect(result.output).toContain("SOURCES_ADDED: doi:10/example")

      const role = yield* seedAgent("research-scout")
      const denied = yield* def
        .execute(
          {
            description: "nested analysis",
            prompt: "analyze the sources",
            subagent_type: "research-analyst",
          },
          {
            sessionID: role.chat.id,
            messageID: role.assistant.id,
            agent: "research-scout",
            abort: new AbortController().signal,
            extra: { promptOps: stubOps() },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )
        .pipe(Effect.exit)

      expect(Exit.isFailure(denied)).toBe(true)
      if (Exit.isFailure(denied)) expect(String(Cause.squash(denied.cause))).toContain("cannot delegate tasks")
    }),
    {
      config: {
        subagent_depth: 2,
        agent: {
          research: { mode: "primary" },
          "research-scout": { mode: "subagent" },
          "research-analyst": {
            mode: "subagent",
            permission: { task: { "research-analyst": "allow" } },
          },
        },
      },
    },
  )

  it.instance("rejects model overrides from the orchestra Lead", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seedAgent("orchestra")
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const exit = yield* def
        .execute(
          {
            description: "override specialist",
            prompt: "implement the requested change",
            subagent_type: "orchestra-implementer",
            model: "test/other-model",
          },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "orchestra",
            abort: new AbortController().signal,
            extra: { promptOps: stubOps() },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )
        .pipe(Effect.exit)

      expect(Exit.isFailure(exit)).toBe(true)
    }),
    {
      config: {
        agent: {
          orchestra: { mode: "primary" },
          "orchestra-implementer": { mode: "subagent" },
        },
      },
    },
  )

  it.instance("allows model overrides from the custom orchestra Lead", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seedAgent("orchestra-custom")
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const prompts: SessionPrompt.PromptInput[] = []

      yield* def.execute(
        {
          description: "override custom specialist",
          prompt: "implement the requested change",
          subagent_type: "orchestra-implementer",
          model: "test-provider/test-model",
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "orchestra-custom",
          abort: new AbortController().signal,
          extra: {
            promptOps: {
              ...stubOps(),
              prompt: (input) => {
                prompts.push(input)
                return Effect.succeed(
                  reply(
                    input,
                    "STATUS: complete\nCHANGES: none\nVALIDATION: pass\nRISKS: none\nNEXT_ACTION: review",
                  ),
                )
              },
            } satisfies TaskPromptOps,
          },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      expect(prompts).toHaveLength(1)
      expect(prompts[0]?.model).toEqual({
        providerID: ProviderV2.ID.make("test-provider"),
        modelID: ModelV2.ID.make("test-model"),
      })
    }),
    {
      config: {
        provider: {
          "test-provider": {
            name: "Test Provider",
            npm: "@ai-sdk/openai-compatible",
            env: [],
            models: {
              "test-model": {
                name: "Test Model",
                tool_call: true,
                limit: { context: 128_000, output: 4096 },
              },
            },
            options: { apiKey: "test-key" },
          },
        },
        agent: {
          "orchestra-custom": { mode: "primary", permission: { model_override: "allow" } },
          "orchestra-implementer": { mode: "subagent" },
        },
      },
    },
  )

  it.instance("inherits the Leader model for custom orchestra specialists without an override", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seedAgent("orchestra-custom")
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const prompts: SessionPrompt.PromptInput[] = []

      yield* def.execute(
        {
          description: "inherit custom Leader model",
          prompt: "review the requested change",
          subagent_type: "orchestra-reviewer",
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "orchestra-custom",
          abort: new AbortController().signal,
          extra: {
            promptOps: {
              ...stubOps(),
              prompt: (input) => {
                prompts.push(input)
                return Effect.succeed(
                  reply(
                    input,
                    "STATUS: approved\nFINDINGS: none\nVALIDATION: pass\nRISKS: none\nNEXT_ACTION: test",
                  ),
                )
              },
            } satisfies TaskPromptOps,
          },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      expect(prompts).toHaveLength(1)
      expect(prompts[0]?.model).toEqual(ref)
    }),
    {
      config: {
        agent: {
          "orchestra-custom": { mode: "primary", permission: { model_override: "allow" } },
          "orchestra-reviewer": { mode: "subagent", model: "opencode/big-pickle" },
        },
      },
    },
  )

  it.instance("persists flexible Orchestra quality state and model provenance without retry caps", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seedAgent("orchestra")
      const sessions = yield* Session.Service
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const outputs = [
        "STATUS: complete\nCHANGES: first implementation\nVALIDATION: pass\nRISKS: none\nNEXT_ACTION: review",
        "STATUS: needs_revision\nFINDINGS: one defect remains\nVALIDATION: reviewed\nRISKS: defect\nNEXT_ACTION: revise",
        "STATUS: complete\nCHANGES: revised implementation\nVALIDATION: pass\nRISKS: none\nNEXT_ACTION: review",
        "STATUS: approved\nFINDINGS: none after revision\nVALIDATION: reviewed\nRISKS: none\nNEXT_ACTION: test",
        "STATUS: failed\nCHANGES: tests added\nVALIDATION: one failure\nFAILURES: regression\nNEXT_ACTION: repair",
        "STATUS: passed\nCHANGES: tests retained\nVALIDATION: all pass\nFAILURES: none\nNEXT_ACTION: complete",
      ]
      const roles = [
        "orchestra-implementer",
        "orchestra-reviewer",
        "orchestra-implementer",
        "orchestra-reviewer",
        "orchestra-tester",
        "orchestra-tester",
      ]

      for (let index = 0; index < roles.length; index++) {
        yield* def.execute(
          {
            description: `workflow step ${index}`,
            prompt: "perform the next evidence-based workflow step",
            subagent_type: roles[index]!,
          },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "orchestra",
            abort: new AbortController().signal,
            extra: { promptOps: stubOps({ text: outputs[index] }) },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )
      }

      const ledger = (yield* sessions.get(chat.id)).metadata?.["opencode.orchestra.workflow"] as any
      expect(ledger.delegations["orchestra-implementer"]).toBe(2)
      expect(ledger.delegations["orchestra-reviewer"]).toBe(2)
      expect(ledger.delegations["orchestra-tester"]).toBe(2)
      expect(ledger.latest["orchestra-reviewer"].status).toBe("approved")
      expect(ledger.latest["orchestra-tester"].status).toBe("passed")
      expect(ledger.latest["orchestra-tester"].resolvedModel).toBe("test/test-model")
      expect(ledger.reviewApproved).toBe(true)
      expect(ledger.testsPassed).toBe(true)
      expect(ledger.releaseReady).toBe(true)
    }),
    {
      config: {
        agent: {
          orchestra: { mode: "primary" },
          "orchestra-implementer": { mode: "subagent" },
          "orchestra-reviewer": { mode: "subagent" },
          "orchestra-tester": { mode: "subagent" },
        },
      },
    },
  )

  it.instance("converts an invalid orchestra handoff to a blocked result", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seedAgent("orchestra")
      const tool = yield* TaskTool
      const def = yield* tool.init()

      const result = yield* def.execute(
        {
          description: "review scoped change",
          prompt: "review the requested change",
          subagent_type: "orchestra-reviewer",
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "orchestra",
          abort: new AbortController().signal,
          extra: { promptOps: stubOps({ text: "The reviewer stopped after reading the diff." }) },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      expect(result.output).toContain('state="completed"')
      expect(result.output).toContain("STATUS: blocked")
      expect(result.output).toContain("The reviewer stopped after reading the diff.")
    }),
    {
      config: {
        agent: {
          orchestra: { mode: "primary" },
          "orchestra-reviewer": {
            mode: "subagent",
          },
        },
      },
    },
  )

  it.instance("tracks five blocked implementer attempts and resets after completion", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seedAgent("orchestra")
      const tool = yield* TaskTool
      const def = yield* tool.init()
      let complete = false
      const context = {
        sessionID: chat.id,
        messageID: assistant.id,
        agent: "orchestra",
        abort: new AbortController().signal,
        extra: {
          promptOps: {
            ...stubOps(),
            prompt: (input: SessionPrompt.PromptInput) =>
              Effect.succeed(
                reply(
                  input,
                  complete
                    ? "STATUS: complete\nCHANGES: fixed blocker\nVALIDATION: passed\nRISKS: none\nNEXT_ACTION: review"
                    : "STATUS: blocked\nCHANGES: retained valid work\nVALIDATION: blocker reproduced\nRISKS: unresolved blocker\nNEXT_ACTION: retry differently",
                ),
              ),
          } satisfies TaskPromptOps,
        },
        messages: [],
        metadata: () => Effect.void,
        ask: () => Effect.void,
      }

      for (let attempt = 1; attempt <= 5; attempt++) {
        const result = yield* def.execute(
          {
            description: `blocked attempt ${attempt}`,
            prompt: "address the concrete blocker with a different strategy",
            subagent_type: "orchestra-implementer",
          },
          context,
        )
        expect(result.output).toContain(`ORCHESTRA_RECOVERY: blocked attempt ${attempt} of 5`)
        if (attempt === 5) expect(result.output).toContain("Terminal for this chat turn")
      }

      const clamped = yield* def.execute(
        {
          description: "blocked attempt six",
          prompt: "this should remain terminal",
          subagent_type: "orchestra-implementer",
        },
        context,
      )
      expect(clamped.output).toContain("blocked attempt 5 of 5")

      complete = true
      const success = yield* def.execute(
        {
          description: "complete recovery",
          prompt: "complete the recovered implementation",
          subagent_type: "orchestra-implementer",
        },
        context,
      )
      expect(success.output).not.toContain("ORCHESTRA_RECOVERY")

      complete = false
      const reset = yield* def.execute(
        {
          description: "new blocked phase",
          prompt: "start the next phase",
          subagent_type: "orchestra-implementer",
        },
        context,
      )
      expect(reset.output).toContain("blocked attempt 1 of 5")
    }),
    {
      config: {
        agent: {
          orchestra: { mode: "primary" },
          "orchestra-implementer": { mode: "subagent" },
        },
      },
    },
  )

  it.instance("rejects model overrides from the Research Lead", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seedAgent("research")
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const exit = yield* def
        .execute(
          {
            description: "override specialist",
            prompt: "search the assigned literature lane",
            subagent_type: "research-scout",
            model: "test/other-model",
          },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "research",
            abort: new AbortController().signal,
            extra: { promptOps: stubOps() },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )
        .pipe(Effect.exit)

      expect(Exit.isFailure(exit)).toBe(true)
      if (Exit.isFailure(exit)) expect(String(Cause.squash(exit.cause))).toContain("Research Lead cannot override")
    }),
    {
      config: {
        agent: {
          research: { mode: "primary" },
          "research-scout": { mode: "subagent" },
        },
      },
    },
  )

  it.instance("scopes blocked implementer attempts to one parent chat turn", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seedAgent("orchestra")
      const sessions = yield* Session.Service
      const tool = yield* TaskTool
      const def = yield* tool.init()
      let output = "STATUS: blocked\nCHANGES: retained work\nVALIDATION: blocked\nRISKS: unresolved\nNEXT_ACTION: retry"
      const execute = (messageID: MessageID) =>
        def.execute(
          {
            description: "implementation attempt",
            prompt: "perform the implementation work",
            subagent_type: "orchestra-implementer",
          },
          {
            sessionID: chat.id,
            messageID,
            agent: "orchestra",
            abort: new AbortController().signal,
            extra: {
              promptOps: {
                ...stubOps(),
                prompt: (input: SessionPrompt.PromptInput) => Effect.succeed(reply(input, output)),
              } satisfies TaskPromptOps,
            },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )

      for (let attempt = 1; attempt <= 5; attempt++) {
        expect((yield* execute(assistant.id)).output).toContain(`blocked attempt ${attempt} of 5`)
      }

      const nextTurn = MessageID.ascending()
      yield* sessions.updateMessage({ ...assistant, id: nextTurn, time: { created: Date.now() + 1 } })
      expect((yield* execute(nextTurn)).output).toContain("blocked attempt 1 of 5")

      output = "STATUS: complete\nCHANGES: fixed old turn\nVALIDATION: passed\nRISKS: none\nNEXT_ACTION: review"
      expect((yield* execute(assistant.id)).output).not.toContain("ORCHESTRA_RECOVERY")

      output = "STATUS: blocked\nCHANGES: retained work\nVALIDATION: blocked\nRISKS: unresolved\nNEXT_ACTION: retry"
      expect((yield* execute(nextTurn)).output).toContain("blocked attempt 2 of 5")
    }),
    {
      config: {
        agent: {
          orchestra: { mode: "primary" },
          "orchestra-implementer": { mode: "subagent" },
        },
      },
    },
  )

  it.instance("resets the blocked implementer streak on reviewer phase transition", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seedAgent("orchestra")
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const execute = (role: "orchestra-implementer" | "orchestra-reviewer", output: string) =>
        def.execute(
          { description: "phase task", prompt: "perform phase work", subagent_type: role },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "orchestra",
            abort: new AbortController().signal,
            extra: { promptOps: stubOps({ text: output }) },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )

      const blocked = "STATUS: blocked\nCHANGES: retained work\nVALIDATION: blocked\nRISKS: unresolved\nNEXT_ACTION: retry"
      expect((yield* execute("orchestra-implementer", blocked)).output).toContain("blocked attempt 1 of 5")
      expect((yield* execute("orchestra-implementer", blocked)).output).toContain("blocked attempt 2 of 5")
      const reviewer = yield* execute(
        "orchestra-reviewer",
        "STATUS: approved\nFINDINGS: none\nVALIDATION: passed\nRISKS: none\nNEXT_ACTION: test",
      )
      expect(reviewer.output).not.toContain("ORCHESTRA_RECOVERY")
      expect((yield* execute("orchestra-implementer", blocked)).output).toContain("blocked attempt 1 of 5")
    }),
    {
      config: {
        agent: {
          orchestra: { mode: "primary" },
          "orchestra-implementer": { mode: "subagent" },
          "orchestra-reviewer": { mode: "subagent" },
        },
      },
    },
  )

  it.instance("does not reset blocked implementer recovery after assistant advice", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seedAgent("orchestra")
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const execute = (role: "orchestra-implementer" | "orchestra-assistant-specialist", output: string) =>
        def.execute(
          { description: "phase task", prompt: "perform phase work", subagent_type: role },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "orchestra",
            abort: new AbortController().signal,
            extra: { promptOps: stubOps({ text: output }) },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )

      const blocked = "STATUS: blocked\nCHANGES: retained work\nVALIDATION: blocked\nRISKS: unresolved\nNEXT_ACTION: retry"
      expect((yield* execute("orchestra-implementer", blocked)).output).toContain("blocked attempt 1 of 5")
      expect((yield* execute("orchestra-assistant-specialist", "ADVISORY: inspect another strategy")).output).not.toContain(
        "ORCHESTRA_RECOVERY",
      )
      expect((yield* execute("orchestra-implementer", blocked)).output).toContain("blocked attempt 2 of 5")
    }),
    {
      config: {
        agent: {
          orchestra: { mode: "primary" },
          "orchestra-implementer": { mode: "subagent" },
          "orchestra-assistant-specialist": { mode: "subagent" },
        },
      },
    },
  )

  it.instance("repairs an empty orchestra handoff with one structured turn", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seedAgent("orchestra")
      const sessions = yield* Session.Service
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const prompts: SessionPrompt.PromptInput[] = []

      const result = yield* def.execute(
        {
          description: "implement scoped change",
          prompt: "implement the requested change",
          subagent_type: "orchestra-implementer",
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "orchestra",
          abort: new AbortController().signal,
          extra: {
            promptOps: {
              ...stubOps(),
              prompt: (input) => {
                prompts.push(input)
                if (prompts.length === 1) return Effect.succeed(reply(input, ""))
                return Effect.succeed(
                  reply(
                    input,
                    "STATUS: complete\nCHANGES: updated migration\nVALIDATION: db:migrate passed\nRISKS: none\nNEXT_ACTION: review",
                  ),
                )
              },
            } satisfies TaskPromptOps,
          },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      expect(prompts).toHaveLength(2)
      expect(prompts[1]?.sessionID).toBe(prompts[0]?.sessionID)
      expect(prompts[1]?.tools).toEqual({ "*": false })
      expect(prompts[1]?.format).toBeUndefined()
      expect(prompts[1]?.parts[0]).toMatchObject({
        type: "text",
        text: expect.stringContaining("Original task:\nimplement the requested change"),
      })
      expect(result.output).toContain("STATUS: complete")
      expect(result.output).toContain("CHANGES: updated migration")
      expect((yield* sessions.get(prompts[1]!.sessionID)).time.archived).toBeUndefined()
      expect((yield* sessions.get(result.metadata.sessionId)).permission).not.toContainEqual({
        permission: "*",
        pattern: "*",
        action: "deny",
      })
    }),
    {
      config: {
        agent: {
          orchestra: { mode: "primary" },
          "orchestra-implementer": { mode: "subagent" },
        },
      },
    },
  )

  it.instance("accepts a complete human-readable implementer handoff without repair", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seedAgent("orchestra")
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const prompts: SessionPrompt.PromptInput[] = []
      const output = [
        "STATUS: complete",
        "SUMMARY: Closed the fixture drift finding.",
        "FILES_CHANGED:",
        "- tools/regenerate-binding-fixtures.mjs",
        "VALIDATION:",
        "- targeted checks passed",
        "RISKS:",
        "- none",
        "NEXT_ACTION:",
        "- review",
      ].join("\n")

      const result = yield* def.execute(
        {
          description: "implement fixture hardening",
          prompt: "implement the requested fixture hardening",
          subagent_type: "orchestra-implementer",
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "orchestra",
          abort: new AbortController().signal,
          extra: {
            promptOps: {
              ...stubOps(),
              prompt: (input) => {
                prompts.push(input)
                return Effect.succeed(reply(input, output))
              },
            } satisfies TaskPromptOps,
          },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      expect(prompts).toHaveLength(1)
      expect(result.output).toContain("STATUS: complete")
      expect(result.output).toContain("FILES_CHANGED:")
      expect(result.output).not.toContain("invalid_handoff")
    }),
    {
      config: {
        agent: {
          orchestra: { mode: "primary" },
          "orchestra-implementer": { mode: "subagent" },
        },
      },
    },
  )

  it.instance("accepts tests-run implementer reports without repair", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seedAgent("orchestra")
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const prompts: SessionPrompt.PromptInput[] = []
      const output = [
        "STATUS: complete",
        "SUMMARY: Completed materialization hardening.",
        "FILES_CHANGED:",
        "- internal/domain/curriculum/materialization/service.go",
        "DESIGN_INVARIANTS:",
        "- Unsupported content fails closed.",
        "TESTS_RUN:",
        "- go test ./... -count=1 - passed",
        "- pnpm db:vet - passed",
        "RISKS: none identified",
        "NEXT_ACTION: Lead should review the scoped diff.",
      ].join("\n")

      const result = yield* def.execute(
        {
          description: "implement materialization",
          prompt: "implement materialization hardening",
          subagent_type: "orchestra-implementer",
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "orchestra",
          abort: new AbortController().signal,
          extra: {
            promptOps: {
              ...stubOps(),
              prompt: (input) => {
                prompts.push(input)
                return Effect.succeed(reply(input, output))
              },
            } satisfies TaskPromptOps,
          },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      expect(prompts).toHaveLength(1)
      expect(result.output).toContain("TESTS_RUN:")
      expect(result.output).not.toContain("invalid_handoff")
    }),
    {
      config: {
        agent: {
          orchestra: { mode: "primary" },
          "orchestra-implementer": { mode: "subagent" },
        },
      },
    },
  )

  it.instance("accepts structurally complete novel handoff headings without repair", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seedAgent("orchestra")
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const prompts: SessionPrompt.PromptInput[] = []
      const output = [
        "STATUS: complete",
        "DELIVERED_WORK: Implemented the requested scoped behavior and preserved unrelated worktree changes.",
        "PROOF_OF_CHECKS: Ran targeted tests, typecheck, and diff validation successfully.",
        "UNRESOLVED_ITEMS: No blocking concern remains after validation of the scoped implementation.",
        "ROUTING_DECISION: Lead should inspect the scoped diff and continue to independent review.",
      ].join("\n")

      const result = yield* def.execute(
        {
          description: "implement scoped behavior",
          prompt: "implement the requested scoped behavior",
          subagent_type: "orchestra-implementer",
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "orchestra",
          abort: new AbortController().signal,
          extra: {
            promptOps: {
              ...stubOps(),
              prompt: (input) => {
                prompts.push(input)
                return Effect.succeed(reply(input, output))
              },
            } satisfies TaskPromptOps,
          },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      expect(prompts).toHaveLength(1)
      expect(result.output).toContain("DELIVERED_WORK:")
      expect(result.output).not.toContain("invalid_handoff")
    }),
    {
      config: {
        agent: {
          orchestra: { mode: "primary" },
          "orchestra-implementer": { mode: "subagent" },
        },
      },
    },
  )

  it.instance("passes a substantive reviewer finding through without formatting repair", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seedAgent("orchestra")
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const prompts: SessionPrompt.PromptInput[] = []
      const output = [
        "STATUS: needs_revision",
        "SUMMARY: The prior root findings are addressed, but one concurrency race remains.",
        "FINDINGS:",
        "- MEDIUM — A duplicate begin racing completion can return unavailable instead of the committed replay.",
        "Evidence: completion may insert the result and delete the claim while begin waits on the claim lock.",
        "After a missing claim, begin should recheck the committed result and cover the interleaving with a deterministic test.",
      ].join("\n")

      const result = yield* def.execute(
        {
          description: "review concurrency fix",
          prompt: "review the concurrency fix",
          subagent_type: "orchestra-reviewer",
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "orchestra",
          abort: new AbortController().signal,
          extra: {
            promptOps: {
              ...stubOps(),
              prompt: (input) => {
                prompts.push(input)
                return Effect.succeed(reply(input, output))
              },
            } satisfies TaskPromptOps,
          },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      expect(prompts).toHaveLength(1)
      expect(result.output).toContain("STATUS: needs_revision")
      expect(result.output).not.toContain("invalid_handoff")
    }),
    {
      config: {
        agent: {
          orchestra: { mode: "primary" },
          "orchestra-reviewer": { mode: "subagent" },
        },
      },
    },
  )

  it.instance("repairs reviewer and tester handoffs with their role-specific fields", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seedAgent("orchestra")
      const sessions = yield* Session.Service
      const tool = yield* TaskTool
      const def = yield* tool.init()

      for (const testCase of [
        {
          role: "orchestra-reviewer",
          output: "STATUS: approved\nFINDINGS: none\nVALIDATION: targeted tests passed\nRISKS: none\nNEXT_ACTION: test",
          field: "FINDINGS: none",
        },
        {
          role: "orchestra-tester",
          output: "STATUS: passed\nCHANGES: none\nVALIDATION: suite passed\nFAILURES: none\nNEXT_ACTION: complete",
          field: "FAILURES: none",
        },
      ]) {
        const prompts: SessionPrompt.PromptInput[] = []
        const result = yield* def.execute(
          {
            description: "finish specialist handoff",
            prompt: "perform the scoped work",
            subagent_type: testCase.role,
          },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "orchestra",
            abort: new AbortController().signal,
            extra: {
              promptOps: {
                ...stubOps(),
                prompt: (input) => {
                  prompts.push(input)
                  if (prompts.length === 1) return Effect.succeed(reply(input, ""))
                  return Effect.succeed(reply(input, testCase.output))
                },
              } satisfies TaskPromptOps,
            },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )

        expect(prompts).toHaveLength(2)
        expect(prompts[1]?.sessionID).toBe(prompts[0]?.sessionID)
        expect(prompts[1]?.agent).toBe(testCase.role)
        expect(prompts[1]?.model).toEqual(prompts[0]?.model)
        expect((yield* sessions.get(prompts[1]!.sessionID)).time.archived).toBeUndefined()
        expect(result.metadata.sessionId).toBe(prompts[0]?.sessionID)
        expect(result.output).toContain(testCase.field)
      }
    }),
    {
      config: {
        agent: {
          orchestra: { mode: "primary" },
          "orchestra-reviewer": { mode: "subagent" },
          "orchestra-tester": { mode: "subagent" },
        },
      },
    },
  )

  it.instance("returns blocked when native handoff repair also fails", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seedAgent("orchestra")
      const tool = yield* TaskTool
      const def = yield* tool.init()
      let prompts = 0

      const result = yield* def.execute(
        {
          description: "review scoped change",
          prompt: "review the requested change",
          subagent_type: "orchestra-reviewer",
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "orchestra",
          abort: new AbortController().signal,
          extra: {
            promptOps: {
              ...stubOps(),
              prompt: (input) => {
                prompts++
                if (prompts === 1) return Effect.succeed(reply(input, ""))
                return Effect.succeed(failedReply(input, "structured output unavailable"))
              },
            } satisfies TaskPromptOps,
          },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      expect(prompts).toBe(2)
      expect(result.output).toContain('state="completed"')
      expect(result.output).toContain("STATUS: blocked")
      expect(result.output).toContain("REASON: invalid_handoff")
      expect(result.output).toContain("structured output unavailable")
    }),
    {
      config: {
        agent: {
          orchestra: { mode: "primary" },
          "orchestra-reviewer": { mode: "subagent" },
        },
      },
    },
  )

  it.instance("returns blocked when native handoff repair throws", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seedAgent("orchestra")
      const tool = yield* TaskTool
      const def = yield* tool.init()
      let prompts = 0

      const result = yield* def.execute(
        {
          description: "test scoped change",
          prompt: "test the requested change",
          subagent_type: "orchestra-tester",
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "orchestra",
          abort: new AbortController().signal,
          extra: {
            promptOps: {
              ...stubOps(),
              prompt: (input) => {
                prompts++
                if (prompts === 1) return Effect.succeed(reply(input, ""))
                return Effect.die(new Error("provider unavailable"))
              },
            } satisfies TaskPromptOps,
          },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      expect(prompts).toBe(2)
      expect(result.output).toContain("STATUS: blocked")
      expect(result.output).toContain("provider unavailable")
    }),
    {
      config: {
        agent: {
          orchestra: { mode: "primary" },
          "orchestra-tester": { mode: "subagent" },
        },
      },
    },
  )

  it.instance("rejects generic delegation from the orchestra Lead", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seedAgent("orchestra")
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const exit = yield* def
        .execute(
          {
            description: "invalid delegation",
            prompt: "do work",
            subagent_type: "general",
          },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "orchestra",
            abort: new AbortController().signal,
            extra: { promptOps: stubOps() },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )
        .pipe(Effect.exit)

      expect(Exit.isFailure(exit)).toBe(true)
    }),
  )

  it.instance("rejects private Orchestra specialists from native Build", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seedAgent("build")
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const exit = yield* def
        .execute(
          {
            description: "invalid private delegation",
            prompt: "implement work",
            subagent_type: "orchestra-implementer",
          },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "build",
            abort: new AbortController().signal,
            extra: { promptOps: stubOps() },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )
        .pipe(Effect.exit)

      expect(Exit.isFailure(exit)).toBe(true)
      if (Exit.isFailure(exit)) expect(String(Cause.squash(exit.cause))).toContain("private Orchestra specialist")
    }),
    {
      config: {
        agent: {
          "orchestra-implementer": { mode: "subagent" },
        },
      },
    },
  )

  it.instance(
    "shows Orchestra specialists only to the Orchestra Lead",
    () =>
      Effect.gen(function* () {
        const agent = yield* Agent.Service
        const registry = yield* ToolRegistry.Service
        const build = yield* agent.get("build")
        const orchestra = yield* agent.get("orchestra")
        const describe = Effect.fnUntraced(function* (caller: Agent.Info) {
          return (yield* registry.tools({ ...ref, agent: caller })).find((tool) => tool.id === TaskTool.id)?.description ?? ""
        })

        const buildDescription = yield* describe(build)
        const orchestraDescription = yield* describe(orchestra)
        expect(buildDescription).not.toContain("orchestra-assistant-specialist")
        expect(buildDescription).not.toContain("orchestra-implementer")
        expect(buildDescription).not.toContain("orchestra-reviewer")
        expect(buildDescription).not.toContain("orchestra-tester")
        expect(orchestraDescription).toContain("orchestra-implementer")
        expect(orchestraDescription).toContain("orchestra-assistant-specialist")
        expect(orchestraDescription).toContain("orchestra-reviewer")
        expect(orchestraDescription).toContain("orchestra-tester")
        expect(orchestraDescription).not.toContain("- general:")
        expect(orchestraDescription).not.toContain("- explore:")
      }),
    {
      config: {
        agent: {
          orchestra: { mode: "primary" },
          "orchestra-assistant-specialist": { mode: "subagent" },
          "orchestra-implementer": { mode: "subagent" },
          "orchestra-reviewer": { mode: "subagent" },
          "orchestra-tester": { mode: "subagent" },
        },
      },
    },
  )

  it.instance(
    "shows Research specialists only to the Research Lead",
    () =>
      Effect.gen(function* () {
        const agent = yield* Agent.Service
        const registry = yield* ToolRegistry.Service
        const build = yield* agent.get("build")
        const orchestra = yield* agent.get("orchestra")
        const research = yield* agent.get("research")
        const describe = Effect.fnUntraced(function* (caller: Agent.Info) {
          return (yield* registry.tools({ ...ref, agent: caller })).find((tool) => tool.id === TaskTool.id)?.description ?? ""
        })

        const buildDescription = yield* describe(build)
        const orchestraDescription = yield* describe(orchestra)
        const researchDescription = yield* describe(research)
        for (const role of [
          "research-methodologist",
          "research-scout",
          "research-analyst",
          "research-critic",
          "research-writer",
          "research-reviewer",
          "research-editor",
        ]) {
          expect(buildDescription).not.toContain(role)
          expect(orchestraDescription).not.toContain(role)
          expect(researchDescription).toContain(role)
        }
        expect(researchDescription).not.toContain("orchestra-implementer")
        expect(researchDescription).not.toContain("- general:")
        expect(researchDescription).not.toContain("- explore:")
      }),
    {
      config: {
        agent: {
          orchestra: { mode: "primary" },
          research: { mode: "primary" },
          "research-methodologist": { mode: "subagent" },
          "research-scout": { mode: "subagent" },
          "research-analyst": { mode: "subagent" },
          "research-critic": { mode: "subagent" },
          "research-writer": { mode: "subagent" },
          "research-reviewer": { mode: "subagent" },
          "research-editor": { mode: "subagent" },
          "orchestra-implementer": { mode: "subagent" },
        },
      },
    },
  )

  it.instance(
    "keeps reserved Orchestra specialists hidden subagents after config merge",
    () =>
      Effect.gen(function* () {
        const agent = yield* Agent.Service
        const specialist = yield* agent.get("orchestra-implementer")
        const assistant = yield* agent.get("orchestra-assistant-specialist")
        expect(specialist.mode).toBe("subagent")
        expect(specialist.hidden).toBe(true)
        expect(assistant.mode).toBe("subagent")
        expect(assistant.hidden).toBe(true)
      }),
    {
      config: {
        agent: {
          "orchestra-implementer": { mode: "primary", hidden: false },
          "orchestra-assistant-specialist": { mode: "primary", hidden: false },
        },
      },
    },
  )

  it.instance(
    "keeps reserved Research specialists hidden subagents after config merge",
    () =>
      Effect.gen(function* () {
        const agent = yield* Agent.Service
        const scout = yield* agent.get("research-scout")
        const writer = yield* agent.get("research-writer")
        expect(scout.mode).toBe("subagent")
        expect(scout.hidden).toBe(true)
        expect(writer.mode).toBe("subagent")
        expect(writer.hidden).toBe(true)
      }),
    {
      config: {
        agent: {
          "research-scout": { mode: "primary", hidden: false },
          "research-writer": { mode: "primary", hidden: false },
        },
      },
    },
  )

  it.instance(
    "appends the five-attempt recovery policy to the Orchestra Lead prompt",
    () =>
      Effect.gen(function* () {
        const agent = yield* Agent.Service
        const orchestra = yield* agent.get("orchestra")
        expect(orchestra.prompt).toContain("at most five consecutive implementer attempts within one user chat turn")
        expect(orchestra.prompt).toContain("new user message in the same session starts a fresh attempt budget")
        expect(orchestra.prompt).toContain("ORCHESTRA_RECOVERY")
        expect(orchestra.prompt).toContain("supersedes any earlier instruction")
      }),
    {
      config: {
        agent: {
          orchestra: { mode: "primary", prompt: "Legacy instruction: stop after one fresh attempt." },
        },
      },
    },
  )

  it.instance(
    "description sorts subagents by name and is stable across calls",
    () =>
      Effect.gen(function* () {
        const agent = yield* Agent.Service
        const build = yield* agent.get("build")
        const registry = yield* ToolRegistry.Service
        const get = Effect.fnUntraced(function* () {
          const tools = yield* registry.tools({ ...ref, agent: build })
          return tools.find((tool) => tool.id === TaskTool.id)?.description ?? ""
        })
        const first = yield* get()
        const second = yield* get()

        expect(first).toBe(second)

        const alpha = first.indexOf("- alpha: Alpha agent")
        const explore = first.indexOf("- explore:")
        const general = first.indexOf("- general:")
        const zebra = first.indexOf("- zebra: Zebra agent")

        expect(alpha).toBeGreaterThan(-1)
        expect(explore).toBeGreaterThan(alpha)
        expect(general).toBeGreaterThan(explore)
        expect(zebra).toBeGreaterThan(general)
      }),
    {
      config: {
        agent: {
          zebra: {
            description: "Zebra agent",
            mode: "subagent",
          },
          alpha: {
            description: "Alpha agent",
            mode: "subagent",
          },
        },
      },
    },
  )

  it.instance(
    "description hides denied subagents for the caller",
    () =>
      Effect.gen(function* () {
        const agent = yield* Agent.Service
        const build = yield* agent.get("build")
        const registry = yield* ToolRegistry.Service
        const description =
          (yield* registry.tools({ ...ref, agent: build })).find((tool) => tool.id === TaskTool.id)?.description ?? ""

        expect(description).toContain("- alpha: Alpha agent")
        expect(description).not.toContain("- zebra: Zebra agent")
      }),
    {
      config: {
        permission: {
          task: {
            "*": "allow",
            zebra: "deny",
          },
        },
        agent: {
          zebra: {
            description: "Zebra agent",
            mode: "subagent",
          },
          alpha: {
            description: "Alpha agent",
            mode: "subagent",
          },
        },
      },
    },
  )

  it.instance("execute resumes an existing task session from task_id", () =>
    Effect.gen(function* () {
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const child = yield* sessions.create({
        parentID: chat.id,
        title: "Existing child",
        agent: "general",
        internalMetadata: Delegation.metadata({ kind: "delegated-task", parentID: chat.id, agent: "general" }),
      })
      const tool = yield* TaskTool
      const def = yield* tool.init()
      let seen: SessionPrompt.PromptInput | undefined
      const promptOps = stubOps({ text: "resumed", onPrompt: (input) => (seen = input) })

      const result = yield* def.execute(
        {
          description: "inspect bug",
          prompt: "look into the cache key path",
          subagent_type: "general",
          task_id: child.id,
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "build",
          abort: new AbortController().signal,
          extra: { promptOps },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      const kids = yield* sessions.children(chat.id)
      expect(kids).toHaveLength(1)
      expect(kids[0]?.id).toBe(child.id)
      expect(result.metadata.sessionId).toBe(child.id)
      expect(result.output).toContain(`<task id="${child.id}" state="completed">`)
      expect(seen?.sessionID).toBe(child.id)
      expect(seen?.variant).toBe("xhigh")
    }),
  )

  it.instance("execute surfaces child errors with a resumable task_id", () =>
    Effect.gen(function* () {
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()

      const exit = yield* def
        .execute(
          {
            description: "inspect bug",
            prompt: "look into the cache key path",
            subagent_type: "general",
          },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "build",
            abort: new AbortController().signal,
            extra: {
              promptOps: stubOps({
                text: "",
                error: new SessionV1.APIError({ message: "Network connection lost", isRetryable: false }).toObject(),
              }),
            },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )
        .pipe(Effect.exit)

      expect(Exit.isFailure(exit)).toBe(true)
      if (Exit.isSuccess(exit)) throw new Error("expected task failure")
      const child = (yield* sessions.children(chat.id))[0]
      expect(child).toBeDefined()
      const failure = Cause.squash(exit.cause)
      expect(failure).toBeInstanceOf(Error)
      if (!(failure instanceof Error)) throw new Error("expected Error defect")
      expect(failure.message).toBe(`Subagent failed (task_id: ${child?.id}): Network connection lost`)
    }),
  )

  it.instance("execute surfaces terminal child tool errors with a resumable task_id", () =>
    Effect.gen(function* () {
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()

      const exit = yield* def
        .execute(
          {
            description: "inspect external directory",
            prompt: "read the external directory",
            subagent_type: "general",
          },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "build",
            abort: new AbortController().signal,
            extra: {
              promptOps: stubOps({
                text: "I will inspect the directory.",
                toolError: "The user rejected permission to use this specific tool call.",
              }),
            },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )
        .pipe(Effect.exit)

      expect(Exit.isFailure(exit)).toBe(true)
      if (Exit.isSuccess(exit)) throw new Error("expected task failure")
      const child = (yield* sessions.children(chat.id))[0]
      const failure = Cause.squash(exit.cause)
      expect(failure).toBeInstanceOf(Error)
      if (!(failure instanceof Error)) throw new Error("expected Error defect")
      expect(failure.message).toBe(
        `Subagent failed (task_id: ${child?.id}): The user rejected permission to use this specific tool call.`,
      )
    }),
  )

  it.instance("execute asks by default and skips checks when bypassed", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const calls: unknown[] = []
      const promptOps = stubOps()

      const exec = (extra?: Record<string, any>) =>
        def.execute(
          {
            description: "inspect bug",
            prompt: "look into the cache key path",
            subagent_type: "general",
          },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "build",
            abort: new AbortController().signal,
            extra: { promptOps, ...extra },
            messages: [],
            metadata: () => Effect.void,
            ask: (input) =>
              Effect.sync(() => {
                calls.push(input)
              }),
          },
        )

      yield* exec()
      yield* exec({ bypassAgentCheck: true })

      expect(calls).toHaveLength(1)
      expect(calls[0]).toEqual({
        permission: "task",
        patterns: ["general"],
        always: ["*"],
        metadata: {
          description: "inspect bug",
          subagent_type: "general",
        },
      })
    }),
  )

  it.instance("execute cancels child session when abort signal fires", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const ready = defer<SessionPrompt.PromptInput>()
      const cancelled = defer<SessionID>()
      const abort = new AbortController()
      const promptOps: TaskPromptOps = {
        cancel: (sessionID) =>
          Effect.sync(() => {
            cancelled.resolve(sessionID)
          }),
        resolvePromptParts: (template) => Effect.succeed([{ type: "text" as const, text: template }]),
        prompt: (input) =>
          Effect.promise(() => {
            ready.resolve(input)
            return cancelled.promise
          }).pipe(Effect.as(reply(input, "cancelled"))),
      }

      const fiber = yield* def
        .execute(
          {
            description: "inspect bug",
            prompt: "look into the cache key path",
            subagent_type: "general",
          },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "build",
            abort: abort.signal,
            extra: { promptOps },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )
        .pipe(Effect.forkChild)

      const input = yield* Effect.promise(() => ready.promise)
      abort.abort()
      expect(yield* Effect.promise(() => cancelled.promise)).toBe(input.sessionID)

      const exit = yield* Fiber.await(fiber)
      expect(Exit.isSuccess(exit)).toBe(true)
    }),
  )

  it.instance("execute creates a child when task_id does not exist", () =>
    Effect.gen(function* () {
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()
      let seen: SessionPrompt.PromptInput | undefined
      const promptOps = stubOps({ text: "created", onPrompt: (input) => (seen = input) })

      const result = yield* def.execute(
        {
          description: "inspect bug",
          prompt: "look into the cache key path",
          subagent_type: "general",
          task_id: "ses_missing",
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "build",
          abort: new AbortController().signal,
          extra: { promptOps },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      const kids = yield* sessions.children(chat.id)
      expect(kids).toHaveLength(1)
      expect(kids[0]?.id).toBe(result.metadata.sessionId)
      expect(result.metadata.sessionId).not.toBe("ses_missing")
      expect(result.output).toContain(`<task id="${result.metadata.sessionId}" state="completed">`)
      expect(seen?.sessionID).toBe(result.metadata.sessionId)
    }),
  )

  it.instance("prevents subagents from launching subagents by default", () =>
    Effect.gen(function* () {
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const child = yield* sessions.create({ parentID: chat.id, title: "child" })
      const nestedAssistant = yield* sessions.updateMessage({
        ...assistant,
        id: MessageID.ascending(),
        parentID: MessageID.ascending(),
        sessionID: child.id,
      })
      const tool = yield* TaskTool
      const def = yield* tool.init()
      let asked = false

      const exit = yield* def
        .execute(
          {
            description: "inspect bug",
            prompt: "look into the cache key path",
            subagent_type: "general",
          },
          {
            sessionID: child.id,
            messageID: nestedAssistant.id,
            agent: "general",
            abort: new AbortController().signal,
            extra: { promptOps: stubOps() },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.sync(() => (asked = true)),
          },
        )
        .pipe(Effect.exit)

      expect(Exit.isFailure(exit)).toBe(true)
      expect(asked).toBe(false)
      expect(yield* sessions.children(child.id)).toHaveLength(0)
    }),
  )

  it.instance(
    "allows nested subagents up to the configured depth",
    () =>
      Effect.gen(function* () {
        const sessions = yield* Session.Service
        const { chat, assistant } = yield* seed()
        const child = yield* sessions.create({ parentID: chat.id, title: "child" })
        const nestedAssistant = yield* sessions.updateMessage({
          ...assistant,
          id: MessageID.ascending(),
          parentID: MessageID.ascending(),
          sessionID: child.id,
        })
        const tool = yield* TaskTool
        const def = yield* tool.init()

        const result = yield* def.execute(
          {
            description: "inspect bug",
            prompt: "look into the cache key path",
            subagent_type: "general",
          },
          {
            sessionID: child.id,
            messageID: nestedAssistant.id,
            agent: "general",
            abort: new AbortController().signal,
            extra: { promptOps: stubOps() },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )

        expect((yield* sessions.get(result.metadata.sessionId)).parentID).toBe(child.id)
      }),
    { config: { subagent_depth: 2 } },
  )

  it.instance(
    "execute shapes child permissions for task, todowrite, and primary tools",
    () =>
      Effect.gen(function* () {
        const sessions = yield* Session.Service
        const { chat, assistant } = yield* seed()
        const tool = yield* TaskTool
        const def = yield* tool.init()
        let seen: SessionPrompt.PromptInput | undefined
        const promptOps = stubOps({ onPrompt: (input) => (seen = input) })

        const result = yield* def.execute(
          {
            description: "inspect bug",
            prompt: "look into the cache key path",
            subagent_type: "reviewer",
          },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "build",
            abort: new AbortController().signal,
            extra: { promptOps },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )

        const child = yield* sessions.get(result.metadata.sessionId)
        expect(child.parentID).toBe(chat.id)
        expect(child.agent).toBe("reviewer")
        expect(child.permission).toEqual([
          {
            permission: "todowrite",
            pattern: "*",
            action: "deny",
          },
          {
            permission: "bash",
            pattern: "*",
            action: "deny",
          },
          {
            permission: "read",
            pattern: "*",
            action: "deny",
          },
        ])
        expect(seen?.tools).toBeUndefined()
      }),
    {
      config: {
        agent: {
          reviewer: {
            mode: "subagent",
            permission: {
              task: "allow",
            },
          },
        },
        experimental: {
          primary_tools: ["bash", "read"],
        },
      },
    },
  )

  it.instance("rejects background execution when the experiment is disabled", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()

      const exit = yield* def
        .execute(
          {
            description: "inspect bug",
            prompt: "look into the cache key path",
            subagent_type: "general",
            background: true,
          },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "build",
            abort: new AbortController().signal,
            extra: { promptOps: stubOps() },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )
        .pipe(Effect.exit)

      expect(Exit.isFailure(exit)).toBe(true)
    }),
  )

  it.instance("promotes a running foreground task without restarting it", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const ready = yield* Deferred.make<void>()
      const done = yield* Deferred.make<void>()
      const injected = yield* Deferred.make<SessionPrompt.PromptInput>()
      let runs = 0
      const promptOps: TaskPromptOps = {
        cancel: () => Effect.void,
        resolvePromptParts: (template) => Effect.succeed([{ type: "text" as const, text: template }]),
        prompt: (input) => {
          if (input.sessionID === chat.id) {
            return Deferred.succeed(injected, input).pipe(Effect.as(reply(input, "injected")))
          }
          return Effect.gen(function* () {
            runs += 1
            yield* Deferred.succeed(ready, undefined)
            yield* Deferred.await(done)
            return reply(input, "background done")
          })
        },
      }

      const fiber = yield* def
        .execute(
          {
            description: "inspect bug",
            prompt: "look into the cache key path",
            subagent_type: "general",
          },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "build",
            abort: new AbortController().signal,
            extra: { promptOps },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )
        .pipe(Effect.forkChild)

      yield* Deferred.await(ready)
      const job = (yield* jobs.list())[0]
      expect(job).toBeDefined()
      if (!job) throw new Error("task job not found")
      expect(job.metadata?.parentSessionId).toBe(chat.id)
      yield* jobs.promote(job.id)

      const result = yield* Fiber.join(fiber)
      expect(result.metadata.background).toBe(true)
      expect(result.output).toContain(`state="running"`)
      expect((yield* jobs.get(result.metadata.sessionId))?.status).toBe("running")
      expect(runs).toBe(1)

      yield* Deferred.succeed(done, undefined)
      expect((yield* jobs.wait({ id: result.metadata.sessionId })).info?.output).toBe("background done")
      expect((yield* Deferred.await(injected)).parts[0]?.type).toBe("text")
      expect(runs).toBe(1)
    }),
  )

  background.instance("execute launches background tasks without waiting for completion", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()

      const result = yield* def.execute(
        {
          description: "inspect bug",
          prompt: "look into the cache key path",
          subagent_type: "general",
          background: true,
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "build",
          abort: new AbortController().signal,
          extra: {
            promptOps: {
              ...stubOps(),
              prompt: () => Effect.never,
            } satisfies TaskPromptOps,
          },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      const job = yield* jobs.get(result.metadata.sessionId)
      expect(result.metadata.background).toBe(true)
      expect(result.output).toContain(`state="running"`)
      expect(job?.status).toBe("running")
    }),
  )

  background.instance("background task completion waits for running updates", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const first = defer<void>()
      const second = defer<void>()
      const updated = defer<SessionPrompt.PromptInput>()
      const injected = defer<SessionPrompt.PromptInput>()
      let prompts = 0
      const promptOps: TaskPromptOps = {
        ...stubOps(),
        prompt: (input) => {
          if (input.sessionID === chat.id) {
            injected.resolve(input)
            return Effect.succeed(reply(input, "done"))
          }
          prompts++
          if (prompts === 1) return Effect.promise(() => first.promise).pipe(Effect.as(reply(input, "first done")))
          updated.resolve(input)
          return Effect.promise(() => second.promise).pipe(Effect.as(reply(input, "second done")))
        },
      }
      const context = {
        sessionID: chat.id,
        messageID: assistant.id,
        agent: "build",
        abort: new AbortController().signal,
        extra: { promptOps },
        messages: [],
        metadata: () => Effect.void,
        ask: () => Effect.void,
      }

      const started = yield* def.execute(
        {
          description: "inspect bug",
          prompt: "look into the cache key path",
          subagent_type: "general",
          background: true,
        },
        context,
      )
      const result = yield* def.execute(
        {
          description: "add investigation scope",
          prompt: "also inspect cancellation",
          subagent_type: "general",
          task_id: started.metadata.sessionId,
        },
        context,
      )

      expect(result.metadata.sessionId).toBe(started.metadata.sessionId)
      expect(result.metadata.background).toBe(true)
      expect(result.output).toContain("Background task updated")
      first.resolve()
      expect((yield* jobs.get(started.metadata.sessionId))?.status).toBe("running")
      expect((yield* Effect.promise(() => updated.promise)).parts).toEqual([
        { type: "text", text: "also inspect cancellation" },
      ])

      second.resolve()
      const waited = yield* jobs.wait({ id: started.metadata.sessionId, timeout: 1_000 })
      expect(waited.info?.status).toBe("completed")
      expect(waited.info?.output).toBe("second done")
      const notification = yield* Effect.promise(() => injected.promise)
      expect(notification.variant).toBe("xhigh")
      expect(notification.parts[0]?.type).toBe("text")
      if (notification.parts[0]?.type === "text") expect(notification.parts[0].text).toContain("second done")
    }),
  )

  it.instance("rejects task_id sessions without delegated-task provenance", () =>
    Effect.gen(function* () {
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const child = yield* sessions.create({ parentID: chat.id, title: "Unowned child", agent: "general" })
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const exit = yield* def
        .execute(
          {
            description: "inspect bug",
            prompt: "look into the cache key path",
            subagent_type: "general",
            task_id: child.id,
          },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "build",
            abort: new AbortController().signal,
            extra: { promptOps: stubOps() },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )
        .pipe(Effect.exit)

      expect(Exit.isFailure(exit)).toBe(true)
    }),
  )

  it.instance("rejects handoff repair sessions as task_id continuations", () =>
    Effect.gen(function* () {
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seedAgent("orchestra")
      const repair = yield* sessions.create({
        parentID: chat.id,
        title: "repair",
        agent: "orchestra-reviewer",
        internalMetadata: Delegation.metadata({ kind: "handoff-repair", parentID: chat.id, agent: "orchestra-reviewer" }),
      })
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const exit = yield* def
        .execute(
          {
            description: "review",
            prompt: "review",
            subagent_type: "orchestra-reviewer",
            task_id: repair.id,
          },
          {
            sessionID: chat.id,
            messageID: assistant.id,
            agent: "orchestra",
            abort: new AbortController().signal,
            extra: { promptOps: stubOps() },
            messages: [],
            metadata: () => Effect.void,
            ask: () => Effect.void,
          },
        )
        .pipe(Effect.exit)

      expect(Exit.isFailure(exit)).toBe(true)
    }),
    {
      config: { agent: { orchestra: { mode: "primary" }, "orchestra-reviewer": { mode: "subagent" } } },
    },
  )

  it.instance("protects delegated-task provenance from user metadata writes", () =>
    Effect.gen(function* () {
      const sessions = yield* Session.Service
      const parent = yield* sessions.create({ title: "parent" })
      const spoofed = yield* sessions.create({
        parentID: parent.id,
        agent: "orchestra-implementer",
        metadata: {
          user: "kept",
          "opencode.task.kind": "delegated-task",
          "opencode.task.parent": parent.id,
          "opencode.task.agent": "orchestra-implementer",
        },
      })
      expect(Delegation.isDelegatedTo(spoofed, "orchestra-implementer")).toBe(false)
      expect(spoofed.metadata).toEqual({ user: "kept" })

      const delegated = yield* sessions.create({
        parentID: parent.id,
        agent: "orchestra-implementer",
        internalMetadata: Delegation.metadata({
          kind: "delegated-task",
          parentID: parent.id,
          agent: "orchestra-implementer",
        }),
      })
      yield* sessions.setMetadata({
        sessionID: delegated.id,
        metadata: {
          user: "updated",
          "opencode.task.kind": "handoff-repair",
        },
      })
      const updated = yield* sessions.get(delegated.id)
      expect(Delegation.isDelegatedTo(updated, "orchestra-implementer")).toBe(true)
      expect(updated.metadata?.user).toBe("updated")
      expect(updated.metadata?.["opencode.task.kind"]).toBe("delegated-task")
    }),
  )

  background.instance("allows concurrent Orchestra tasks with the same role but rejects a running continuation", () =>
    Effect.gen(function* () {
      const { chat, assistant } = yield* seedAgent("orchestra")
      const tool = yield* TaskTool
      const def = yield* tool.init()
      const first = defer<void>()
      let prompts = 0
      const promptOps: TaskPromptOps = {
        ...stubOps(),
        prompt: (input) => {
          prompts++
          const handoff = reply(
            input,
            "STATUS: blocked\nCHANGES: none\nVALIDATION: blocked\nRISKS: none\nNEXT_ACTION: continue",
          )
          return Effect.promise(() => first.promise).pipe(Effect.as(handoff))
        },
      }
      const context = {
        sessionID: chat.id,
        messageID: assistant.id,
        agent: "orchestra",
        abort: new AbortController().signal,
        extra: { promptOps },
        messages: [],
        metadata: () => Effect.void,
        ask: () => Effect.void,
      }

      const started = yield* def.execute(
        {
          description: "implement change",
          prompt: "implement the requested change",
          subagent_type: "orchestra-implementer",
          background: true,
        },
        context,
      )
      const continuation = yield* def
        .execute(
          {
            description: "finish handoff",
            prompt: "stop exploring and return the handoff",
            subagent_type: "orchestra-implementer",
            task_id: started.metadata.sessionId,
          },
          context,
        )
        .pipe(Effect.exit)
      expect(Exit.isFailure(continuation)).toBe(true)

      const concurrent = yield* def.execute(
        {
          description: "parallel implementation",
          prompt: "implement another independent change",
          subagent_type: "orchestra-implementer",
          background: true,
        },
        context,
      )

      expect(concurrent.metadata.sessionId).not.toBe(started.metadata.sessionId)
      expect(concurrent.output).toContain('state="running"')
      first.resolve()
    }),
    {
      config: {
        agent: {
          orchestra: { mode: "primary" },
          "orchestra-implementer": { mode: "subagent" },
        },
      },
    },
  )

  background.instance("background tasks complete through the background job service", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()

      const result = yield* def.execute(
        {
          description: "inspect bug",
          prompt: "look into the cache key path",
          subagent_type: "general",
          background: true,
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "build",
          abort: new AbortController().signal,
          extra: { promptOps: stubOps({ text: "background done" }) },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      const waited = yield* jobs.wait({ id: result.metadata.sessionId, timeout: 1_000 })
      expect(waited.timedOut).toBe(false)
      expect(waited.info?.status).toBe("completed")
      expect(waited.info?.output).toBe("background done")
    }),
  )

  background.instance("background task completion does not wait for the parent async prompt", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()

      const result = yield* def.execute(
        {
          description: "inspect bug",
          prompt: "look into the cache key path",
          subagent_type: "general",
          background: true,
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "build",
          abort: new AbortController().signal,
          extra: {
            promptOps: {
              ...stubOps({ text: "background done" }),
              prompt: (input) =>
                input.sessionID === chat.id ? Effect.never : Effect.succeed(reply(input, "background done")),
            } satisfies TaskPromptOps,
          },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      const waited = yield* jobs.wait({ id: result.metadata.sessionId, timeout: 1_000 })
      expect(waited.timedOut).toBe(false)
      expect(waited.info?.status).toBe("completed")
    }),
  )

  background.instance("removing the parent session cancels running background tasks", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()

      const result = yield* def.execute(
        {
          description: "inspect bug",
          prompt: "look into the cache key path",
          subagent_type: "general",
          background: true,
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "build",
          abort: new AbortController().signal,
          extra: {
            promptOps: {
              ...stubOps(),
              prompt: () => Effect.never,
            } satisfies TaskPromptOps,
          },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      yield* sessions.remove(chat.id)
      const waited = yield* jobs.wait({ id: result.metadata.sessionId, timeout: 1_000 })
      expect(waited.timedOut).toBe(false)
      expect(waited.info?.status).toBe("cancelled")
    }),
  )

  background.instance("removing the child task session cancels its running background task", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const sessions = yield* Session.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()

      const result = yield* def.execute(
        {
          description: "inspect bug",
          prompt: "look into the cache key path",
          subagent_type: "general",
          background: true,
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "build",
          abort: new AbortController().signal,
          extra: {
            promptOps: {
              ...stubOps(),
              prompt: () => Effect.never,
            } satisfies TaskPromptOps,
          },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      yield* sessions.remove(result.metadata.sessionId)
      const waited = yield* jobs.wait({ id: result.metadata.sessionId, timeout: 1_000 })
      expect(waited.timedOut).toBe(false)
      expect(waited.info?.status).toBe("cancelled")
    }),
  )

  background.instance("cancelling the parent run cancels running background tasks", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const runState = yield* SessionRunState.Service
      const { chat, assistant } = yield* seed()
      const tool = yield* TaskTool
      const def = yield* tool.init()

      const result = yield* def.execute(
        {
          description: "inspect bug",
          prompt: "look into the cache key path",
          subagent_type: "general",
          background: true,
        },
        {
          sessionID: chat.id,
          messageID: assistant.id,
          agent: "build",
          abort: new AbortController().signal,
          extra: {
            promptOps: {
              ...stubOps(),
              prompt: () => Effect.never,
            } satisfies TaskPromptOps,
          },
          messages: [],
          metadata: () => Effect.void,
          ask: () => Effect.void,
        },
      )

      yield* runState.cancel(chat.id)
      const waited = yield* jobs.wait({ id: result.metadata.sessionId, timeout: 1_000 })
      expect(waited.timedOut).toBe(false)
      expect(waited.info?.status).toBe("cancelled")
    }),
  )

  it.instance("cancelling a child run cancels its own pre-runner task job", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const runState = yield* SessionRunState.Service
      const sessions = yield* Session.Service
      const { chat } = yield* seed()
      const child = yield* sessions.create({ parentID: chat.id, title: "child" })

      yield* jobs.start({
        id: child.id,
        type: "task",
        metadata: { parentSessionId: chat.id, sessionId: child.id },
        run: Effect.never,
      })

      yield* runState.cancel(child.id)

      expect((yield* jobs.get(child.id))?.status).toBe("cancelled")
    }),
  )

  it.instance("cancelling a parent run recursively cancels descendant background tasks", () =>
    Effect.gen(function* () {
      const jobs = yield* BackgroundJob.Service
      const runState = yield* SessionRunState.Service
      const sessions = yield* Session.Service
      const { chat } = yield* seed()
      const child = yield* sessions.create({ parentID: chat.id, title: "child" })
      const grandchild = yield* sessions.create({ parentID: child.id, title: "grandchild" })

      yield* jobs.start({
        id: child.id,
        type: "task",
        metadata: { parentSessionId: chat.id, sessionId: child.id },
        run: Effect.never,
      })
      yield* jobs.start({
        id: grandchild.id,
        type: "task",
        metadata: { parentSessionId: child.id, sessionId: grandchild.id },
        run: Effect.never,
      })

      yield* runState.cancel(chat.id)

      expect((yield* jobs.get(child.id))?.status).toBe("cancelled")
      expect((yield* jobs.get(grandchild.id))?.status).toBe("cancelled")
    }),
  )
})
