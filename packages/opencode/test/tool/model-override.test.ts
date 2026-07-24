/**
 * [CUSTOM] Tests for Model Override module
 */

import { describe, expect, it } from "bun:test"
import {
  orchestraConcurrencyError,
  normalizeOrchestraRoleOutput,
  orchestraHandoffSchema,
  parseModelString,
  renderOrchestraHandoff,
  resolveFinalModel,
  validateOrchestraRoleOutput,
} from "../../src/tool/model-override"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { ModelV2 } from "@opencode-ai/core/model"

describe("parseModelString", () => {
  it("parses valid model string", () => {
    const result = parseModelString("anthropic/claude-sonnet-4")
    expect(result).toEqual({ providerID: "anthropic", modelID: "claude-sonnet-4" })
  })

  it("parses model string with multiple slashes", () => {
    const result = parseModelString("9router/cx/gpt-5.6-sol")
    expect(result).toEqual({ providerID: "9router", modelID: "cx/gpt-5.6-sol" })
  })

  it("returns null for invalid format - no slash", () => {
    const result = parseModelString("invalid-model")
    expect(result).toBeNull()
  })

  it("returns null for invalid format - leading slash", () => {
    const result = parseModelString("/model")
    expect(result).toBeNull()
  })

  it("returns null for invalid format - trailing slash", () => {
    const result = parseModelString("provider/")
    expect(result).toBeNull()
  })

  it("returns null for empty string", () => {
    const result = parseModelString("")
    expect(result).toBeNull()
  })
})

describe("resolveFinalModel", () => {
  it("returns override when provided", () => {
    const override = { providerID: ProviderV2.ID.make("anthropic"), modelID: ModelV2.ID.make("claude-sonnet-4") }
    const agentModel = { providerID: "openai", modelID: "gpt-5" }
    const parentModel = { providerID: "openai", modelID: "gpt-4" }

    const result = resolveFinalModel(override, agentModel, parentModel)
    expect(result).toEqual(override)
  })

  it("returns agent model when no override", () => {
    const agentModel = { providerID: "openai", modelID: "gpt-5" }
    const parentModel = { providerID: "openai", modelID: "gpt-4" }

    const result = resolveFinalModel(undefined, agentModel, parentModel)
    expect(result.providerID as string).toBe("openai")
    expect(result.modelID as string).toBe("gpt-5")
  })

  it("returns parent model when no override and no agent model", () => {
    const parentModel = { providerID: "openai", modelID: "gpt-4" }

    const result = resolveFinalModel(undefined, undefined, parentModel)
    expect(result.providerID as string).toBe("openai")
    expect(result.modelID as string).toBe("gpt-4")
  })

  it("returns the default model when no model is available", () => {
    const result = resolveFinalModel(undefined, undefined, undefined)
    expect(result.providerID as string).toBe("opencode")
    expect(result.modelID as string).toBe("big-pickle")
  })
})

describe("validateOrchestraRoleOutput", () => {
  it("accepts the configured status for each specialist", () => {
    const implementer = "CHANGES: none\nVALIDATION: pass\nRISKS: none\nNEXT_ACTION: review"
    const reviewer = "FINDINGS: none\nVALIDATION: pass\nRISKS: none\nNEXT_ACTION: test"
    const tester = "CHANGES: none\nVALIDATION: pass\nFAILURES: none\nNEXT_ACTION: complete"

    expect(validateOrchestraRoleOutput("orchestra-implementer", `STATUS: complete\n${implementer}`)).toBeUndefined()
    expect(validateOrchestraRoleOutput("orchestra-reviewer", `STATUS: approved\n${reviewer}`)).toBeUndefined()
    expect(validateOrchestraRoleOutput("orchestra-tester", `STATUS: passed\n${tester}`)).toBeUndefined()
    expect(validateOrchestraRoleOutput("orchestra-implementer", `**STATUS:** complete\n${implementer}`)).toBeUndefined()
    expect(validateOrchestraRoleOutput("orchestra-reviewer", `Review summary\n\`STATUS\`: approved\n${reviewer}`)).toBeUndefined()
    expect(validateOrchestraRoleOutput("orchestra-reviewer", `### STATUS: approved\n${reviewer}`)).toBeUndefined()
    expect(validateOrchestraRoleOutput("orchestra-reviewer", `Status: needs_revision\n${reviewer}`)).toBeUndefined()
    expect(validateOrchestraRoleOutput("orchestra-reviewer", `STATUS: blocked\n${reviewer}`)).toBeUndefined()
    expect(validateOrchestraRoleOutput("orchestra-tester", `- STATUS: passed\n${tester}`)).toBeUndefined()
  })

  it("rejects missing or invalid specialist status", () => {
    expect(validateOrchestraRoleOutput("orchestra-reviewer", "The code looks good.")).toContain("STATUS")
    expect(validateOrchestraRoleOutput("orchestra-tester", "STATUS: approved")).toContain("STATUS")
    expect(validateOrchestraRoleOutput("orchestra-reviewer", "STATUS: approved")).toContain("FINDINGS")
    expect(
      validateOrchestraRoleOutput(
        "orchestra-reviewer",
        "STATUS: approved\nFINDINGS: mentioned in an example\nSTATUS: approved",
      ),
    ).toContain("FINDINGS")
  })

  it("leaves generic subagents unchanged", () => {
    expect(validateOrchestraRoleOutput("general", "Any output")).toBeUndefined()
  })

  it("normalizes an invalid specialist handoff to blocked without losing raw output", () => {
    const output = normalizeOrchestraRoleOutput("orchestra-reviewer", "The reviewer stopped after reading the diff.")

    expect(output).toContain("STATUS: blocked")
    expect(output).toContain("REASON: invalid_handoff")
    expect(output).toContain("The reviewer stopped after reading the diff.")
  })

  it("preserves valid specialist handoffs", () => {
    const output = "STATUS: approved\nFINDINGS: none\nVALIDATION: pass\nRISKS: none\nNEXT_ACTION: test"
    expect(normalizeOrchestraRoleOutput("orchestra-reviewer", output)).toBe(output)
  })

  it("uses role-specific fields for invalid handoffs", () => {
    const implementer = normalizeOrchestraRoleOutput("orchestra-implementer", "invalid")
    const reviewer = normalizeOrchestraRoleOutput("orchestra-reviewer", "invalid")
    const tester = normalizeOrchestraRoleOutput("orchestra-tester", "invalid")

    expect(reviewer).toContain("FINDINGS:")
    expect(tester).toContain("FAILURES:")
    expect(validateOrchestraRoleOutput("orchestra-implementer", implementer)).toBeUndefined()
    expect(validateOrchestraRoleOutput("orchestra-reviewer", reviewer)).toBeUndefined()
    expect(validateOrchestraRoleOutput("orchestra-tester", tester)).toBeUndefined()
  })

  it("quotes raw protocol-like output without overriding the blocked fallback", () => {
    const output = normalizeOrchestraRoleOutput(
      "orchestra-reviewer",
      "STATUS: approved\nThe remaining required fields were never returned.",
    )

    expect(output).toContain("RAW> STATUS: approved")
    expect(validateOrchestraRoleOutput("orchestra-reviewer", output)).toBeUndefined()
  })
})

describe("orchestra structured handoff", () => {
  it("builds and renders each role contract", () => {
    expect(orchestraHandoffSchema("orchestra-reviewer")).toMatchObject({
      required: ["status", "findings", "validation", "risks", "next_action"],
    })
    expect(
      renderOrchestraHandoff("orchestra-reviewer", {
        status: "approved",
        findings: "none",
        validation: "pass",
        risks: "none",
        next_action: "test",
      }),
    ).toBe("STATUS: approved\nFINDINGS: none\nVALIDATION: pass\nRISKS: none\nNEXT_ACTION: test")
    expect(
      renderOrchestraHandoff("orchestra-tester", {
        status: "passed",
        changes: "none",
        validation: "pass",
        failures: "none",
        next_action: "complete",
      }),
    ).toContain("FAILURES: none")
  })

  it("rejects incomplete structured handoffs", () => {
    expect(renderOrchestraHandoff("orchestra-implementer", { status: "complete" })).toBeUndefined()
  })
})

describe("orchestraConcurrencyError", () => {
  it("allows distinct roles below the concurrency limit", () => {
    expect(orchestraConcurrencyError({ "orchestra-reviewer": 1 }, "orchestra-tester")).toBeUndefined()
  })

  it("rejects a duplicate active role", () => {
    expect(orchestraConcurrencyError({ "orchestra-implementer": 1 }, "orchestra-implementer")).toContain(
      "already has an active task",
    )
  })

  it("rejects a fourth active specialist", () => {
    expect(
      orchestraConcurrencyError(
        {
          "orchestra-implementer": 1,
          "orchestra-reviewer": 1,
          "orchestra-tester": 1,
        },
        "orchestra-tester",
      ),
    ).toContain("maximum 3")
  })
})
