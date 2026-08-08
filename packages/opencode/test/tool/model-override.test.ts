/**
 * [CUSTOM] Tests for Model Override module
 */

import { describe, expect, it } from "bun:test"
import {
  ORCHESTRA_MAX_BLOCKED_ATTEMPTS,
  orchestraBlockedRecovery,
  orchestraConcurrencyError,
  orchestraHandoffInstruction,
  isOrchestraLead,
  workflowTaskAccessError,
  normalizeOrchestraRoleOutput,
  parseModelString,
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

  it("accepts human-readable implementer headings with summary and files changed", () => {
    const output = [
      "STATUS: complete",
      "SUMMARY: Closed the fixture drift finding.",
      "FILES_CHANGED:",
      "- tools/regenerate-binding-fixtures.mjs",
      "- Enforces the exact owned fixture tree.",
      "VALIDATION:",
      "- pnpm curriculum:review-evidence:test - passed",
      "RISKS:",
      "- Existing unrelated worktree changes were preserved.",
      "NEXT_ACTION:",
      "- Lead can review the scoped change.",
    ].join("\n")

    expect(validateOrchestraRoleOutput("orchestra-implementer", output)).toBeUndefined()
    expect(normalizeOrchestraRoleOutput("orchestra-implementer", output)).toBe(output)
  })

  it("accepts a substantive reviewer finding without every suggested heading", () => {
    const output = [
      "STATUS: needs_revision",
      "SUMMARY:",
      "The prior root findings are addressed, but one begin/completion race violates completed replay semantics.",
      "FINDINGS:",
      "- MEDIUM — A duplicate request racing successful completion can return ErrUnavailable instead of the committed replay.",
      "Evidence: begin checks for a result before locking the claim. Completion can insert the result and delete the claim while begin waits, so begin resumes without rechecking the committed result.",
      "After a missing claim, begin should recheck for a committed result before failing and add a deterministic concurrency test.",
    ].join("\n")

    expect(validateOrchestraRoleOutput("orchestra-reviewer", output)).toBeUndefined()
    expect(normalizeOrchestraRoleOutput("orchestra-reviewer", output)).toBe(output)
  })

  it("accepts long-form implementer reports with design invariants and tests run", () => {
    const output = [
      "STATUS: complete",
      "SUMMARY: Audited the existing materialization implementation and corrected fail-closed mapping defects.",
      "FILES_CHANGED:",
      "- migrations/00059_curriculum_intake_edition_materialization.sql",
      "- internal/domain/curriculum/materialization/service.go",
      "DESIGN_INVARIANTS:",
      "- Materialization remains draft, non-serving, and non-RAG.",
      "- Unsupported content fails closed.",
      "TESTS_RUN:",
      "- go test ./... -count=1",
      "- pnpm db:vet",
      "- git diff --check",
      "RISKS: none identified",
      "NEXT_ACTION: Lead should review the scoped uncommitted diff.",
    ].join("\n")

    expect(validateOrchestraRoleOutput("orchestra-implementer", output)).toBeUndefined()
    expect(normalizeOrchestraRoleOutput("orchestra-implementer", output)).toBe(output)
  })

  it("accepts bounded validation aliases for every Orchestra role", () => {
    for (const heading of [
      "VALIDATION",
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
    ]) {
      expect(
        validateOrchestraRoleOutput(
          "orchestra-implementer",
          `STATUS: complete\nCHANGES: changed src/a.ts\n${heading}: passed\nRISKS: none\nNEXT_ACTION: review`,
        ),
      ).toBeUndefined()
      expect(
        validateOrchestraRoleOutput(
          "orchestra-reviewer",
          `STATUS: approved\nFINDINGS: none\n${heading}: passed\nRISKS: none\nNEXT_ACTION: test`,
        ),
      ).toBeUndefined()
      expect(
        validateOrchestraRoleOutput(
          "orchestra-tester",
          `STATUS: passed\nCHANGES: none\n${heading}: passed\nFAILURES: none\nNEXT_ACTION: complete`,
        ),
      ).toBeUndefined()
    }
  })

  it("accepts common semantic heading families across Orchestra roles", () => {
    const implementerChanges = [
      "CHANGES",
      "FILES_CHANGED",
      "CHANGED_FILES",
      "FILES_MODIFIED",
      "MODIFIED_FILES",
      "IMPLEMENTATION",
      "IMPLEMENTATION_SUMMARY",
      "MODIFICATIONS",
      "WORK_COMPLETED",
    ]
    const risks = ["RISKS", "KNOWN_RISKS", "REMAINING_RISKS", "RESIDUAL_RISKS"]
    const nextActions = ["NEXT_ACTION", "NEXT_ACTIONS", "NEXT_STEP", "NEXT_STEPS", "FOLLOW_UP", "FOLLOW_UP_ACTIONS"]
    const findings = ["FINDINGS", "ISSUES", "ISSUES_FOUND", "REVIEW_FINDINGS", "REVIEW_RESULTS"]
    const failures = ["FAILURES", "FAILED_TESTS", "FAILING_TESTS", "TEST_FAILURES", "TEST_ISSUES"]

    for (const heading of implementerChanges) {
      expect(
        validateOrchestraRoleOutput(
          "orchestra-implementer",
          `STATUS: complete\n${heading}: changed src/a.ts\nVALIDATION: passed\nRISKS: none\nNEXT_ACTION: review`,
        ),
      ).toBeUndefined()
    }
    for (const heading of risks) {
      expect(
        validateOrchestraRoleOutput(
          "orchestra-reviewer",
          `STATUS: approved\nFINDINGS: none\nVALIDATION: passed\n${heading}: none\nNEXT_ACTION: test`,
        ),
      ).toBeUndefined()
    }
    for (const heading of nextActions) {
      expect(
        validateOrchestraRoleOutput(
          "orchestra-tester",
          `STATUS: passed\nCHANGES: none\nVALIDATION: passed\nFAILURES: none\n${heading}: complete`,
        ),
      ).toBeUndefined()
    }
    for (const heading of findings) {
      expect(
        validateOrchestraRoleOutput(
          "orchestra-reviewer",
          `STATUS: approved\n${heading}: none\nVALIDATION: passed\nRISKS: none\nNEXT_ACTION: test`,
        ),
      ).toBeUndefined()
    }
    for (const heading of failures) {
      expect(
        validateOrchestraRoleOutput(
          "orchestra-tester",
          `STATUS: passed\nCHANGES: none\nVALIDATION: passed\n${heading}: none\nNEXT_ACTION: complete`,
        ),
      ).toBeUndefined()
    }
    for (const heading of [...implementerChanges, "TEST_CHANGES", "TESTS_ADDED"]) {
      expect(
        validateOrchestraRoleOutput(
          "orchestra-tester",
          `STATUS: passed\n${heading}: tests updated\nVALIDATION: passed\nFAILURES: none\nNEXT_ACTION: complete`,
        ),
      ).toBeUndefined()
    }
  })

  it("accepts unknown but structurally complete heading vocabularies", () => {
    for (const role of ["orchestra-implementer", "orchestra-reviewer", "orchestra-tester"]) {
      const status = role === "orchestra-implementer" ? "complete" : role === "orchestra-reviewer" ? "approved" : "passed"
      expect(
        validateOrchestraRoleOutput(
          role,
          [
            `STATUS: ${status}`,
            "WORK_LOG: Updated the scoped implementation and preserved unrelated files.",
            "QUALITY_GATES: Targeted tests, typecheck, and diff validation all passed.",
            "OPEN_CONCERNS: No known blocking concerns remain after validation.",
            "HANDOFF_TARGET: Lead should inspect the scoped diff and continue the workflow.",
          ].join("\n"),
        ),
      ).toBeUndefined()
    }
  })

  it("accepts varied enum-independent structural headings", () => {
    const vocabularies = [
      ["DELIVERED_WORK", "PROOF_OF_CHECKS", "UNRESOLVED_ITEMS", "ROUTING_DECISION"],
      ["HASIL_KERJA", "BUKTI_VALIDASI", "CATATAN_RISIKO", "TINDAK_LANJUT"],
      ["PATCH_OVERVIEW", "CONFIDENCE_EVIDENCE", "CAVEATS", "OWNER_HANDOFF"],
    ]
    for (const headings of vocabularies) {
      expect(
        validateOrchestraRoleOutput(
          "orchestra-implementer",
          [
            "STATUS: complete",
            `${headings[0]}: Completed the requested scoped implementation with deterministic behavior.`,
            `${headings[1]}: Ran targeted tests and repository validation successfully.`,
            `${headings[2]}: No blocking concern remains; unrelated worktree files were preserved.`,
            `${headings[3]}: Lead should review the scoped diff and continue to the next gate.`,
          ].join("\n"),
        ),
      ).toBeUndefined()
    }
  })

  it("accepts concise substantive handoffs without canonical headings but rejects placeholder bodies", () => {
    expect(
      validateOrchestraRoleOutput(
        "orchestra-implementer",
        "STATUS: complete\nWORK_LOG: changed code\nQUALITY_GATES: passed\nHANDOFF_TARGET: review",
      ),
    ).toBeUndefined()
    expect(
      validateOrchestraRoleOutput(
        "orchestra-implementer",
        "STATUS: complete\nSUMMARY: long summary text\nNOTES: more notes\nSCOPE: files\nDETAILS: details",
      ),
    ).toContain("concrete validation")
    expect(
      validateOrchestraRoleOutput(
        "orchestra-implementer",
        "STATUS: complete\nA: one\nB: two\nC: three\nD: four",
      ),
    ).toContain("substantive handoff")
  })

  it("accepts bounded aliases, Markdown headings, and field order variations", () => {
    expect(
      validateOrchestraRoleOutput(
        "orchestra-reviewer",
        [
          "- **STATUS:** approved",
          "## Checks",
          "targeted tests passed",
          "### Review Findings",
          "none",
          "NEXT STEPS: proceed to test",
          "REMAINING-RISKS: none",
        ].join("\n"),
      ),
    ).toBeUndefined()
    expect(
      validateOrchestraRoleOutput(
        "orchestra-tester",
        "STATUS: passed\nTESTS: suite passed\nTEST CHANGES: none\nFAILING TESTS: none\nNEXT ACTION: complete",
      ),
    ).toBeUndefined()
  })

  it("accepts CRLF, multiline values, and a final valid alias declaration", () => {
    expect(
      validateOrchestraRoleOutput(
        "orchestra-implementer",
        [
          "STATUS: complete",
          "CHANGES:",
          "FILES_CHANGED:",
          "- src/a.ts",
          "- src/b.ts",
          "VALIDATION: targeted tests passed",
          "RISKS: none",
          "NEXT_STEPS: review",
        ].join("\r\n"),
      ),
    ).toBeUndefined()
  })

  it("keeps commands, URLs, Windows paths, and prose labels inside section content", () => {
    expect(
      validateOrchestraRoleOutput(
        "orchestra-implementer",
        [
          "STATUS: complete",
          "CHANGES: updated src/a.ts",
          "VALIDATION:",
          "Command: pnpm test:unit",
          "URL: https://example.test/check",
          "Path: C:\\workspace\\report.txt",
          "Result: passed",
          "RISKS: none",
          "NEXT_ACTION: review",
        ].join("\n"),
      ),
    ).toBeUndefined()
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

  it("accepts summary-only and partially populated suggested sections when the body is substantive", () => {
    expect(
      validateOrchestraRoleOutput(
        "orchestra-implementer",
        "STATUS: complete\nCHANGES:\nVALIDATION: pass\nRISKS: none\nNEXT_ACTION: review",
      ),
    ).toBeUndefined()
    expect(
      validateOrchestraRoleOutput(
        "orchestra-implementer",
        "STATUS: complete\nSUMMARY: changed files\nVALIDATION: pass\nRISKS: none\nNEXT_ACTION: review",
      ),
    ).toBeUndefined()
    expect(
      validateOrchestraRoleOutput(
        "orchestra-reviewer",
        "STATUS: approved\nSUMMARY: looks good\nVALIDATION: pass\nRISKS: none\nNEXT_ACTION: test",
      ),
    ).toBeUndefined()
  })

  it("accepts readable surrounding evidence even when a suggested field uses a placeholder", () => {
    for (const changes of ["<text>", "...", "-", "TODO"]) {
      expect(
        validateOrchestraRoleOutput(
          "orchestra-implementer",
          `STATUS: complete\nCHANGES: ${changes}\nVALIDATION: pass\nRISKS: none\nNEXT_ACTION: review`,
        ),
      ).toBeUndefined()
    }
    expect(
      validateOrchestraRoleOutput(
        "orchestra-implementer",
        "STATUS: complete\nCHANGES:\nDETAILS: changed src/a.ts\nVALIDATION: pass\nRISKS: none\nNEXT_ACTION: review",
      ),
    ).toBeUndefined()
  })

  it("uses only the substantive body after the final valid status", () => {
    const validBeforeFinalStatus = [
      "STATUS: complete",
      "CHANGES: changed src/a.ts",
      "VALIDATION: pass",
      "RISKS: none",
      "NEXT_ACTION: review",
      "STATUS: complete",
      "SUMMARY: no final fields",
    ].join("\n")
    expect(validateOrchestraRoleOutput("orchestra-implementer", validBeforeFinalStatus)).toContain("substantive handoff")

    const emptiedAtEnd = [
      "STATUS: complete",
      "CHANGES: changed src/a.ts",
      "FILES_CHANGED:",
      "VALIDATION: pass",
      "RISKS: none",
      "NEXT_ACTION: review",
    ].join("\n")
    expect(validateOrchestraRoleOutput("orchestra-implementer", emptiedAtEnd)).toBeUndefined()
  })

  it("ignores quoted and fenced protocol examples", () => {
    const quoted = [
      "> STATUS: complete",
      "> CHANGES: changed src/a.ts",
      "> VALIDATION: pass",
      "> RISKS: none",
      "> NEXT_ACTION: review",
    ].join("\n")
    expect(validateOrchestraRoleOutput("orchestra-implementer", quoted)).toContain("STATUS")

    const fenced = [
      "```text",
      "STATUS: complete",
      "CHANGES: changed src/a.ts",
      "VALIDATION: pass",
      "RISKS: none",
      "NEXT_ACTION: review",
      "```",
    ].join("\n")
    expect(validateOrchestraRoleOutput("orchestra-implementer", fenced)).toContain("STATUS")

    const exampleThenFinal = [
      fenced,
      "STATUS: complete",
      "FILES_CHANGED: src/a.ts",
      "VALIDATION: pass",
      "RISKS: none",
      "NEXT_ACTION: review",
    ].join("\n")
    expect(validateOrchestraRoleOutput("orchestra-implementer", exampleThenFinal)).toBeUndefined()
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

  it("blocks a success status without concrete validation evidence", () => {
    const output = normalizeOrchestraRoleOutput(
      "orchestra-reviewer",
      "STATUS: approved\nThe remaining required fields were never returned.",
    )

    expect(output).toContain("STATUS: blocked")
    expect(output).toContain("REASON: invalid_handoff")
  })
})

describe("orchestra handoff instruction", () => {
  it("describes each role contract without provider-specific structured output", () => {
    expect(orchestraHandoffInstruction("orchestra-implementer")).toContain("CHANGES")
    expect(orchestraHandoffInstruction("orchestra-reviewer")).toContain("FINDINGS")
    expect(orchestraHandoffInstruction("orchestra-tester")).toContain("FAILURES")
    expect(orchestraHandoffInstruction("general")).toBeUndefined()
  })
})

describe("orchestraConcurrencyError", () => {
  it("allows distinct roles below the concurrency limit", () => {
    expect(orchestraConcurrencyError({ "orchestra-reviewer": 1 }, "orchestra-tester")).toBeUndefined()
  })

  it("allows concurrent tasks with the same role below the total limit", () => {
    expect(orchestraConcurrencyError({ "orchestra-reviewer": 3 }, "orchestra-reviewer")).toBeUndefined()
  })

  it("allows four concurrent implementers", () => {
    expect(orchestraConcurrencyError({ "orchestra-implementer": 3 }, "orchestra-implementer")).toBeUndefined()
  })

  it("rejects a fifth active implementer", () => {
    expect(
      orchestraConcurrencyError({ "orchestra-implementer": 4 }, "orchestra-implementer"),
    ).toContain("maximum 4")
  })
})

describe("orchestraBlockedRecovery", () => {
  it("provides distinct bounded recovery strategies through attempt five", () => {
    const outputs = Array.from({ length: ORCHESTRA_MAX_BLOCKED_ATTEMPTS }, (_, index) =>
      orchestraBlockedRecovery(index + 1),
    )
    expect(new Set(outputs).size).toBe(ORCHESTRA_MAX_BLOCKED_ATTEMPTS)
    for (let index = 0; index < outputs.length; index++) {
      expect(outputs[index]).toContain(`attempt ${index + 1} of 5`)
      expect(outputs[index]).toContain("RECOVERY_STRATEGY:")
    }
    expect(outputs[0]).toContain("4 further implementer attempts remain")
    expect(outputs[4]).toContain("Terminal for this chat turn")
    expect(outputs[4]).toContain("do not start a sixth implementer attempt")
    expect(outputs[4]).toContain("new user message starts a fresh attempt budget")
    expect(orchestraBlockedRecovery(99)).toBe(outputs[4])
  })
})

describe("workflowTaskAccessError", () => {
  it("keeps Orchestra specialists private to the Orchestra Lead", () => {
    expect(workflowTaskAccessError("orchestra", "orchestra-assistant-specialist")).toBeUndefined()
    expect(workflowTaskAccessError("orchestra", "orchestra-implementer")).toBeUndefined()
    expect(workflowTaskAccessError("orchestra", "orchestra-reviewer")).toBeUndefined()
    expect(workflowTaskAccessError("orchestra", "orchestra-tester")).toBeUndefined()
    expect(workflowTaskAccessError("build", "orchestra-implementer")).toContain("private Orchestra specialist")
    expect(workflowTaskAccessError("plan", "orchestra-reviewer")).toContain("private Orchestra specialist")
    expect(workflowTaskAccessError("build", "orchestra-assistant-specialist")).toContain(
      "private Orchestra specialist",
    )
    expect(workflowTaskAccessError("orchestra-implementer", "orchestra-assistant-specialist")).toContain(
      "cannot delegate tasks",
    )
    expect(workflowTaskAccessError("orchestra-assistant-specialist", "general")).toContain("cannot delegate tasks")
    expect(workflowTaskAccessError("build", "general")).toBeUndefined()
    expect(workflowTaskAccessError("orchestra", "general")).toContain("must delegate using")
    expect(workflowTaskAccessError("orchestra-custom", "orchestra-implementer")).toBeUndefined()
    expect(workflowTaskAccessError("orchestra-custom", "orchestra-reviewer")).toBeUndefined()
    expect(workflowTaskAccessError("orchestra-custom", "general")).toContain("must delegate using")
  })

  it("recognizes both Orchestra Lead profiles", () => {
    expect(isOrchestraLead("orchestra")).toBe(true)
    expect(isOrchestraLead("orchestra-custom")).toBe(true)
    expect(isOrchestraLead("build")).toBe(false)
  })

  it("keeps Research specialists private to the Research Lead", () => {
    expect(workflowTaskAccessError("research", "research-scout")).toBeUndefined()
    expect(workflowTaskAccessError("research", "research-methodologist")).toBeUndefined()
    expect(workflowTaskAccessError("research", "research-editor")).toBeUndefined()
    expect(workflowTaskAccessError("research", "general")).toContain("must delegate using")
    expect(workflowTaskAccessError("build", "research-analyst")).toContain("private Research specialist")
    expect(workflowTaskAccessError("orchestra", "research-critic")).toContain("must delegate using")
    expect(workflowTaskAccessError("research-writer", "research-reviewer")).toContain("cannot delegate tasks")
  })
})
