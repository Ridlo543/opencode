import { describe, expect, it } from "bun:test"
import * as Delegation from "../../src/agent/delegation"
import type { Session } from "../../src/session/session"

function session(input: Partial<Session.Info> = {}): Session.Info {
  return {
    id: "ses_test" as Session.Info["id"],
    slug: "test",
    projectID: "project" as Session.Info["projectID"],
    directory: "/tmp",
    title: "test",
    version: "test",
    time: { created: 1, updated: 1 },
    ...input,
  }
}

describe("agent delegation provenance", () => {
  it("allows private specialists only in matching delegated-task sessions", () => {
    const parentID = "ses_parent"
    const valid = session({
      parentID: parentID as Session.Info["parentID"],
      agent: "orchestra-reviewer",
      metadata: Delegation.metadata({ kind: "delegated-task", parentID, agent: "orchestra-reviewer" }),
    })
    expect(Delegation.selectionError(valid, "orchestra-reviewer", true)).toBeUndefined()
    expect(Delegation.selectionError(valid, "orchestra-reviewer")).toContain("private Orchestra specialist")
    expect(Delegation.selectionError(valid, "orchestra-tester")).toContain("private Orchestra specialist")
    expect(Delegation.selectionError(session({ agent: "build" }), "orchestra-implementer")).toContain(
      "private Orchestra specialist",
    )
    expect(Delegation.selectionError(session({ agent: "build" }), "general")).toBeUndefined()
    expect(Delegation.selectionError(session({ agent: "build" }), "orchestra", false, true)).toBeUndefined()
    expect(Delegation.selectionError(session({ agent: "build" }), "orchestra")).toBeUndefined()
    expect(Delegation.selectionError(session(), "orchestra")).toBeUndefined()
  })

  it("rejects missing, forged, and repair provenance", () => {
    const parentID = "ses_parent"
    const base = {
      parentID: parentID as Session.Info["parentID"],
      agent: "orchestra-implementer",
    }
    expect(Delegation.isDelegatedTo(session(base), "orchestra-implementer")).toBe(false)
    expect(
      Delegation.isDelegatedTo(
        session({
          ...base,
          metadata: Delegation.metadata({ kind: "handoff-repair", parentID, agent: "orchestra-implementer" }),
        }),
        "orchestra-implementer",
      ),
    ).toBe(false)
    expect(
      Delegation.isDelegatedTo(
        session({
          ...base,
          metadata: Delegation.metadata({ kind: "delegated-task", parentID: "ses_other", agent: "orchestra-implementer" }),
        }),
        "orchestra-implementer",
      ),
    ).toBe(false)
  })
})
