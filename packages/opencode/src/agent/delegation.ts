import type { Session } from "@/session/session"
import { isOrchestraRole } from "@/tool/model-override"

const KIND = "opencode.task.kind"
const PARENT = "opencode.task.parent"
const AGENT = "opencode.task.agent"
const KEYS = new Set([KIND, PARENT, AGENT])

export type Kind = "delegated-task" | "handoff-repair"

export function metadata(input: { kind: Kind; parentID: string; agent: string }) {
  return {
    [KIND]: input.kind,
    [PARENT]: input.parentID,
    [AGENT]: input.agent,
  }
}

export function userMetadata(value: Session.Info["metadata"]) {
  return value ? Object.fromEntries(Object.entries(value).filter(([key]) => !KEYS.has(key))) : undefined
}

export function internalMetadata(value: Session.Info["metadata"]) {
  return value ? Object.fromEntries(Object.entries(value).filter(([key]) => KEYS.has(key))) : undefined
}

export function matches(session: Session.Info, input: { kind: Kind; parentID: string; agent: string }) {
  return (
    session.metadata?.[KIND] === input.kind &&
    session.metadata?.[PARENT] === input.parentID &&
    session.metadata?.[AGENT] === input.agent &&
    session.parentID === input.parentID &&
    session.agent === input.agent
  )
}

export function isDelegatedTo(session: Session.Info, agent: string) {
  return !!session.parentID && matches(session, { kind: "delegated-task", parentID: session.parentID, agent })
}

export function selectionError(session: Session.Info, agent: string, delegated = false, workflowTransition = false) {
  if (isOrchestraRole(agent)) {
    if (delegated && isDelegatedTo(session, agent)) return undefined
    return `${agent} is a private Orchestra specialist and cannot run as a primary session agent.`
  }
  return undefined
}
