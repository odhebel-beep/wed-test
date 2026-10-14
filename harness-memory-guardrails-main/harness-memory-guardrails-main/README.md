# LLM memory and guardrails: Codespaces demo

Uses OpenAI Responses API. 
Default model: `gpt-5.4-mini`

Exit saves an empty session: the LLM must record something such as “A
session was opened and ended without discussion.” 

**Exit is an app button, not closing the browser tab.** Closing the tab, stopping
the Codespace, or killing Python does not trigger a save. Wait for the saved
confirmation before leaving. Submit any typed message before clicking Exit;
unsent text is not part of the conversation.

| File / function | Purpose |
| --- | --- |
| `app.py`: `TOOLS` | JSON schemas describing three callable tools |
| `read_guardrails` | Reads the instructor policy |
| `read_memory` | Reads previous sessions from disk |
| `update_memory` | Validates entries; locks and atomically updates the file |
| `chat` | Runs the model/tool loop; excludes the writing tool |
| `exit_session` | Forces a memory tool call and confirms a successful write |
| `guardrails.md` | Editable instructor instructions |
| `memory.md` | Inspectable persistent notes; no model weights are updated |

The guardrail is behavioral guidance the model reads at the start of a session:
it must never criticize Fort Worth. Edit `guardrails.md` to change the policy,
then start a new session so the model reads the new instructions.

## Persistence and scope

- Memory survives browser refreshes, new sessions, and restarting the same
  Codespace because it is in the workspace. Current chat history is browser-session
  state and can be lost on refresh; only Exit writes durable memory.
- A different/new Codespace does not inherit uncommitted memory from the old one.
  Deleting the Codespace can lose local changes. Commit examples if you
  want them in Git; use a database for centrally managed production memory.
- One `memory.md` is shared by everyone using this app instance. This demo does
  not implement user accounts or separate users' memories. Separate student
  Codespaces naturally have separate files.
- Memory is an append-only session journal. It does not implement expiration,
  semantic retrieval, automatic correction of old facts, or context compaction.
  All stored memory is supplied at session initialization, so keep it small.
- Saved summaries can omit details or be wrong; inspect them. Memory/transcripts
  are sent to OpenAI to produce replies and summaries. `store=False` disables
  Responses storage but is not a general zero-retention guarantee.
- To reset, restore `memory.md` to its initial content and start a new session.