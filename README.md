# ganglion-memory

Dumb memory, smart retrieval — let the LLM think.

## Quick start

```bash
pip install ganglion-memory
```

```python
from ganglion.memory import memory

agent = memory(agent)
```

Your agent now accumulates experience across runs. One call per invocation — no counterfactual evaluation, no double-call. Before each call, relevant experience is injected into the prompt. After each call, the outcome is stored.

## How it works

One type: **Experience**. No Observation/Belief/Delta split. No confidence scoring. No biological metaphors. Just content, tags, confirmation/contradiction counts, and an optional embedding vector.

```python
from ganglion.memory import Memory, SqliteBackend

mem = Memory(backend=SqliteBackend("memory.db"))

# Store
exp = await mem.add("batch_size=64 works well", tags=("mining",))

# Search
results = await mem.search("what batch size?", tags=("mining",))

# Confirm / Contradict
await mem.confirm(exp.id)
await mem.contradict(exp.id)

# Compress similar experiences
await mem.compress(tags=("mining",))
```

## Agent wrapper

Three touch points, any agent, any task:

```python
from ganglion.memory import Memory, Agent, SqliteBackend

mem = Memory(backend=SqliteBackend("memory.db"))
agent = Agent(memory=mem, capability="mining", bot_id="alpha")

# 1. BEFORE acting — inject context into prompt
context = await agent.remember(query="optimize batch size")

# 2. AFTER acting — store the outcome
exp = await agent.learn(result_dict, input_text="...", output_text="...")

# 3. BETWEEN runs — compress similar experiences
await agent.between_runs()
```

## One-line wrapper

```python
from ganglion.memory import memory

@memory
def ask(question: str) -> str:
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": question},
        ],
    ).choices[0].message.content

ask("What batch size should I use?")
```

Works with OpenAI, Anthropic, or any function that takes a prompt and returns a response.

## Configuration

```python
agent = memory(
    agent,
    capability="mining",     # domain grouping
    bot_id="alpha",          # unique agent id
    db_path="memory.db",     # SQLite path
    judge=my_judge_fn,       # custom success/failure evaluation
    embedder=my_embedder,    # custom embedding function
)
```

## ReMem-style refine tools

Opt-in tools for LLM-driven memory management:

```python
from ganglion.memory.refine import (
    merge_experiences,
    split_experience,
    rewrite_experience,
    forget_experience,
)

# LLM decides two experiences should be merged
merged = await merge_experiences(mem, [exp1.id, exp2.id], "combined insight")

# LLM splits a compound experience
parts = await split_experience(mem, exp.id, ["fact A", "fact B"])

# LLM rewrites stale memory
updated = await rewrite_experience(mem, exp.id, "corrected understanding")

# LLM decides to forget
await forget_experience(mem, exp.id)
```

## What changed from v3

- **One type**: Experience replaces Observation/Belief/Delta
- **No biological metaphors**: No Hebbian strengthening, lateral inhibition, apoptosis, or crisis detection
- **No counterfactual evaluation**: One LLM call per invocation (not two)
- **No confidence scoring**: Raw confirmation/contradiction counts — let the LLM interpret
- **Embedding-based retrieval** with optional LLM synthesis on compress
- **ReMem-inspired refine tools** for active memory editing (opt-in)

## What stayed

- `memory(fn)` one-line wrapper still works
- `SqliteBackend` with embedding blob storage
- `Embedder` protocol (SentenceTransformer, CallableEmbedder)
- Agent wrapper with remember/learn/between_runs

## License

MIT
