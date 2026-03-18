# ganglion-memory

Biological memory for agents. One line to add, zero lines to change.

## Quick start

```bash
pip install ganglion-memory
```

```python
from ganglion.memory import memory

agent = memory(agent)
```

Your agent now accumulates knowledge across runs. What worked gets strengthened. What failed gets recorded. Contradictions get detected. The strongest beliefs surface first in prompts.

## How it works

Before each call, `memory()` injects accumulated knowledge into your agent's system prompt. After each call, it evaluates the response and feeds the outcome back. The agent itself never changes.

```python
from openai import OpenAI
from ganglion.memory import memory

client = OpenAI()

def ask(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a research assistant."},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content

ask = memory(ask)
ask("What batch size should I use?")
```

Works with OpenAI, Anthropic, or any function that takes a prompt and returns a response. Works as a decorator:

```python
@memory
def review(code: str) -> str:
    ...
```

## Configuration

```python
agent = memory(
    agent,
    capability="mining",        # what domain (groups beliefs)
    bot_id="alpha",             # unique agent id (for cross-agent learning)
    db_path="memory.db",        # where to store beliefs
    judge=my_judge_fn,          # custom success/failure evaluation
)
```

## Custom judge

By default, the wrapper auto-detects OpenAI and Anthropic response types and records them as successful. For domain-specific evaluation:

```python
def my_judge(response) -> dict:
    score = extract_score(response)
    return {
        "success": score > 0.5,
        "description": f"Got score {score}",
        "metric_name": "accuracy",
        "metric_value": score,
    }

agent = memory(agent, judge=my_judge)
```

The judge returns a dict with `success` (required) and optionally `description`, `metric_name`, `metric_value`, `config`, `tags`, `error`.

## Full API

For more control, use the components directly:

```python
from ganglion.memory import MemoryLoop, MemoryAgent, SqliteMemoryBackend, between_runs

# Create the memory store
mem = MemoryLoop(backend=SqliteMemoryBackend("memory.db"))

# Create a memory-aware agent wrapper
agent = MemoryAgent(memory=mem, capability="mining", bot_id="alpha")

# Three touch points:
context = await agent.remember()           # before — inject into prompt
result = await your_agent.run(context)     # act — unchanged
delta = await agent.learn(result_dict)     # after — feed outcome back

# Between runs — consolidate and forget
await between_runs(mem)
```

## MemoryLoop parameters

```python
MemoryLoop(
    backend=SqliteMemoryBackend("memory.db"),
    strengthen_rate=0.1,          # confidence boost on agreement
    weaken_rate=0.3,              # confidence reduction on contradiction
    death_threshold=0.1,          # beliefs below this die
    metric_shift_threshold=0.15,  # fractional change to count as drift
    max_beliefs=1000,             # capacity before eviction
    salience=True,                # surprise-gated encoding strength
    inhibition_rate=0.05,         # lateral inhibition between competitors
    inhibition_floor=0.2,         # inhibition can't kill, only weaken
    exploration_rate=0.0,         # noise on retrieval ranking (0 = deterministic)
    cross_agent_bonus=2.0,        # independent replication worth 2x self-confirmation
    crisis_multiplier=3.0,        # plasticity boost on consecutive contradictions
    consolidation_threshold=0.5,  # Jaccard threshold for merging similar beliefs
)
```

## Backends

```python
from ganglion.memory import SqliteMemoryBackend, JsonMemoryBackend, FederatedMemoryBackend, PeerDiscovery

# SQLite — production use, one table, indexed
backend = SqliteMemoryBackend("memory.db")

# JSON — development, one file
backend = JsonMemoryBackend("./memory/")

# Federated — multi-agent on shared filesystem
local = JsonMemoryBackend(f"/shared/bots/{bot_id}")
backend = FederatedMemoryBackend(local, PeerDiscovery("/shared/bots", bot_id))
```

## Cortex — advanced retrieval

```python
from ganglion.memory.cortex import spread_activation, temporal_neighbors

# Find beliefs associated with a seed through shared entities/tags
related = await spread_activation(seed_belief, backend, max_hops=1, limit=10)

# Find beliefs from the same time period
neighbors = await temporal_neighbors(belief, backend, window=timedelta(hours=1))
```

## Biology

The system implements five biological memory mechanisms:

* **Hebbian strengthening** — repeated confirmation increases belief strength
* **Apoptosis** — beliefs contradicted enough times die and get replaced
* **Salience** — surprising observations encode with higher initial strength
* **Lateral inhibition** — strengthening one strategy weakens alternatives
* **Consolidation** — similar beliefs merge into meta-beliefs during the sleep phase (between runs)

Three emergence mechanisms:

* **Cross-agent confirmation** — independent replication weighted higher than self-confirmation
* **Exploration pressure** — Gaussian noise on retrieval prevents premature convergence
* **Strategy bundling** — beliefs from the same run are tagged for coherent retrieval

One adaptation mechanism:

* **Crisis detection** — consecutive contradictions temporarily increase plasticity

## Types

Three types:

* **Observation** — everything that enters memory (capability, description, valence, entities, tags, metrics)
* **Belief** — everything memory stores (observation fields + confidence, confirmation_count, strength)
* **Delta** — everything memory notices changing (contradiction or metric shift)

## License

MIT
