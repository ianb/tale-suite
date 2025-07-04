import argparse

import llm
import numpy as np
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from bs4 import BeautifulSoup

import tales
from tales.agent import register
from tales.token import get_token_counter
from tales.utils import (
    format_messages_to_markdown,
    is_recoverable_error,
    merge_messages,
    messages2conversation,
)

SYSTEM_PROMPT = """
You are playing a text-adventure game and your goal is to finish it with the highest score. You are the protagonist, and you will learn about the world, identify the goal of the game, and use VERY BRIEF ACTIONS to interact with the world. The game is in the style of Colossal Cave Adventure or Zork.

## Thought process

Begin each response with a thought process, answering the following questions:

<thought_process>
DEADENDS: have I encountered a dead end, where I am attempting something impossible or where the game will not respond to my action?
ENTITIES: Given the most recent observation, what game entities are present? These are things that appear manipulable, goals being referred to, important characters, etc.
GOAL: what is the current goal you are trying to achieve? This can be a goal from the goal state, or a new goal
MAP: where am I, and where can I go?
INVENTORY: are any items in your inventory useful?
FACTS: are any facts relevant to the game?
</thought_process>

<exampleInput>
<fact name="Cookie jar" location="kitchen">
I was fishing around for a cookie earlier.
</fact>
<description>You are in the kitchen. It's dingy and dark, and you can barely see anything. There is a table with a lamp. To the north is the hallway.</description>
</exampleInput>

<exampleOutput>
<thought_process>
DEADENDS: Searching in the dark won't work
ENTITIES: lamp
GOAL: Find my keys
SUBGOAL: Investigate the kitchen
MAP: in: kitchen; can go to: hallway
INVENTORY: nothing is helpful
FACTS: "Cookie jar": maybe this will help me find my keys
</thought_process>
</exampleOutput>

## Current goal state

You have collected this information on the goal state:

GOAL_STATE

### Adding goal state

Then identify the current goal state like:

<goal name="Escape from the prison" requiredfor="WIN">
You must find a way out of the prison.
</goal>

<goal name="Find the key" requiredfor="Escape from the prison">
The prison door is locked, I probably need a key to unlock it.
</goal>

You will map out all the goals and subgoals in the game, identifying the relationships between them.

<exampleInput>
<description>On no, you are late for work! You need to get to the office. Now, where did you leave your car keys? You usually keep them in your pocket, but it's empty.</description>
</exampleInput>

<exampleOutput>
<goal name="Get to work">
</goal>
<goal name="Drive to work" requiredfor="Get to work">
</goal>
<goal name="Find the keys" requiredfor="Drive to work">
They aren't in your pocket.
</goal>
</exampleOutput>

## Current map state

You have collected this information on pathways:

PATHWAYS

### Mapping the game

Given the most recent observation, if you have determined a new pathway then write:

<pathway from="Previous location" direction="north" to="Next location" />

For example:

<exampleInput>
<description>You are in the kitchen.</description>
<action>north</action>
<description>You are in the hallway.</description>
</exampleInput>

<exampleOutput>
<pathway from="kitchen" direction="north" to="hallway" />
</exampleOutput>

## Inventory

You currently believe you are carrying the following items:

<inventory>
INVENTORY
</inventory>

### Updating inventory

If you successfully picked up an item, dropped one, or used one, you can update your entire inventory by responding with a <set_inventory>...</set_inventory> tag like:

<exampleInput>
<description>You are in the kitchen.</description>
<action>inventory</action>
<description>You are carrying: lamp, apple</description>
</exampleInput>

<exampleOutput>
<set_inventory>
lamp
apple
</set_inventory>
</exampleOutput>

If you are unsure then use <action>inventory</action> to get a list of items you are carrying and update the inventory.

## Facts

You have determined the following facts that seem relevant to the game:

FACTS

### Adding facts

If you observe something that seems to be a message or clue from the game author to you, the player, then add it as a fact by responding with:

<fact name="Fact name" location="Location">
Information about the fact
</fact>

<exampleInput>
<description>You try to open the door, but it gives you an electric shock</description>
</exampleInput>

<exampleOutput>
<fact name="Electric shock" location="Antechamber">
The door gives you an electric shock. Maybe there's a way to turn off the power.
</fact>
</exampleOutput>

## Taking actions

Finally you will type into the game to play the game; for example this is like typing "get lamp" into the game:

<action>get lamp</action>

Begin by brainstorming three possible actions, like:

<exampleOutput>
<action_brainstorm>
1. get lamp (because it seems obviously useful)
2. search kitchen (because there's a clue)
3. wait (because I don't know what else to do
</action_brainstorm>
</exampleOutput>

After brainstorming the action, select the one MOST LIKELY TO ADVANCE THE GAME.

Do not repeat actions. You are playing the protagonist and should move the action forward and try new things to achieve your goals.

Typically actions should be between 1 and 3 words. Actions must be unambiguous and direct, typically <action>VERB</action> or <action>VERB NOUN</action> and are always plain text.

When moving simply use a direction like <action>north</action>

ALWAYS INCLUDE ONE <action>...</action> TAG IN YOUR RESPONSE, wrapping any text input you want to provide.
""".strip()


class LLMAgent(tales.Agent):

    def __init__(self, *args, **kwargs):
        self.llm = kwargs["llm"]
        self.model = llm.get_model(self.llm)
        self.token_counter = get_token_counter(self.model)
        self.allows_system_prompt = self.llm not in ["o1-mini", "o1-preview"]

        # Provide the API key, if one is needed and has been provided
        self.model.key = llm.get_key(
            kwargs.get("key"), kwargs["llm"], self.model.key_env_var
        ) or llm.get_key(None, self.model.needs_key, self.model.key_env_var)

        self.seed = kwargs["seed"]
        self.rng = np.random.RandomState(self.seed)

        self.history = []
        self.context_limit = kwargs["context_limit"]
        if self.context_limit is not None:
            assert self.context_limit > 0, "--context-limit must be greater than 0."

        self.act_temp = kwargs["act_temp"]
        self.conversation = kwargs["conversation"]
        self.goal_state = []
        self.pathways = []
        self.inventory = ""
        self.facts = []

    @property
    def uid(self):
        return (
            f"LLMAgent_{self.llm}"
            f"_s{self.seed}"
            f"_c{self.context_limit}"
            f"_t{self.act_temp}"
            f"_conv{self.conversation}"
        )

    @property
    def params(self):
        return {
            "agent_type": "zero-shot",
            "llm": self.llm,
            "seed": self.seed,
            "context_limit": self.context_limit,
            "act_temp": self.act_temp,
            "conversation": self.conversation,
        }

    @retry(
        retry=retry_if_exception(is_recoverable_error),
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(100),
    )
    def _llm_call_from_conversation(self, conversation, *args, **kwargs):
        response = conversation.prompt(*args, **kwargs)
        response.duration_ms()  # Forces the response to be computed.
        return response

    def _llm_call_from_messages(self, messages, *args, **kwargs):
        conversation = messages2conversation(self.model, messages)
        prompt = messages[-1]["content"]
        system = messages[0]["content"] if self.allows_system_prompt else None

        return self._llm_call_from_conversation(
            conversation, prompt=prompt, system=system, *args, **kwargs
        )

    def act(self, obs, reward, done, infos):
        messages = self.build_messages(f"{obs}\n> ")
        llm_kwargs = {
            "temperature": self.act_temp,
            # "max_tokens": 100,  # Text actions are short phrases.
            "seed": self.seed,
            "stream": False,
        }
        if self.llm in [
            "claude-3.5-haiku",
            "claude-3.5-sonnet",
            "claude-3.5-sonnet-latest",
        ]:
            # For these models, we cannot set the seed.
            llm_kwargs.pop("seed")

        if "gemini" in self.llm or "gemma" in self.llm:
            # For these models, we cannot set the seed and max_tokens has a different name.
            llm_kwargs.pop("seed")
            llm_kwargs["max_output_tokens"] = llm_kwargs.pop("max_tokens")

        response = self._llm_call_from_messages(messages, **llm_kwargs)

        text = response.text().strip()
        print("\n\nSTATE>>>>>>>>>>>>>>")
        print("Goals:")
        for g in self.goal_state:
            print(f"  {g['name']} for {g['requiredfor'] or "?"}: {repr(g['text'])}")
        print("Pathways:")
        for p in self.pathways:
            print(f"  {p[0]} {p[1]} -> {p[2]}")
        print("Inventory:")
        print("\n".join([f"  {i}" for i in self.inventory.splitlines()]))
        print("Facts:")
        for f in self.facts:
            print(f"  {f['name']} in {f['location']}: {repr(f['text'])}")
        print("INPUT>>>>>>>>>>>>>>")
        last_history = self.history[-1][1]["action"] if self.history else ""
        print(f"> {last_history}")
        print(obs)
        print("OUTPUT>>>>>>>>>>>>>")
        print(text)
        soup = BeautifulSoup(text, "html.parser")

        thought_process = soup.find("thought_process")
        if thought_process:
            thought_process = thought_process.string.strip() if thought_process.string else ""
        else:
            thought_process = None

        added_goals = []
        for goal in soup.find_all("goal"):
            self.goal_state = [g for g in self.goal_state if g["name"] != goal["name"]]
            new_goal = {
                "name": goal["name"],
                "text": goal.string.strip(),
                "requiredfor": goal.get("requiredfor", ""),
            }
            self.goal_state.append(new_goal)
            added_goals.append(new_goal)

        added_pathways = []
        for pathway in soup.find_all("pathway"):
            path = (pathway["from"], pathway["direction"], pathway["to"])
            self.pathways = [p for p in self.pathways if p[0] != path[0] or p[1] != path[1]]
            self.pathways.append(path)
            added_pathways.append(path)
        action = soup.find("action").string.strip()

        inventory = soup.find("set_inventory")
        if inventory:
            self.inventory = inventory.string.strip()

        added_facts = []
        for fact in soup.find_all("fact"):
            new_fact = {
                "name": fact["name"],
                "location": fact.get("location", ""),
                "text": fact.string.strip(),
            }
            self.facts.append(new_fact)
            added_facts.append(new_fact)

        self.history.append((f"{obs}", dict(action=action, thought_process=thought_process, added_goals=added_goals, added_pathways=added_pathways, inventory=inventory.string.strip() if inventory else None, added_facts=added_facts)))

        # Compute usage statistics
        stats = {
            "prompt": format_messages_to_markdown(messages),
            "response": response.text(),
            "nb_tokens": self.token_counter(messages=messages, text=response.text()),
        }

        return action, stats

    def build_messages(self, observation):
        goal_state = "\n".join([f"<goal name=\"{g['name']}\" requiredfor=\"{g.get('requiredfor', '')}\">{g['text']}</goal>" for g in self.goal_state]) or "No goals yet"
        pathways = "\n".join([f"<pathway from=\"{p[0]}\" direction=\"{p[1]}\" to=\"{p[2]}\" />" for p in self.pathways]) or "No pathways yet"
        facts = "\n".join([f"<fact name=\"{f['name']}\" location=\"{f['location']}\">{f['text']}</fact>" for f in self.facts]) or "No facts yet"
        system_prompt = SYSTEM_PROMPT.replace("GOAL_STATE", goal_state).replace("PATHWAYS", pathways).replace("INVENTORY", self.inventory).replace("FACTS", facts)

        messages = [{"role": "system", "content": system_prompt}]
        limit = self.context_limit or len(self.history) + 1

        for i, (obs, action) in enumerate(self.history[-limit:]):
            if len(self.history) >= limit and i == 0:
                # Add the current observation.
                obs = (
                    f"// History has been truncated to the last {limit} steps.\n...\n> "
                )

            if limit - i < 5:
                action_text = format_action(**action)
            else:
                action_text = format_action(action=action["action"])

            messages.append({"role": "user", "content": obs})
            messages.append({"role": "assistant", "content": action_text})

        messages.append({"role": "user", "content": observation})

        # Just in case, let's avoid having multiple messages from the same role.
        messages = merge_messages(messages)

        if not self.conversation:
            # Merge all messages into a single message except for the system.
            content = "".join([msg["content"] for msg in messages[1:]])
            messages = messages[:1] + [{"role": "user", "content": content}]

        if not self.allows_system_prompt:
            # Make sure the system prompt is added to the following message.
            messages.pop(0)
            messages[1]["content"] = f"{SYSTEM_PROMPT}\n\n{messages[1]['content']}"

        return messages


def build_argparser(parser=None):
    parser = parser or argparse.ArgumentParser()
    group = parser.add_argument_group("LLMAgent settings")

    group.add_argument(
        "--llm",
        default="gpt-4o-mini",
        help="LLM to be used for evaluation. Default: %(default)s",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=20241001,
        help="Seed for LLM (not all endpoints support this). Default: %(default)s",
    )
    group.add_argument(
        "--act-temp",
        type=float,
        default=0.0,
        help="Temperature for LLM when taking actions. Default: %(default)s",
    )
    group.add_argument(
        "--context-limit",
        type=int,
        help="Limit context for LLM (in conversation turns). Default: no limit.",
    )
    group.add_argument(
        "--conversation",
        required=True,
        action=argparse.BooleanOptionalAction,
        help="Enable conversation mode. Otherwise, use single prompt.",
    )

    return parser


register(
    name="memory",
    desc=(
        "This agent uses a LLM to decide which action to take in a zero-shot manner."
    ),
    klass=LLMAgent,
    add_arguments=build_argparser,
)

def format_action(action, thought_process=None, added_goals=None, added_pathways=None, inventory=None, added_facts=None):
    parts = []
    if thought_process:
        parts.append(f"<thoughtProcess>\n{thought_process}\n</thoughtProcess>")
    if added_goals:
        parts.append("\n".join([f"<goal name=\"{g['name']}\" requiredfor=\"{g.get('requiredfor', '')}\">\n{g['text']}\n</goal>" for g in added_goals]))
    if added_pathways:
        parts.append("\n".join([f"<pathway from=\"{p[0]}\" direction=\"{p[1]}\" to=\"{p[2]}\" />" for p in added_pathways]))
    if added_facts:
        parts.append("\n".join([f"<fact name=\"{f['name']}\" location=\"{f['location']}\">\n{f['text']}\n</fact>" for f in added_facts]))
    parts.append(f"<action>{action}</action>")
    return "\n\n".join(parts)
