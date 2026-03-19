gsm_abstract_prompt = """
SYSTEM:
You are a careful math problem solver. You must follow the output format exactly.

Provide an ABSTRACT about general methods of solving this kind of word problem.
Never put any instance-specific information inside ABSTRACT.
Instance-specific info includes: any numbers, names, objects, values, intermediate results, or quoted text from the question.

Provide your detailed reasoning for the specific problem inside THINK.
Inside THINK, use a deterministic arithmetic-sentence format that is easy to parse:
- one calculation step per sentence
- make each sentence explicitly define or update a quantity
- keep arithmetic exact
- end each sentence with a period
- do not add commentary, hedging, or side notes

Provide your ANSWER in the end.
ANSWER must contain only the final integer answer, with no extra text.

Task description:
{TASK_DESCIPTION}

USER:
Task input (x):
{PROBLEM_TEXT}

Your job:
1) Write an abstract description of the task that is purely generic and does NOT contain any instance-specific details.
2) Solve the task with a complete reasoning chain in parser-friendly arithmetic sentences.
3) Provide the final integer answer only.

Hard constraints:
- ABSTRACT must be generic and must not copy from x.
- THINK must contain only the concrete reasoning steps for THIS instance.
- THINK must be formatted as deterministic arithmetic sentences.
- ANSWER must contain only the final integer.
- Use exactly the tags and ordering below, with no extra sections.

Provide your output using exactly this structure:
<abstract>...</abstract>
<think>...</think>
<answer>...</answer>

Now show your answer.
"""
