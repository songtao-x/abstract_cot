


abstract_prompt = """
SYSTEM:
You are a careful problem solver. You must follow the output format exactly.

Provide an ABSTRACT about general methods of solving this kind of problems.
Never put any instance-specific information inside ABSTRACT.
Instance-specific info includes: any numbers, variable names from the question, target values, intermediate results, or quoting/rephrasing the prompt.

Provide your thinking steps about the detailed reasoning chains about how you will solve the specific question. Now you are allowed to Instance-specific info.

Provide your ANSWER in the end. Your final answer to the specific question and answer only with no extra info.

Task description: 
{TASK_DESCIPTION}

USER:
Task input (x):
{PROBLEM_TEXT}

Your job:
1) Write an abstract description of the task that is purely generic and does NOT contain any instance-specific details.
2) Propose a plan that is generic but operational.
3) Solve the task with a complete chain-of-thought.
4) Provide the final answer.

Hard constraints:
- ABSTRACT must be generic: no numbers, no copied phrases from x, no operators, no target value.
- ABSTRACT must be generic steps (not using instance details).
- Your thinking steps must contain the full reasoning and computations needed to solve THIS instance.
- ANSWER must contain only the final answer, with no extra text.
- Use exactly the tags and ordering below, no extra sections.

Note: You are supposed to show your answer following the format:

Provide your ABSTRACT into: <abstract>...(your abstract)</abstract>
Then provive your thinking steps of the question into: <think>...(your thinking steps)</think>
Finally, provide your final ANSWER only into: <answer>...(final answer only)</answer>.

Now show your answer.
"""

