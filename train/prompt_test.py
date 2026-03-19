from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from prompt_template import abstract_prompt

DEFAULT_TASK_DESCRIPTION = (
    "Countdown arithmetic task: use the provided numbers to reach the target with +, -, *, /. "
    "Use each provided number exactly once and provide a valid solution in the required tag format."
)
DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "cd4_test.jsonl"


def parse_countdown_input(raw_input: str) -> tuple[list[int], int]:
    values = [int(x.strip()) for x in raw_input.split(",") if x.strip()]
    if len(values) < 2:
        raise ValueError(f"Invalid input: {raw_input}")
    return values[:-1], values[-1]


def build_problem_text(numbers: list[int], target: int) -> str:
    return (
        f"Numbers: {', '.join(str(x) for x in numbers)}\\n"
        f"Target: {target}\\n"
        "Find a valid expression that reaches the target using each listed number exactly once."
    )


def load_sample_from_jsonl(data_file: Path) -> dict[str, str]:
    ds = []
    with data_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            
            item = json.loads(line)
            ds.append({"pid": i, "input": item["input"], "output": item.get("output", "")})
    
    
    return ds


def make_prompt(raw_input: str, task_description: str) -> tuple[str, int]:
    numbers, target = parse_countdown_input(raw_input)
    prompt = abstract_prompt.format(
        TASK_DESCIPTION=task_description,
        PROBLEM_TEXT=build_problem_text(numbers, target),
    )
    return prompt, target


def generate_with_vllm(ds, args: argparse.Namespace) -> str:
    if args.target_device is not None:
        os.environ["VLLM_TARGET_DEVICE"] = args.target_device
    
    if isinstance(ds, str):
        ds = [ds]

    from vllm import LLM, SamplingParams

    try:
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=4096,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
    except RuntimeError as e:
        msg = str(e)
        if "Failed to infer device type" in msg or "Device string must not be empty" in msg:
            raise RuntimeError(
                "vLLM failed to infer device type. Run on a node with GPU support or set "
                "`--target_device`/`VLLM_TARGET_DEVICE` appropriately."
            ) from e
        raise
    
    print(f'\nLoading model done ... \n')

    if "Qwen" in args.model:
        enable_thinking = False
    else:
        enable_thinking = True
    
    print(f"\n enable_thinking is: {enable_thinking}...\n")
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    responses = []
    for i, ex in enumerate(ds):
        prompt = ex['prompt']
        label = ex['label']
        target = ex['target']
        outputs = llm.generate(["/no_think\n" + prompt], 
                                sampling_params,
                                )[0].outputs[0].text.strip()
        # responses.append(outputs)
        responses.append({'Prompt': prompt, 'target': target, 'label': label, 'Responses': outputs})
        print(f'\n\nPrompt: {prompt}, target: {target}, label: {label}\nResponses: {responses}')
    with open('responses_nothink.json', 'w') as f:
        json.dump(responses, f, indent=4, ensure_ascii=True)
    return responses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple prompt test for abstract countdown task with vLLM generation.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--input", type=str, default=None, help="Countdown sample like '90,11,37,95,55'.")
    parser.add_argument("--task_description", type=str, default=DEFAULT_TASK_DESCRIPTION)
    parser.add_argument("--data_file", type=str, default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--index", type=int, default=0)

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--target_device", type=str, default=None, help="Optional VLLM_TARGET_DEVICE override (e.g., cuda/cpu/xpu/tpu).")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--print_prompt", action="store_true")
    return parser.parse_args()


def data_process(file):

    ds = load_sample_from_jsonl(file)
    processed = []
    for ex in ds:
        
        raw_input = ex["input"]
        label = ex["output"]
        pid = ex['pid']

        prompt, target = make_prompt(raw_input=raw_input, task_description=DEFAULT_TASK_DESCRIPTION)
        ex_ = {"pid": pid, "prompt": prompt, "target": target, "label": label}
        processed.append(ex_)
    
    return processed





def main() -> None:
    args = parse_args()
    raw_input = args.input
    label = ""
    
    ds = data_process(Path(args.data_file))

    ds = ds[:10]
    generation = generate_with_vllm(ds, args)
    print("----- generation begin -----")
    print(generation)
    print("----- generation end -----")


if __name__ == "__main__":
    main()
