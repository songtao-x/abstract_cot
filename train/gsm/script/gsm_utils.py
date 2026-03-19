from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

GSM_DC_DIR = Path(__file__).resolve().parents[1] / "gsm_dc"
GSM_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
CONDITION_ORDER = ("light", "medium", "hard")
CONDITION_ALIASES = {
    "condition_1": "light",
    "condition_2": "medium",
    "condition_3": "hard",
    "1": "light",
    "2": "medium",
    "3": "hard",
}
DEFAULT_GSM_HF_DATASET = "YMinglai/GSM-DC-Dataset-Sample"
DEFAULT_GSM_HF_DATA_FILE = "all_problems.json"
DEFAULT_GSM_SAMPLE_SOURCE_FILE = GSM_DATA_DIR / "gsm_dc_sample_all.json"
DEFAULT_GSM_TRAIN_FILE = GSM_DATA_DIR / "gsm_sample_train.jsonl"
DEFAULT_GSM_EVAL_FILE = GSM_DATA_DIR / "gsm_sample_valid.jsonl"
DEFAULT_GSM_TEST_FILE = GSM_DATA_DIR / "gsm_sample_test.jsonl"


def ensure_gsmdc_path() -> None:
    path = str(GSM_DC_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)


def _load_gsm_components() -> tuple[Any, Any, Any, Any]:
    ensure_gsmdc_path()
    from data_gen.prototype.id_gen import IdGen_PT
    from math_gen.problem_gen import Problem
    from tools.irr_tools_test import true_correct
    from tools.tools import fix_seed, tokenizer

    return IdGen_PT, Problem, true_correct, (fix_seed, tokenizer)


def seed_gsm_generation(seed: int) -> None:
    import numpy as np
    import torch

    _, _, _, (fix_seed, _) = _load_gsm_components()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    fix_seed(seed)


def parse_param_str(raw_value: str) -> tuple[int, int, int, int]:
    inside = raw_value.strip("()")
    parts = inside.split(",")
    if len(parts) != 4:
        raise ValueError(f"param tuple must have length 4, got {raw_value}")
    return tuple(int(x.strip()) for x in parts)


def build_name2param_dict(problem: Any) -> None:
    problem.name2param_dict = {}
    for param in problem.all_param:
        level, i, j, k = param
        if level == -1:
            param_name = "RNG"
        elif level == 0:
            name0 = problem.N[i][j]
            name1 = problem.N[i + 1][k]
            param_name = f"{name0}{problem.args['dot']}{name1}"
        elif level == 1:
            name0 = problem.N[i][j]
            cat = problem.ln[k]
            param_name = f"{name0}{problem.args['dot']}{cat}"
        else:
            param_name = f"UnsupportedParam{param}"
        problem.name2param_dict[param_name] = param


def problem_to_json_dict(problem: Any) -> dict[str, Any]:
    question_index = None
    if getattr(problem, "ques_idx", None) is not None:
        question_index = [int(x) for x in problem.ques_idx]

    final_answer = getattr(problem, "ans", None)
    if final_answer is not None:
        final_answer = int(final_answer)

    node_data: dict[str, dict[str, Any]] = {}
    for (layer, idx), data in problem.graph.nodes(data=True):
        label = str(problem.N[layer][idx])
        node_data[label] = {
            "node": f"({int(layer)}, {int(idx)})",
            "unique": bool(data.get("unique", False)),
        }

    edges = [[str(u), str(v)] for u, v in problem.graph.edges()]
    template_edges = [[str(u), str(v)] for u, v in problem.template.edges()]

    whole_template_edges: list[list[str]] = []
    if hasattr(problem, "whole_template"):
        whole_template_edges = [[str(u), str(v)] for u, v in problem.whole_template.edges()]

    all_param = [str(param) for param in getattr(problem, "all_param", [])]

    topo_list: list[dict[str, str]] = []
    for param in getattr(problem, "topological_order", []) or []:
        level, i, j, k = param
        if level == -1:
            description = "RNG"
        elif level == 0:
            name0 = problem.N[i][j]
            name1 = problem.N[i + 1][k]
            description = f"{name0}{problem.args['dot']}{name1}"
        elif level == 1:
            name0 = problem.N[i][j]
            cat = problem.ln[k]
            description = f"{name0}{problem.args['dot']}{cat}"
        else:
            description = f"UnsupportedParam{param}"
        topo_list.append({"param": str(param), "description": description})

    return {
        "problem_info": {
            "d": int(problem.d),
            "w0": int(problem.w0),
            "w1": int(problem.w1),
            "e": int(problem.e),
            "p": float(problem.p),
            "final_answer": final_answer,
            "question_index": question_index,
        },
        "node_data": node_data,
        "edges": edges,
        "template_edges": template_edges,
        "whole_template_edges": whole_template_edges,
        "ln": [str(x) for x in problem.ln],
        "all_param": all_param,
        "problem_text": list(getattr(problem, "problem", []) or []),
        "solution_text": list(getattr(problem, "solution", []) or []),
        "topological_order": topo_list,
    }


def rebuild_problem_from_dict(data_dict: dict[str, Any]) -> Any:
    import networkx as nx
    import numpy as np

    _, Problem, _, _ = _load_gsm_components()

    info = data_dict["problem_info"]
    final_answer = info["final_answer"]
    if final_answer is not None:
        final_answer = int(final_answer)

    args = {
        "rand_perm": "none",
        "define_var": True,
        "define_detail": True,
        "inter_var": True,
        "name_omit": False,
        "cal_omit": False,
        "dot": "'s ",
        "symbol_method": "rand",
        "sol_sort": False,
        "perm": False,
    }
    problem = Problem(
        int(info["d"]),
        int(info["w0"]),
        int(info["w1"]),
        int(info["e"]),
        float(info["p"]),
        args=args,
    )

    node_data = data_dict["node_data"]
    layer_counts: dict[int, int] = {}
    for value in node_data.values():
        layer_str, idx_str = value["node"].strip("()").split(",")
        layer = int(layer_str)
        idx = int(idx_str)
        layer_counts[layer] = max(layer_counts.get(layer, 0), idx + 1)

    for i in range(problem.d):
        problem.l[i] = layer_counts.get(i, 0)

    problem.graph = nx.DiGraph()
    for i in range(problem.d):
        for j in range(problem.l[i]):
            problem.graph.add_node((i, j), unique=False)

    problem.N = [[""] * problem.l[i] for i in range(problem.d)]

    for key, value in node_data.items():
        layer_str, idx_str = value["node"].strip("()").split(",")
        layer = int(layer_str)
        idx = int(idx_str)
        problem.N[layer][idx] = key
        problem.graph.nodes[(layer, idx)]["unique"] = bool(value["unique"])
        if bool(value["unique"]):
            problem.unique.append((layer, idx))

    for u_str, v_str in data_dict["edges"]:
        u_layer, u_idx = [int(x.strip()) for x in u_str.strip("()").split(",")]
        v_layer, v_idx = [int(x.strip()) for x in v_str.strip("()").split(",")]
        problem.graph.add_edge((u_layer, u_idx), (v_layer, v_idx), chosen=False)

    problem.G = []
    for i in range(problem.d - 1):
        matrix = np.zeros((problem.l[i], problem.l[i + 1]), dtype=bool)
        for j in range(problem.l[i]):
            for k in range(problem.l[i + 1]):
                if problem.graph.has_edge((i, j), (i + 1, k)):
                    matrix[j, k] = True
        problem.G.append(matrix)

    problem.ln = [str(x) for x in data_dict.get("ln", [])]

    problem.template = nx.DiGraph()
    template_edges = data_dict["template_edges"]
    template_nodes = set()
    for u_str, v_str in template_edges:
        template_nodes.add(u_str)
        template_nodes.add(v_str)
    for raw_node in template_nodes:
        problem.template.add_node(parse_param_str(raw_node))
    for u_str, v_str in template_edges:
        problem.template.add_edge(parse_param_str(u_str), parse_param_str(v_str))

    problem.whole_template = nx.DiGraph()
    whole_template_edges = data_dict.get("whole_template_edges", [])
    whole_template_nodes = set()
    for u_str, v_str in whole_template_edges:
        whole_template_nodes.add(u_str)
        whole_template_nodes.add(v_str)
    for raw_node in whole_template_nodes:
        problem.whole_template.add_node(parse_param_str(raw_node))
    for u_str, v_str in whole_template_edges:
        problem.whole_template.add_edge(parse_param_str(u_str), parse_param_str(v_str))

    problem.all_param = [parse_param_str(raw_param) for raw_param in data_dict.get("all_param", [])]
    problem.ans = final_answer if final_answer is not None else 0

    question_index = info["question_index"]
    if question_index is not None:
        problem.ques_idx = tuple(question_index)

    topo_list = data_dict.get("topological_order", [])
    if topo_list:
        problem.topological_order = [
            parse_param_str(item["param"]) if isinstance(item, dict) else parse_param_str(item)
            for item in topo_list
        ]
    else:
        problem.topological_order = []

    n_op = 0
    for param in problem.topological_order:
        num_pre = len(list(problem.template.predecessors(param)))
        n_op += 1 if num_pre <= 2 else num_pre - 1
    problem.n_op = n_op

    build_name2param_dict(problem)
    problem.problem = list(data_dict.get("problem_text", []) or [])
    problem.solution = list(data_dict.get("solution_text", []) or [])
    return problem


def problem_to_json_string(problem: Any) -> str:
    return json.dumps(problem_to_json_dict(problem), ensure_ascii=True, separators=(",", ":"))


def _compact_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def normalize_problem_text_value(problem_text: Any) -> str:
    if isinstance(problem_text, (list, tuple)):
        parts = [str(item).strip().rstrip(".") for item in problem_text if str(item).strip()]
        text = ". ".join(parts).strip()
    else:
        text = str(problem_text).strip()
    if text and not text.endswith((".", "?", "!")):
        text = f"{text}."
    return text


def normalize_solution_text_value(solution_text: Any) -> str:
    if isinstance(solution_text, (list, tuple)):
        return " ".join(str(item).strip() for item in solution_text if str(item).strip()).strip()
    return str(solution_text).strip()


def normalize_problem_text(problem: Any) -> str:
    return normalize_problem_text_value(getattr(problem, "problem", []))


def derive_sample_metadata(row_index: int) -> tuple[int, str]:
    if row_index < 0:
        raise ValueError(f"row_index must be non-negative, got {row_index}")
    op = 2 + (row_index // 300)
    condition = CONDITION_ORDER[(row_index % 300) // 100]
    return op, condition


def normalize_condition_value(raw_value: Any, default: str) -> str:
    normalized = str(raw_value).strip().lower()
    if not normalized:
        return default
    return CONDITION_ALIASES.get(normalized, normalized)


def _load_problem_payload(raw_row: dict[str, Any]) -> dict[str, Any] | None:
    if "problem_info" in raw_row:
        return raw_row

    problem_json = raw_row.get("problem_json")
    if isinstance(problem_json, dict):
        if "problem_info" in problem_json:
            return problem_json
        return None
    if isinstance(problem_json, str):
        try:
            payload = json.loads(problem_json)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict) and "problem_info" in payload:
            return payload
    return None


def build_export_row_from_sample(raw_row: dict[str, Any], row_index: int) -> dict[str, Any]:
    derived_op, derived_condition = derive_sample_metadata(row_index)

    condition_value = raw_row.get("condition", raw_row.get("noise_level", derived_condition))
    condition = normalize_condition_value(condition_value, derived_condition)

    op_value = raw_row.get("op", derived_op)
    try:
        op = int(op_value)
    except (TypeError, ValueError):
        op = derived_op

    payload = _load_problem_payload(raw_row)
    if payload is not None:
        problem_text = raw_row.get("problem_text", payload.get("problem_text", ""))
        reference_solution = raw_row.get("reference_solution", payload.get("solution_text", ""))
        final_answer = raw_row.get("final_answer", payload.get("problem_info", {}).get("final_answer"))
        problem_json = _compact_json_dumps(payload)
    else:
        if "problem_json" not in raw_row:
            problem_json = _compact_json_dumps(raw_row)
        elif isinstance(raw_row["problem_json"], str):
            try:
                problem_json = _compact_json_dumps(json.loads(raw_row["problem_json"]))
            except json.JSONDecodeError:
                problem_json = raw_row["problem_json"].strip()
        else:
            problem_json = _compact_json_dumps(raw_row["problem_json"])
        problem_text = raw_row.get("problem_text", "")
        reference_solution = raw_row.get("reference_solution", raw_row.get("solution_text", ""))
        final_answer = raw_row.get("final_answer")

    if final_answer is None:
        raise ValueError(f"Sample row {row_index} is missing final_answer")

    return {
        "problem_text": normalize_problem_text_value(problem_text),
        "problem_json": problem_json,
        "final_answer": int(final_answer),
        "reference_solution": normalize_solution_text_value(reference_solution),
        "condition": condition,
        "op": op,
    }


def build_export_row(problem: Any, condition: str, op: int) -> dict[str, Any]:
    return {
        "problem_text": normalize_problem_text(problem),
        "problem_json": problem_to_json_string(problem),
        "final_answer": int(problem.ans),
        "reference_solution": " ".join(problem.solution).strip(),
        "condition": condition,
        "op": int(op),
    }


def generate_gsm_problem(op: int, condition: str) -> Any:
    IdGen_PT, _, _, _ = _load_gsm_components()
    id_gen = IdGen_PT(
        style="light",
        op_style="light",
        op=op,
        perm_level=5,
        detail_level=0,
        noise_level=condition,
    )
    id_gen.gen_prob([i for i in range(5)], p_format="pq")
    return id_gen.problem


def load_problem_from_json_blob(problem_json: str | dict[str, Any]) -> Any:
    if isinstance(problem_json, str):
        payload = json.loads(problem_json)
    else:
        payload = problem_json
    return rebuild_problem_from_dict(payload)


def extract_first_integer(text: str) -> int | None:
    digits: list[str] = []
    for char in text:
        if char.isdigit():
            digits.append(char)
        elif digits:
            break
    if not digits:
        return None
    return int("".join(digits))


def score_gsm_reasoning(think_text: str, problem: Any) -> tuple[bool, bool]:
    _, _, true_correct, _ = _load_gsm_components()
    di_correct, correct, _, _ = true_correct(think_text, problem)
    return bool(correct), bool(di_correct)
