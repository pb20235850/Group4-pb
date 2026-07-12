#!/usr/bin/env python3
"""MBPP evaluation with a lightweight Tree-of-Thoughts style search.

流程：
1. 为每道题生成多个中间思路 thought。
2. 每个 thought 生成代码并用 tests 评分。
3. 保留 top beam_width 条路径。
4. 对保留路径进行扩展/修正，再评分。
5. 选择测试通过数最高的最终代码。
"""
from __future__ import annotations

import argparse, ast, json, os, re, subprocess, sys, tempfile
from pathlib import Path
from typing import Any

ASSIGNMENT_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = ASSIGNMENT_ROOT.parent
DEFAULT_MBPP_DIR = WORK_ROOT / "mbpp"
DEFAULT_MODEL_PATH = ASSIGNMENT_ROOT / "dpo" / "outputs" / "qwen15_code_lora_grpo_v5"
DEFAULT_OUTPUT_DIR = ASSIGNMENT_ROOT / "dpo" / "outputs" / "mbpp_tot"
OFFICIAL_PROMPT_TEMPLATE = "You are an expert Python programmer, and here is your task: {prompt} Your code should pass these tests:\n\n{tests}\n[BEGIN]{code}\n[DONE]"
FENCED_CODE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--mbpp_dir", type=Path, default=DEFAULT_MBPP_DIR); p.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH); p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--config", choices=("sanitized","full"), default="sanitized"); p.add_argument("--split", default="test")
    p.add_argument("--prompt_mode", choices=("zero_shot","one_shot","three_shot"), default="zero_shot"); p.add_argument("--prompt_task_ids", default="2,3,4")
    p.add_argument("--limit", type=int, default=0); p.add_argument("--start_index", type=int, default=0); p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--num_thoughts", type=int, default=6); p.add_argument("--branches_per_thought", type=int, default=2); p.add_argument("--beam_width", type=int, default=4); p.add_argument("--expand_rounds", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.7); p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--test_timeout", type=float, default=5.0); p.add_argument("--memory_mb", type=int, default=1024); p.add_argument("--include_challenge_tests", action="store_true")
    p.add_argument("--skip_generation", action="store_true"); p.add_argument("--use_reference_code", action="store_true"); p.add_argument("--predictions", type=Path, default=None); p.add_argument("--trust_remote_code", action="store_true", default=True)
    return p.parse_args()

def normalize_text(v): return str(v or "").replace("\r\n","\n").replace("\r","\n").strip()
def listify(v):
    if v is None: return []
    if hasattr(v,"tolist"): v=v.tolist()
    if isinstance(v, tuple): v=list(v)
    if isinstance(v, list): return [str(x).strip() for x in v if str(x).strip()]
    return [str(v).strip()] if str(v).strip() else []
def read_parquet_rows(path):
    try:
        import pyarrow.parquet as pq; return pq.read_table(path).to_pylist()
    except Exception:
        import pandas as pd; return pd.read_parquet(path).to_dict("records")
def load_split(mbpp_dir, config, split):
    path=mbpp_dir/config/f"{split}-00000-of-00001.parquet"
    if not path.exists(): raise FileNotFoundError(path)
    return read_parquet_rows(path)
def row_prompt(row, config): return normalize_text(row.get("prompt") or row.get("text")) if config=="sanitized" else normalize_text(row.get("text") or row.get("prompt"))
def row_setup(row, config): return "\n".join(listify(row.get("test_imports"))) if config=="sanitized" else normalize_text(row.get("test_setup_code"))
def row_tests(row, include_challenge):
    t=listify(row.get("test_list"));
    if include_challenge: t.extend(listify(row.get("challenge_test_list")))
    return t
def official_block(prompt, tests, code=""): return OFFICIAL_PROMPT_TEMPLATE.format(prompt=prompt, tests="\n".join(tests), code=code)
def build_prompt_prefix(prompt_rows, config, prompt_mode, ids):
    if prompt_mode=="zero_shot": return ""
    if prompt_mode=="one_shot": ids=ids[:1]
    by_id={int(r["task_id"]):r for r in prompt_rows}; sel=[by_id[i] for i in ids if i in by_id]
    if len(sel)<len(ids): sel=prompt_rows[:len(ids)]
    return "\n\n".join(official_block(row_prompt(r,config), row_tests(r,False), normalize_text(r.get("code"))) for r in sel)
def build_prompt(prefix,row,config,include_challenge):
    target=official_block(row_prompt(row,config), row_tests(row,include_challenge), "").rsplit("[DONE]",1)[0]
    return target if not prefix else f"{prefix}\n\n{target}"
def apply_chat_template(tok,prompt):
    if getattr(tok,"chat_template",None): return tok.apply_chat_template([{"role":"user","content":prompt}], tokenize=False, add_generation_prompt=True)
    return prompt
def load_model(path, trust):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok=AutoTokenizer.from_pretrained(path, trust_remote_code=trust); tok.padding_side="left"
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    model=AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None, trust_remote_code=trust)
    if not torch.cuda.is_available(): model.to("cpu")
    model.eval(); return tok,model,torch
def extract_candidate_code(text):
    text=normalize_text(text)
    if "[BEGIN]" in text: text=text.rsplit("[BEGIN]",1)[-1]
    if "[DONE]" in text: text=text.split("[DONE]",1)[0]
    fenced=FENCED_CODE_RE.findall(text)
    if fenced: return normalize_text(fenced[-1])
    raw=text.splitlines(); first=0
    for i,l in enumerate(raw):
        if l.lstrip().startswith(("import ","from ","def ","class ","@")): first=i; break
    lines=[]
    for l in raw[first:]:
        s=l.strip()
        if s in {"[DONE]","DONE"}: break
        if s.startswith("```"): continue
        lines.append(l)
    return normalize_text("\n".join(lines))
def syntax_ok(code):
    if not normalize_text(code): return False
    try:
        import warnings
        with warnings.catch_warnings(): warnings.simplefilter("ignore", SyntaxWarning); ast.parse(code)
        return True
    except SyntaxError: return False
def limit_resources(memory_mb, timeout):
    try:
        import resource
        m=memory_mb*1024*1024; c=max(1,int(timeout)+1)
        resource.setrlimit(resource.RLIMIT_AS,(m,m)); resource.setrlimit(resource.RLIMIT_CPU,(c,c)); resource.setrlimit(resource.RLIMIT_FSIZE,(10*1024*1024,10*1024*1024))
    except Exception: return
def run_one_assert(code, setup, test, timeout, memory_mb):
    runner="\n\n".join(x for x in ["import warnings\nwarnings.filterwarnings('ignore', category=SyntaxWarning)","import faulthandler\nfaulthandler.enable()", setup, code, test] if normalize_text(x))
    with tempfile.TemporaryDirectory(prefix="mbpp_eval_") as td:
        path=Path(td)/"candidate_test.py"; path.write_text(runner+"\n", encoding="utf-8"); env=os.environ.copy(); env["HOME"]=td
        try:
            res=subprocess.run([sys.executable,str(path)], cwd=td, env=env, text=True, capture_output=True, timeout=timeout, preexec_fn=lambda: limit_resources(memory_mb,timeout) if os.name=="posix" else None)
        except subprocess.TimeoutExpired as e: return {"passed":False,"error_type":"timeout","stdout":"","stderr":str(e)}
    return {"passed":res.returncode==0,"error_type":"" if res.returncode==0 else "runtime_error","stdout":res.stdout[-1000:],"stderr":res.stderr[-2000:]}
def score_code(code,row,config,include_challenge,timeout,memory_mb):
    tests=row_tests(row,include_challenge); setup=row_setup(row,config); per=[run_one_assert(code,setup,t,timeout,memory_mb) for t in tests]; p=sum(1 for x in per if x["passed"]); total=len(tests)
    return {"passed_tests":p,"total_tests":total,"all_pass":p==total and total>0,"syntax_ok":syntax_ok(code),"test_results":per}
def summarize_failures(score, tests, max_items=2):
    out=[]
    for test,res in zip(tests, score.get("test_results",[])):
        if res.get("passed"): continue
        err=normalize_text(res.get("stderr") or res.get("stdout") or res.get("error_type")); out.append(f"Failed test: {test}\nError: {err[-600:]}")
        if len(out)>=max_items: break
    return "\n\n".join(out) if out else "Some tests failed."
def gen_text(tok,model,torch,prompt,max_new,do_sample,temp,top_p):
    rendered=apply_chat_template(tok,prompt); inp=tok(rendered, return_tensors="pt", truncation=True); inp={k:v.to(model.device) for k,v in inp.items()}; w=inp["input_ids"].shape[1]
    kw=dict(max_new_tokens=max_new, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    if do_sample: kw.update(do_sample=True, temperature=temp, top_p=top_p)
    else: kw.update(do_sample=False, num_beams=1)
    with torch.no_grad(): out=model.generate(**inp, **kw)
    return tok.decode(out[0][w:], skip_special_tokens=True)
def thought_prompt(row,config,include_challenge):
    return f"""You are solving a Python programming task.
Task: {row_prompt(row,config)}
Tests:
{chr(10).join(row_tests(row,include_challenge))}

Generate one concise high-level solution idea. Do not write code yet.
Idea:"""
def code_from_thought_prompt(row,config,include_challenge,thought):
    return f"""You are an expert Python programmer.
Task: {row_prompt(row,config)}
Tests:
{chr(10).join(row_tests(row,include_challenge))}

Use this solution idea:
{thought}

Return only Python code between [BEGIN] and [DONE].
[BEGIN]
"""
def expand_prompt(row,config,include_challenge,thought,code,fail):
    return f"""You are improving a Python solution.
Task: {row_prompt(row,config)}
Tests:
{chr(10).join(row_tests(row,include_challenge))}

Current idea:
{thought}

Current code:
[BEGIN]
{code}
[DONE]

Failure information:
{fail}

Revise the idea if needed and return corrected Python code between [BEGIN] and [DONE].
[BEGIN]
"""
def record(row,config,include_challenge,completion,code,thought,path_score,summary):
    return {"task_id":int(row["task_id"]),"prompt":row_prompt(row,config),"completion":completion,"code":code,"reference_code":normalize_text(row.get("code")),"tests":row_tests(row,False),"inference_method":"tree_of_thoughts","thought":thought,"path_score":path_score,"tree_summary":summary}
def generate_completions(rows,prompts,args):
    tok,model,torch=load_model(args.model_path,args.trust_remote_code); gens=[]
    for idx,row in enumerate(rows,1):
        paths=[]
        for _ in range(args.num_thoughts):
            th=gen_text(tok,model,torch,thought_prompt(row,args.config,args.include_challenge_tests),160,True,args.temperature,args.top_p)
            for _b in range(args.branches_per_thought):
                comp=gen_text(tok,model,torch,code_from_thought_prompt(row,args.config,args.include_challenge_tests,th),args.max_new_tokens,True,args.temperature,args.top_p)
                code=extract_candidate_code(comp); sc=score_code(code,row,args.config,args.include_challenge_tests,args.test_timeout,args.memory_mb)
                paths.append({"thought":th,"completion":comp,"code":code,"score":sc})
        paths.sort(key=lambda x:(x["score"]["passed_tests"], x["score"]["syntax_ok"]), reverse=True)
        beam=paths[:args.beam_width]
        for _round in range(args.expand_rounds):
            expanded=[]
            for p in beam:
                if p["score"]["all_pass"]: expanded.append(p); continue
                fail=summarize_failures(p["score"], row_tests(row,args.include_challenge_tests))
                comp=gen_text(tok,model,torch,expand_prompt(row,args.config,args.include_challenge_tests,p["thought"],p["code"],fail),args.max_new_tokens,True,max(0.2,args.temperature-0.2),args.top_p)
                code=extract_candidate_code(comp); sc=score_code(code,row,args.config,args.include_challenge_tests,args.test_timeout,args.memory_mb)
                expanded.append(p); expanded.append({"thought":p["thought"],"completion":comp,"code":code,"score":sc,"expanded_from":p["score"]})
            expanded.sort(key=lambda x:(x["score"]["passed_tests"], x["score"]["syntax_ok"]), reverse=True)
            beam=expanded[:args.beam_width]
        best=beam[0]
        summary=[{"passed_tests":p["score"]["passed_tests"],"total_tests":p["score"]["total_tests"],"all_pass":p["score"]["all_pass"],"syntax_ok":p["score"]["syntax_ok"]} for p in beam]
        gens.append(record(row,args.config,args.include_challenge_tests,best["completion"],best["code"],best["thought"],best["score"],summary))
        print(f"Generated {len(gens)} / {len(rows)}", flush=True)
    return gens
def evaluate_generation(gen,row,config,include_challenge,timeout,memory_mb):
    code=normalize_text(gen.get("code")); sc=score_code(code,row,config,include_challenge,timeout,memory_mb)
    return {"task_id":int(row["task_id"]),"prompt":row_prompt(row,config),"code":code,"reference_code":normalize_text(row.get("code")),"syntax_ok":sc["syntax_ok"],"passed":sc["all_pass"],"passed_tests":sc["passed_tests"],"total_tests":sc["total_tests"],"test_results":sc["test_results"],"completion":gen.get("completion",""),"thought":gen.get("thought",""),"tree_summary":gen.get("tree_summary",[])}
def save_json(path,data): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n", encoding="utf-8")
def save_jsonl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
def read_jsonl(path):
    with path.open(encoding="utf-8") as f: return [json.loads(l) for l in f if l.strip()]
def main():
    args=parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True); pred=args.predictions or (args.output_dir/"mbpp_generations.jsonl")
    ids=[int(x) for x in args.prompt_task_ids.split(",") if x.strip()]; prompt_rows=load_split(args.mbpp_dir,args.config,"prompt"); eval_rows=load_split(args.mbpp_dir,args.config,args.split)[args.start_index:]
    if args.limit>0: eval_rows=eval_rows[:args.limit]
    prefix=build_prompt_prefix(prompt_rows,args.config,args.prompt_mode,ids); prompts=[build_prompt(prefix,r,args.config,args.include_challenge_tests) for r in eval_rows]
    if args.use_reference_code:
        gens=[record(r,args.config,args.include_challenge_tests,normalize_text(r.get("code")),normalize_text(r.get("code")),"reference",{},[]) for r in eval_rows]; save_jsonl(pred,gens)
    elif args.skip_generation: gens=read_jsonl(pred)
    else: gens=generate_completions(eval_rows,prompts,args); save_jsonl(pred,gens)
    by={int(g["task_id"]):g for g in gens}; cases=[evaluate_generation(by[int(r["task_id"])],r,args.config,args.include_challenge_tests,args.test_timeout,args.memory_mb) for r in eval_rows if int(r["task_id"]) in by]
    total=len(cases); passed=sum(c["passed"] for c in cases); syntax=sum(c["syntax_ok"] for c in cases); total_tests=sum(c["total_tests"] for c in cases); passed_tests=sum(c["passed_tests"] for c in cases)
    metrics={"benchmark":"MBPP","config":args.config,"split":args.split,"model_path":str(args.model_path),"predictions":str(pred),"num_tasks":total,"pass_at_1":passed/total if total else 0.0,"syntax_pass_rate":syntax/total if total else 0.0,"avg_test_pass_rate":passed_tests/total_tests if total_tests else 0.0,"passed_tasks":passed,"total_tests":total_tests,"passed_tests":passed_tests,"inference_method":"tree_of_thoughts","num_thoughts":args.num_thoughts,"branches_per_thought":args.branches_per_thought,"beam_width":args.beam_width,"expand_rounds":args.expand_rounds,"temperature":args.temperature,"top_p":args.top_p,"prompt":{"source":"Google Research MBPP README","template":OFFICIAL_PROMPT_TEMPLATE,"mode":args.prompt_mode,"example_task_ids":ids if args.prompt_mode=="three_shot" else ids[:1] if args.prompt_mode=="one_shot" else []},"execution_note":"Generated Python is executed in a temporary subprocess with timeout and basic resource limits."}
    save_json(args.output_dir/"mbpp_metrics.json",metrics); save_jsonl(args.output_dir/"mbpp_cases.jsonl",cases); print(json.dumps(metrics,ensure_ascii=False,indent=2)); print(f"Wrote generations to {pred}"); print(f"Wrote metrics to {args.output_dir/'mbpp_metrics.json'}"); print(f"Wrote cases to {args.output_dir/'mbpp_cases.jsonl'}")
if __name__=="__main__": main()
