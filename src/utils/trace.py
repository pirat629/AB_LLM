def show_trace(run):
    for m in run.messages[1:]:
        calls = "; ".join(f"{c["function"]["name"]}{c["function"]["arguments"]}" for c in m.get("tool_calls", []))
        text = " ".join((m.get("content") or "").split())[:180]
        print(f"{m["role"]:9s}| {text}{calls}")
    print(f"Шагов: {run.steps}, цена: {run.cost * 100:.3f} ¢, время: {run.seconds:.1f} c")