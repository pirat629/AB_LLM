import pandas as pd
from dataclasses import dataclass, field

@dataclass
class Ledger:
    calls: list = field(default_factory=list)

    def add(self, tag, model, usage, seconds = 0.0):
        p, c = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        cost = usage.get("cost", 0.0)
        self.calls.append({
            "tag": tag,
            "model": model.split("/")[-1],
            "prompt": p,
            "completion": c,
            "cost": cost,
            "seconds": round(seconds, 2),
        })

    @property
    def total(self):
        return sum(c["cost"] for c in self.calls)

    def table(self):
        df = pd.DataFrame(self.calls)
        return df.groupby(["tag","model"]).agg(calls=("cost", "size"),
                                               prompt=("prompt","sum"),
                                               completion=("completion","sum"),
                                               cost=("cost","sum"),
                                               seconds=("seconds","sum")).round(5)

ledger = Ledger()