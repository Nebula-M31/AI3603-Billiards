import json
import os
from datetime import datetime


def _to_float_list(arr):
    return [float(x) for x in arr]


def _serialize_balls(balls):
    data = {}
    for bid, ball in balls.items():
        state = ball.state
        data[bid] = {
            "pos": _to_float_list(state.rvw[0]),
            "vel": _to_float_list(state.rvw[1]),
            "spin": _to_float_list(state.rvw[2]),
            "s": int(state.s),
            "t": float(state.t),
        }
    return data


def _serialize_step_info(step_info):
    if not isinstance(step_info, dict):
        return step_info
    cleaned = {}
    for key, value in step_info.items():
        if key == "BALLS" and isinstance(value, dict):
            cleaned[key] = _serialize_balls(value)
        else:
            cleaned[key] = value
    return cleaned


class ShotLogger:
    """Record per-shot data to a JSONL file for post analysis."""

    def __init__(self, out_dir="logs", filename=None):
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_trace_{ts}.jsonl"
        self.out_dir = out_dir
        self.path = os.path.join(out_dir, filename)
        os.makedirs(out_dir, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")

    def log_shot(self, game_idx, hit_count, player, action, my_targets, balls, step_info, done, info):
        record = {
            "game_idx": int(game_idx),
            "hit_count": int(hit_count),
            "player": player,
            "action": dict(action),
            "my_targets": list(my_targets),
            "balls": _serialize_balls(balls),
            "step_info": _serialize_step_info(step_info),
            "done": bool(done),
            "result": info,
        }
        self._fh.write(json.dumps(record, ensure_ascii=True) + "\n")
        self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None
