import json
import os
import re
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

    def __init__(self, out_dir="logs", filename=None, agent_a_tag=None, agent_b_tag=None):
        self._agent_a_tag = self._sanitize_tag(agent_a_tag)
        self._agent_b_tag = self._sanitize_tag(agent_b_tag)
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"eval_trace_{ts}"
            if self._agent_a_tag and self._agent_b_tag:
                base = f"{base}_{self._agent_a_tag}_vs_{self._agent_b_tag}"
            filename = f"{base}.jsonl"
        self.out_dir = out_dir
        self.path = os.path.join(out_dir, filename)
        os.makedirs(out_dir, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")

    def _sanitize_tag(self, tag):
        if not tag:
            return None
        tag = str(tag).strip()
        if not tag:
            return None
        return re.sub(r"[^A-Za-z0-9_-]", "", tag)

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

    def finalize(self, results, n_games):
        if not results or not n_games:
            return
        a_score = float(results.get("AGENT_A_SCORE", 0.0))
        b_score = float(results.get("AGENT_B_SCORE", 0.0))
        a_rate = 100.0 * a_score / float(n_games)
        b_rate = 100.0 * b_score / float(n_games)

        base, ext = os.path.splitext(os.path.basename(self.path))
        newagent_rate = None
        if self._agent_a_tag == "NewAgent":
            newagent_rate = a_rate
        elif self._agent_b_tag == "NewAgent":
            newagent_rate = b_rate
        suffix = f"_NewAgent{newagent_rate:.1f}" if newagent_rate is not None else ""
        new_name = f"{base}{suffix}{ext}"
        new_path = os.path.join(self.out_dir, new_name)
        if new_path == self.path:
            return
        if os.path.exists(new_path):
            idx = 1
            while True:
                candidate = os.path.join(self.out_dir, f"{base}{suffix}_{idx}{ext}")
                if not os.path.exists(candidate):
                    new_path = candidate
                    break
                idx += 1
        try:
            if self._fh:
                self._fh.flush()
            os.rename(self.path, new_path)
            self.path = new_path
        except OSError:
            pass
