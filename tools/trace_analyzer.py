import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _assign_agent(game_idx, player):
    # evaluate.py: agent_a = BApro, agent_b = NewAgent, players = [agent_a, agent_b]
    # Player A uses players[i % 2], Player B uses players[(i + 1) % 2]
    if player == "A":
        return "BApro" if game_idx % 2 == 0 else "NewAgent"
    return "NewAgent" if game_idx % 2 == 0 else "BApro"


def analyze(path):
    per_agent = defaultdict(Counter)
    per_type = defaultdict(Counter)
    win = Counter()
    shots = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            shots += 1
            game_idx = rec["game_idx"]
            player = rec["player"]
            agent = _assign_agent(game_idx, player)
            step = rec.get("step_info", {}) or {}
            info = rec.get("result", {}) or {}
            done = rec.get("done", False)
            action = rec.get("action", {}) or {}
            meta = action.get("_meta", {}) if isinstance(action, dict) else {}
            shot_type = meta.get("type", "unknown")

            if step.get("FOUL_FIRST_HIT"):
                per_agent[agent]["foul_first_hit"] += 1
                per_type[shot_type]["foul_first_hit"] += 1
            if step.get("NO_POCKET_NO_RAIL"):
                per_agent[agent]["no_pocket_no_rail"] += 1
                per_type[shot_type]["no_pocket_no_rail"] += 1
            if step.get("NO_HIT"):
                per_agent[agent]["no_hit"] += 1
                per_type[shot_type]["no_hit"] += 1
            if step.get("WHITE_BALL_INTO_POCKET"):
                per_agent[agent]["scratch"] += 1
                per_type[shot_type]["scratch"] += 1
            if step.get("BLACK_BALL_INTO_POCKET") and info.get("winner") and info.get("winner") != player:
                per_agent[agent]["black_foul"] += 1
                per_type[shot_type]["black_foul"] += 1

            per_type[shot_type]["shots"] += 1
            per_agent[agent]["shots"] += 1

            if done:
                winner = info.get("winner")
                if winner in {"A", "B"}:
                    win_agent = _assign_agent(game_idx, winner)
                    win[win_agent] += 1
                else:
                    win["SAME"] += 1

    print(f"Total shots: {shots}")
    print(f"Wins: {dict(win)}")
    for agent in sorted(per_agent.keys()):
        print(f"{agent}: {dict(per_agent[agent])}")
    print("By shot type:")
    for shot_type in sorted(per_type.keys()):
        print(f"{shot_type}: {dict(per_type[shot_type])}")


def main():
    parser = argparse.ArgumentParser(description="Analyze eval_trace jsonl logs.")
    parser.add_argument("path", type=Path, help="Path to eval_trace_*.jsonl")
    args = parser.parse_args()
    analyze(args.path)


if __name__ == "__main__":
    main()
