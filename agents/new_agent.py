import copy
import math
import random

import numpy as np
import pooltool as pt

from .agent import Agent

class NewAgent(Agent):
    """自定义Agent"""

    def __init__(self):
        super().__init__()
        self._default_ball_radius = 0.028575
        self._sim_noise = {"V0": 0.1, "phi": 0.2, "theta": 0.1, "a": 0.003, "b": 0.003}
        self._max_candidates = 36
        self._simulations = 4
        self._mcts_simulations = 56
        self._rollouts_per_pick = 3
        self._c_puct = 1.25
        self._max_cut_angle = 68.0
        self._bank_penalty = 1.6

    def decision(self, balls=None, my_targets=None, table=None):
        """决策方法

        参数：
            observation: (balls, my_targets, table)

        返回：
            dict: {'V0', 'phi', 'theta', 'a', 'b'}
        """
        if balls is None or my_targets is None or table is None:
            return self._random_action()

        remaining_targets = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
        if len(remaining_targets) == 0:
            remaining_targets = ["8"]
        self._remaining_count = len(remaining_targets)

        cue_pos = self._xy(balls["cue"].state.rvw[0])
        ball_r = self._get_ball_radius(balls["cue"])
        pocket_centers = [self._xy(p.center) for p in table.pockets.values()]
        bounds = self._table_bounds(table)

        candidates = []
        angle_jitter = [0.0, 0.35, -0.35]
        speed_scales = [0.9, 1.0, 1.12]
        for target_id in remaining_targets:
            if target_id not in balls or balls[target_id].state.s == 4:
                continue
            obj_pos = self._xy(balls[target_id].state.rvw[0])
            for pocket_pos in pocket_centers:
                direct = self._build_direct_candidate(
                    cue_pos,
                    obj_pos,
                    pocket_pos,
                    target_id,
                    balls,
                    table,
                    ball_r,
                    bounds,
                    angle_jitter,
                    speed_scales,
                )
                if direct:
                    candidates.extend(direct)

                bank = self._build_bank_candidates(
                    cue_pos,
                    obj_pos,
                    pocket_pos,
                    target_id,
                    balls,
                    table,
                    ball_r,
                    bounds,
                    angle_jitter,
                    speed_scales,
                )
                if bank:
                    candidates.extend(bank)

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            base_top = candidates[: self._max_candidates]
            expanded = self._expand_spin_variants(base_top)
            expanded.sort(key=lambda x: x[0], reverse=True)
            top = expanded[: self._max_candidates]
            best = self._mcts_pick(balls, table, remaining_targets, top)
            if best is not None:
                return best

        safety_candidates = self._rail_safety_candidates(cue_pos, remaining_targets, balls, ball_r)
        if safety_candidates:
            best = self._mcts_pick(balls, table, remaining_targets, safety_candidates)
            if best is not None:
                return best

        return self._safety_shot(cue_pos, remaining_targets, balls, table, ball_r)

    def _xy(self, v):
        v = np.asarray(v, dtype=float)
        return v[:2].copy()

    def _get_ball_radius(self, ball):
        for attr in ("radius", "R"):
            if hasattr(ball, attr):
                try:
                    val = float(getattr(ball, attr))
                    if val > 0:
                        return val
                except Exception:
                    pass
        if hasattr(ball, "params") and hasattr(ball.params, "R"):
            try:
                val = float(ball.params.R)
                if val > 0:
                    return val
            except Exception:
                pass
        return self._default_ball_radius

    def _distance_point_to_segment(self, p, a, b):
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < 1e-12:
            return float(np.linalg.norm(p - a))
        t = float(np.dot(p - a, ab) / denom)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    def _table_bounds(self, table):
        pockets = [self._xy(p.center) for p in table.pockets.values()]
        xs = [p[0] for p in pockets]
        ys = [p[1] for p in pockets]
        return (min(xs), max(xs), min(ys), max(ys))

    def _is_point_playable(self, p, bounds, margin):
        min_x, max_x, min_y, max_y = bounds
        return (min_x + margin) <= p[0] <= (max_x - margin) and (min_y + margin) <= p[1] <= (max_y - margin)

    def _is_path_blocked(self, a, b, balls, exclude_ids, clearance):
        for bid, ball in balls.items():
            if bid in exclude_ids:
                continue
            if ball.state.s == 4:
                continue
            p = self._xy(ball.state.rvw[0])
            if self._distance_point_to_segment(p, a, b) < clearance:
                if np.linalg.norm(p - a) > 1e-4 and np.linalg.norm(p - b) > 1e-4:
                    return True
        return False

    def _is_first_contact_likely(self, cue_pos, aim_dir, aim_dist, target_id, balls, ball_r):
        target_pos = self._xy(balls[target_id].state.rvw[0])
        target_dist_along = float(np.dot(target_pos - cue_pos, aim_dir))
        if target_dist_along <= 0:
            return False
        for bid, ball in balls.items():
            if bid in {"cue", target_id}:
                continue
            if ball.state.s == 4:
                continue
            p = self._xy(ball.state.rvw[0])
            along = float(np.dot(p - cue_pos, aim_dir))
            if along <= 0 or along >= target_dist_along:
                continue
            lateral = float(np.linalg.norm((p - cue_pos) - along * aim_dir))
            if lateral < 2.05 * ball_r:
                return False
        return True

    def _pick_speed(self, d_cue_to_obj, d_obj_to_pocket, cut_angle_deg):
        remaining = getattr(self, "_remaining_count", 7)
        base = 1.0 + 1.05 * d_cue_to_obj + 0.85 * d_obj_to_pocket
        base *= 1.0 + 0.004 * cut_angle_deg
        if remaining <= 2:
            base *= 1.1
        else:
            base *= 0.95
        return float(np.clip(base, 0.8, 5.6))

    def _rank_score(self, d_cue_to_obj, d_obj_to_pocket, cut_angle_deg, bank=False):
        remaining = getattr(self, "_remaining_count", 7)
        angle_penalty = 0.10 * cut_angle_deg
        if remaining <= 2:
            angle_penalty = 0.13 * cut_angle_deg
        score = -1.1 * d_cue_to_obj - 1.3 * d_obj_to_pocket - angle_penalty
        if bank:
            score -= self._bank_penalty
        return score

    def _scratch_risk_penalty(self, cue_pos, aim_dir, table, ball_r, v0):
        remaining = getattr(self, "_remaining_count", 7)
        penalty = 0.0
        speed_scale = 0.8 + 0.25 * min(1.0, v0 / 4.0)
        for p in table.pockets.values():
            pocket = self._xy(p.center)
            ap = pocket - cue_pos
            along = float(np.dot(ap, aim_dir))
            if along <= 0:
                continue
            lateral = float(np.linalg.norm(ap - along * aim_dir))
            if lateral < 2.5 * ball_r and along < 0.85:
                if remaining > 3:
                    penalty += 2.2 * speed_scale
                else:
                    penalty += 1.4 * speed_scale
        return penalty

    def _build_direct_candidate(
        self,
        cue_pos,
        obj_pos,
        pocket_pos,
        target_id,
        balls,
        table,
        ball_r,
        bounds,
        angle_jitter,
        speed_scales,
    ):
        to_pocket = pocket_pos - obj_pos
        to_pocket_norm = np.linalg.norm(to_pocket)
        if to_pocket_norm < 1e-6:
            return []
        to_pocket_dir = to_pocket / to_pocket_norm
        ghost_pos = obj_pos - to_pocket_dir * (2.0 * ball_r)
        if not self._is_point_playable(ghost_pos, bounds, 1.05 * ball_r):
            return []

        aim_vec = ghost_pos - cue_pos
        aim_dist = float(np.linalg.norm(aim_vec))
        if aim_dist < 1e-6:
            return []
        aim_dir = aim_vec / aim_dist
        phi = float(math.degrees(math.atan2(aim_dir[1], aim_dir[0])) % 360.0)

        if self._is_path_blocked(cue_pos, ghost_pos, balls, exclude_ids={"cue", target_id}, clearance=2.15 * ball_r):
            return []
        if self._is_path_blocked(obj_pos, pocket_pos, balls, exclude_ids={"cue", target_id}, clearance=2.1 * ball_r):
            return []
        if self._is_first_contact_likely(cue_pos, aim_dir, aim_dist, target_id, balls, ball_r) is False:
            return []

        cue_to_obj = obj_pos - cue_pos
        cue_to_obj_dist = float(np.linalg.norm(cue_to_obj))
        if cue_to_obj_dist < 1e-6:
            return []
        cue_to_obj_dir = cue_to_obj / cue_to_obj_dist
        cut_cos = float(np.clip(np.dot(cue_to_obj_dir, to_pocket_dir), -1.0, 1.0))
        cut_angle = float(math.degrees(math.acos(cut_cos)))
        if cut_angle > self._max_cut_angle:
            return []

        base_v0 = self._pick_speed(cue_to_obj_dist, float(to_pocket_norm), cut_angle)
        scratch_penalty = self._scratch_risk_penalty(cue_pos, aim_dir, table, ball_r, base_v0)
        score = self._rank_score(cue_to_obj_dist, float(to_pocket_norm), cut_angle) - scratch_penalty
        out = []
        for delta in angle_jitter:
            for scale in speed_scales:
                adj_phi = (phi + delta) % 360.0
                v0 = float(np.clip(base_v0 * scale, 0.8, 5.6))
                action = {"V0": v0, "phi": adj_phi, "theta": 0.0, "a": 0.0, "b": 0.0, "_meta": {"type": "direct"}}
                out.append((score, action))
        return out

    def _reflect_point(self, p, axis, line):
        if axis == "x":
            return np.array([2 * line - p[0], p[1]], dtype=float)
        return np.array([p[0], 2 * line - p[1]], dtype=float)

    def _bank_point(self, obj_pos, pocket_pos, bounds, side):
        min_x, max_x, min_y, max_y = bounds
        if side == "left":
            reflected = self._reflect_point(pocket_pos, "x", min_x)
            line_x = min_x
        elif side == "right":
            reflected = self._reflect_point(pocket_pos, "x", max_x)
            line_x = max_x
        elif side == "bottom":
            reflected = self._reflect_point(pocket_pos, "y", min_y)
            line_y = min_y
        else:
            reflected = self._reflect_point(pocket_pos, "y", max_y)
            line_y = max_y

        d = reflected - obj_pos
        if abs(d[0]) < 1e-6 and (side in {"left", "right"}):
            return None, None
        if abs(d[1]) < 1e-6 and (side in {"bottom", "top"}):
            return None, None

        if side in {"left", "right"}:
            t = (line_x - obj_pos[0]) / d[0]
            if t <= 0 or t >= 1:
                return None, None
            y = obj_pos[1] + t * d[1]
            if y < min_y or y > max_y:
                return None, None
            bank_point = np.array([line_x, y], dtype=float)
        else:
            t = (line_y - obj_pos[1]) / d[1]
            if t <= 0 or t >= 1:
                return None, None
            x = obj_pos[0] + t * d[0]
            if x < min_x or x > max_x:
                return None, None
            bank_point = np.array([x, line_y], dtype=float)
        return bank_point, reflected

    def _build_bank_candidates(
        self,
        cue_pos,
        obj_pos,
        pocket_pos,
        target_id,
        balls,
        table,
        ball_r,
        bounds,
        angle_jitter,
        speed_scales,
    ):
        out = []
        for side in ("left", "right", "bottom", "top"):
            bank_point, reflected = self._bank_point(obj_pos, pocket_pos, bounds, side)
            if bank_point is None:
                continue
            to_pocket = reflected - obj_pos
            to_pocket_norm = np.linalg.norm(to_pocket)
            if to_pocket_norm < 1e-6:
                continue
            to_pocket_dir = to_pocket / to_pocket_norm
            ghost_pos = obj_pos - to_pocket_dir * (2.0 * ball_r)
            if not self._is_point_playable(ghost_pos, bounds, 1.05 * ball_r):
                continue

            aim_vec = ghost_pos - cue_pos
            aim_dist = float(np.linalg.norm(aim_vec))
            if aim_dist < 1e-6:
                continue
            aim_dir = aim_vec / aim_dist
            phi = float(math.degrees(math.atan2(aim_dir[1], aim_dir[0])) % 360.0)

            if self._is_path_blocked(cue_pos, ghost_pos, balls, exclude_ids={"cue", target_id}, clearance=2.1 * ball_r):
                continue
            if self._is_path_blocked(obj_pos, bank_point, balls, exclude_ids={"cue", target_id}, clearance=2.05 * ball_r):
                continue
            if self._is_path_blocked(bank_point, pocket_pos, balls, exclude_ids={"cue", target_id}, clearance=2.05 * ball_r):
                continue
            if self._is_first_contact_likely(cue_pos, aim_dir, aim_dist, target_id, balls, ball_r) is False:
                continue

            cue_to_obj = obj_pos - cue_pos
            cue_to_obj_dist = float(np.linalg.norm(cue_to_obj))
            if cue_to_obj_dist < 1e-6:
                continue
            cue_to_obj_dir = cue_to_obj / cue_to_obj_dist
            cut_cos = float(np.clip(np.dot(cue_to_obj_dir, to_pocket_dir), -1.0, 1.0))
            cut_angle = float(math.degrees(math.acos(cut_cos)))
            if cut_angle > (self._max_cut_angle - 10.0):
                continue

            base_v0 = self._pick_speed(cue_to_obj_dist, float(to_pocket_norm), cut_angle)
            base_v0 = float(np.clip(base_v0 * 1.1, 1.0, 6.2))
            scratch_penalty = self._scratch_risk_penalty(cue_pos, aim_dir, table, ball_r, base_v0)
            score = self._rank_score(cue_to_obj_dist, float(to_pocket_norm), cut_angle, bank=True) - scratch_penalty
            for delta in angle_jitter:
                for scale in speed_scales:
                    adj_phi = (phi + delta) % 360.0
                    v0 = float(np.clip(base_v0 * scale, 1.0, 6.2))
                    action = {"V0": v0, "phi": adj_phi, "theta": 0.0, "a": 0.0, "b": 0.0, "_meta": {"type": "bank"}}
                    out.append((score, action))
        return out

    def _expand_spin_variants(self, candidates):
        if not candidates:
            return candidates
        expanded = []
        spin_variants = [
            (0.0, 0.0, 0.0),
            (0.0, 0.12, 0.0),
            (0.0, -0.12, 0.0),
            (0.0, 0.0, 0.12),
            (0.0, 0.0, -0.12),
            (6.0, 0.0, -0.08),
        ]
        for score, action in candidates:
            for theta, a, b in spin_variants:
                new_action = dict(action)
                new_action["theta"] = theta
                new_action["a"] = a
                new_action["b"] = b
                expanded.append((score - 0.1 * (abs(a) + abs(b) + theta / 10.0), new_action))
        return expanded

    def _safety_shot(self, cue_pos, remaining_targets, balls, table, ball_r):
        best = None
        for tid in remaining_targets:
            if tid not in balls or balls[tid].state.s == 4:
                continue
            tpos = self._xy(balls[tid].state.rvw[0])
            vec = tpos - cue_pos
            dist = float(np.linalg.norm(vec))
            if dist < 1e-6:
                continue
            dir2 = vec / dist
            if self._is_first_contact_likely(cue_pos, dir2, dist, tid, balls, ball_r) is False:
                continue
            phi = float(math.degrees(math.atan2(dir2[1], dir2[0])) % 360.0)
            v0 = float(np.clip(1.0 + 0.7 * dist, 0.9, 3.2))
            score = -dist - 0.1 * self._scratch_risk_penalty(cue_pos, dir2, table, ball_r, v0)
            cand = (score, {"V0": v0, "phi": phi, "theta": 0.0, "a": 0.0, "b": -0.08, "_meta": {"type": "safety"}})
            if best is None or cand[0] > best[0]:
                best = cand
        if best is not None:
            return best[1]
        return {"V0": 1.2, "phi": 0.0, "theta": 0.0, "a": 0.0, "b": 0.0}

    def _rail_safety_candidates(self, cue_pos, remaining_targets, balls, ball_r):
        candidates = []
        angle_jitter = [0.0, 0.5, -0.5]
        speed_scales = [1.0, 1.3]
        for tid in remaining_targets:
            if tid not in balls or balls[tid].state.s == 4:
                continue
            tpos = self._xy(balls[tid].state.rvw[0])
            vec = tpos - cue_pos
            dist = float(np.linalg.norm(vec))
            if dist < 1e-6:
                continue
            dir2 = vec / dist
            if self._is_first_contact_likely(cue_pos, dir2, dist, tid, balls, ball_r) is False:
                continue
            phi_base = float(math.degrees(math.atan2(dir2[1], dir2[0])) % 360.0)
            base_v0 = float(np.clip(1.0 + 0.8 * dist, 1.0, 4.2))
            base_score = -dist - 0.2 * dist
            for delta in angle_jitter:
                for scale in speed_scales:
                    phi = (phi_base + delta) % 360.0
                    v0 = float(np.clip(base_v0 * scale, 0.9, 4.6))
                    candidates.append(
                        (
                            base_score,
                            {"V0": v0, "phi": phi, "theta": 0.0, "a": 0.0, "b": -0.08, "_meta": {"type": "rail_safety"}},
                        )
                    )
        return candidates

    def _simulate_action(self, balls, table, action):
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)

        def _n(val, key, lo, hi):
            std = float(self._sim_noise.get(key, 0.0))
            val = val + random.gauss(0.0, std)
            return float(np.clip(val, lo, hi))

        try:
            V0 = _n(action["V0"], "V0", 0.5, 8.0)
            phi = _n(action["phi"], "phi", -1e9, 1e9) % 360.0
            theta = _n(action["theta"], "theta", 0.0, 90.0)
            a = _n(action["a"], "a", -0.5, 0.5)
            b = _n(action["b"], "b", -0.5, 0.5)
            cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
            pt.simulate(shot, inplace=True)
            return shot
        except Exception:
            return None

    def _analyze_shot_for_reward(self, shot, last_state, player_targets):
        new_pocketed = [
            bid
            for bid, ball in shot.balls.items()
            if ball.state.s == 4 and last_state[bid].state.s != 4
        ]

        own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
        enemy_pocketed = [
            bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]
        ]

        cue_pocketed = "cue" in new_pocketed
        eight_pocketed = "8" in new_pocketed

        first_contact_ball_id = None
        foul_first_hit = False
        valid_ball_ids = {
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
        }

        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, "ids") else []
            if ("cushion" not in et) and ("pocket" not in et) and ("cue" in ids):
                other_ids = [i for i in ids if i != "cue" and i in valid_ball_ids]
                if other_ids:
                    first_contact_ball_id = other_ids[0]
                    break

        if first_contact_ball_id is None:
            if len(last_state) > 2 or player_targets != ["8"]:
                foul_first_hit = True
        else:
            if first_contact_ball_id not in player_targets:
                foul_first_hit = True

        cue_hit_cushion = False
        target_hit_cushion = False
        foul_no_rail = False

        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, "ids") else []
            if "cushion" in et:
                if "cue" in ids:
                    cue_hit_cushion = True
                if first_contact_ball_id is not None and first_contact_ball_id in ids:
                    target_hit_cushion = True

        if (
            len(new_pocketed) == 0
            and first_contact_ball_id is not None
            and (not cue_hit_cushion)
            and (not target_hit_cushion)
        ):
            foul_no_rail = True

        score = 0
        if cue_pocketed and eight_pocketed:
            score -= 300
        elif cue_pocketed:
            score -= 160
        elif eight_pocketed:
            if player_targets == ["8"]:
                score += 130
            else:
                score -= 260

        if foul_first_hit:
            score -= 45
        if foul_no_rail:
            score -= 80

        score += len(own_pocketed) * 55
        score -= len(enemy_pocketed) * 30

        if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
            score = 8

        return score

    def _simulate_pick(self, balls, table, remaining_targets, candidates):
        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        best_action = None
        best_score = -1e9

        for _, action in candidates:
            total = 0.0
            n_ok = 0
            for _ in range(max(1, self._simulations)):
                shot = self._simulate_action(balls, table, action)
                if shot is None:
                    continue
                total += self._analyze_shot_for_reward(shot, last_state_snapshot, remaining_targets)
                n_ok += 1
            avg = total / n_ok if n_ok else -500.0
            if avg > best_score:
                best_score = avg
                best_action = action
            if best_score >= 120:
                break

        return best_action

    def _mcts_pick(self, balls, table, remaining_targets, candidates):
        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        n_candidates = len(candidates)
        if n_candidates == 0:
            return None
        N = np.zeros(n_candidates, dtype=float)
        Q = np.zeros(n_candidates, dtype=float)

        for i in range(self._mcts_simulations):
            if i < n_candidates:
                idx = i
            else:
                total_n = float(np.sum(N)) + 1e-6
                avg = Q / (N + 1e-6)
                ucb = avg + self._c_puct * np.sqrt(np.log(total_n + 1.0) / (N + 1e-6))
                idx = int(np.argmax(ucb))

            _, action = candidates[idx]
            total_reward = 0.0
            foul_hits = 0
            n_ok = 0
            for _ in range(self._rollouts_per_pick):
                shot = self._simulate_action(balls, table, action)
                if shot is None:
                    raw_reward = -500.0
                else:
                    raw_reward = self._analyze_shot_for_reward(shot, last_state_snapshot, remaining_targets)
                total_reward += raw_reward
                n_ok += 1
                if raw_reward <= -140:
                    foul_hits += 1

            avg_reward = total_reward / float(n_ok) if n_ok else -500.0
            foul_rate = foul_hits / float(n_ok) if n_ok else 1.0
            normalized = (avg_reward - (-500.0)) / 650.0
            normalized = float(np.clip(normalized, 0.0, 1.0))
            normalized *= (1.0 - 0.4 * foul_rate)
            N[idx] += 1.0
            Q[idx] += normalized

        avg_rewards = Q / (N + 1e-6)
        best_idx = int(np.argmax(avg_rewards))
        return candidates[best_idx][1]
