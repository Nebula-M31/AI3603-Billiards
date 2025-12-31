import copy
import math
import random

import numpy as np
import pooltool as pt

from .agent import Agent

class NewAgent(Agent):
    """自定义 Agent"""
    
    def __init__(self):
        super().__init__()
        self._default_ball_radius = 0.028575
        self._sim_noise = {"V0": 0.1, "phi": 0.2, "theta": 0.1, "a": 0.003, "b": 0.003}
        self._max_candidates = 24
        self._simulations = 4
    
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

        candidates = []
        angle_jitter = [0.0, 0.4, -0.4]
        speed_scales = [0.9, 1.0, 1.15]
        for target_id in remaining_targets:
            if target_id not in balls or balls[target_id].state.s == 4:
                continue
            obj_pos = self._xy(balls[target_id].state.rvw[0])
            for pocket_pos in pocket_centers:
                to_pocket = pocket_pos - obj_pos
                to_pocket_norm = np.linalg.norm(to_pocket)
                if to_pocket_norm < 1e-6:
                    continue
                to_pocket_dir = to_pocket / to_pocket_norm
                ghost_pos = obj_pos - to_pocket_dir * (2.0 * ball_r)

                aim_vec = ghost_pos - cue_pos
                aim_dist = float(np.linalg.norm(aim_vec))
                if aim_dist < 1e-6:
                    continue
                aim_dir = aim_vec / aim_dist
                phi = float(math.degrees(math.atan2(aim_dir[1], aim_dir[0])) % 360.0)

                if self._is_path_blocked(cue_pos, ghost_pos, balls, exclude_ids={"cue", target_id}, clearance=2.05 * ball_r):
                    continue
                if self._is_path_blocked(obj_pos, pocket_pos, balls, exclude_ids={"cue", target_id}, clearance=2.05 * ball_r):
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

                base_score = self._rank_score(cue_to_obj_dist, float(to_pocket_norm), cut_angle)
                scratch_penalty = self._scratch_risk_penalty(cue_pos, aim_dir, table, ball_r)
                score = base_score - scratch_penalty

                base_v0 = self._pick_speed(cue_to_obj_dist, float(to_pocket_norm), cut_angle)
                for delta in angle_jitter:
                    for scale in speed_scales:
                        adj_phi = (phi + delta) % 360.0
                        v0 = float(np.clip(base_v0 * scale, 0.8, 5.6))
                        candidates.append(
                            (
                                score,
                                {"V0": v0, "phi": adj_phi, "theta": 0.0, "a": 0.0, "b": 0.0},
                            )
                        )

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            top = candidates[: self._max_candidates]
            best = self._simulate_pick(balls, table, remaining_targets, top)
            if best is not None:
                return best

        safety = self._safety_shot(cue_pos, remaining_targets, balls, table, ball_r)
        return safety

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

    def _rank_score(self, d_cue_to_obj, d_obj_to_pocket, cut_angle_deg):
        remaining = getattr(self, "_remaining_count", 7)
        angle_penalty = 0.10 * cut_angle_deg
        if remaining <= 2:
            angle_penalty = 0.13 * cut_angle_deg
        return -1.1 * d_cue_to_obj - 1.3 * d_obj_to_pocket - angle_penalty

    def _scratch_risk_penalty(self, cue_pos, aim_dir, table, ball_r):
        remaining = getattr(self, "_remaining_count", 7)
        penalty = 0.0
        for p in table.pockets.values():
            pocket = self._xy(p.center)
            ap = pocket - cue_pos
            along = float(np.dot(ap, aim_dir))
            if along <= 0:
                continue
            lateral = float(np.linalg.norm(ap - along * aim_dir))
            if lateral < 2.4 * ball_r and along < 0.8:
                if remaining > 3:
                    penalty += 2.0
                else:
                    penalty += 1.2
        return penalty

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
            v0 = float(np.clip(0.9 + 0.6 * dist, 0.8, 2.6))
            score = -dist - 0.1 * self._scratch_risk_penalty(cue_pos, dir2, table, ball_r)
            cand = (score, {'V0': v0, 'phi': phi, 'theta': 0.0, 'a': 0.0, 'b': 0.0})
            if best is None or cand[0] > best[0]:
                best = cand
        if best is not None:
            return best[1]
        return {'V0': 1.2, 'phi': 0.0, 'theta': 0.0, 'a': 0.0, 'b': 0.0}

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
            score -= 150
        elif cue_pocketed:
            score -= 100
        elif eight_pocketed:
            if player_targets == ["8"]:
                score += 100
            else:
                score -= 150

        if foul_first_hit:
            score -= 30
        if foul_no_rail:
            score -= 30

        score += len(own_pocketed) * 50
        score -= len(enemy_pocketed) * 20

        if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
            score = 10

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
