import math
import numpy as np

from .agent import Agent

class NewAgent(Agent):
    """自定义 Agent"""
    
    def __init__(self):
        super().__init__()
        self._default_ball_radius = 0.028575
    
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

        cue_pos = self._xy(balls["cue"].state.rvw[0])
        ball_r = self._get_ball_radius(balls["cue"])
        pocket_centers = [self._xy(p.center) for p in table.pockets.values()]

        candidates = []
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

                v0 = self._pick_speed(cue_to_obj_dist, float(to_pocket_norm), cut_angle)
                base_score = self._rank_score(cue_to_obj_dist, float(to_pocket_norm), cut_angle)
                scratch_penalty = self._scratch_risk_penalty(cue_pos, aim_dir, table, ball_r)
                score = base_score - scratch_penalty

                candidates.append((score, {'V0': v0, 'phi': phi, 'theta': 0.0, 'a': 0.0, 'b': 0.0}))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

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
        base = 1.0 + 1.25 * d_cue_to_obj + 0.85 * d_obj_to_pocket
        base *= 1.0 + 0.008 * cut_angle_deg
        return float(np.clip(base, 0.8, 6.8))

    def _rank_score(self, d_cue_to_obj, d_obj_to_pocket, cut_angle_deg):
        return -1.2 * d_cue_to_obj - 1.0 * d_obj_to_pocket - 0.06 * cut_angle_deg

    def _scratch_risk_penalty(self, cue_pos, aim_dir, table, ball_r):
        penalty = 0.0
        for p in table.pockets.values():
            pocket = self._xy(p.center)
            ap = pocket - cue_pos
            along = float(np.dot(ap, aim_dir))
            if along <= 0:
                continue
            lateral = float(np.linalg.norm(ap - along * aim_dir))
            if lateral < 2.2 * ball_r and along < 0.7:
                penalty += 1.5
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
