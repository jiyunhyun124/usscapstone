import numpy as np
from scipy.ndimage import laplace  # [핵심] 심화 모델의 상징!

class ScientificCleanupRecovery:
    def __init__(self, H=64, W=64):
        self.H = H
        self.W = W
        
        # [Scientific Constants]
        self.k_deg_natural = 0.05   # 자연 분해율
        self.k_deg_active = 0.2     # 방제 작업 시 분해율
        self.k_reaer = 0.4          # 산소 재폭기 계수
        self.O2_demand = 3.0        # 산소 소비율
        
        # 생태 성장 관련
        self.growth_rate = 0.5
        self.carrying_capacity = 100.0 
        self.toxicity_threshold = 0.5
        
        # [심화] 확산 계수 (Diffusion) - 이게 있어야 고급 모델입니다!
        self.D_oil = 0.02
        self.D_oxygen = 0.15 
        self.D_plankton = 0.05

        # 민감도 맵
        y, x = np.ogrid[:H, :W]
        self.bio_sensitivity = 0.5 + 0.5 * (x / W)

    def step_1_calculate_hotspots(self, oil_map, toc_map):
        norm_oil = oil_map / (np.max(oil_map) + 1e-6)
        norm_toc = toc_map / (np.max(toc_map) + 1e-6)
        chem_score = norm_oil + 0.5 * norm_toc
        return chem_score * self.bio_sensitivity

    def step_2_physics_cleaning(self, oil_map, toc_map, priority_map, ships=3, capacity=10):
        cleaned_oil = oil_map.copy()
        cleaned_toc = toc_map.copy()
        flat_priority = priority_map.flatten()
        target_indices = np.argpartition(flat_priority, -ships * capacity)[-ships * capacity:]
        
        removal_eff = 0.9
        ys, xs = np.unravel_index(target_indices, (self.H, self.W))
        cleaned_oil[ys, xs] *= (1 - removal_eff)
        cleaned_toc[ys, xs] *= (1 - removal_eff)
        
        mask = np.zeros_like(oil_map)
        mask[ys, xs] = 1.0
        return cleaned_oil, cleaned_toc, mask

    def step_3_recovery_odes(self, oil_map, do_map, plankton_map, dt_days=1.0):
        # [심화] 1. 확산 (Diffusion) 계산
        diff_oil = self.D_oil * laplace(oil_map)
        diff_do = self.D_oxygen * laplace(do_map)
        diff_plank = self.D_plankton * laplace(plankton_map)
        
        # 2. 반응 (Reaction) 계산
        delta_oil = self.k_deg_natural * oil_map 
        
        consumption = delta_oil * self.O2_demand
        dampening = 1.0 / (1.0 + 10.0 * oil_map)
        reaeration = self.k_reaer * dampening * (8.0 - do_map)
        react_do = reaeration - consumption
        
        tox_inhib = np.clip(oil_map / self.toxicity_threshold, 0, 1.0)
        growth = self.growth_rate * (1 - plankton_map/self.carrying_capacity) * (1 - tox_inhib)
        react_plank = growth * plankton_map

        # [심화] 3. 최종 업데이트 (기존 값 + 확산 + 반응)
        next_oil = oil_map + (diff_oil - delta_oil) * dt_days
        next_do = do_map + (diff_do + react_do) * dt_days
        next_plank = plankton_map + (diff_plank + react_plank) * dt_days
        
        # 물리적 한계 설정
        next_oil = np.clip(next_oil, 0, None)
        next_do = np.clip(next_do, 0, 8.0)
        next_plank = np.clip(next_plank, 0, None)
        
        return next_oil, next_do, next_plank, None
