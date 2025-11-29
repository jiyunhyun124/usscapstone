import numpy as np

class ScientificCleanupRecovery:
    def __init__(self, H=64, W=64):
        self.H = H
        self.W = W
        # [Scientific Constants]
        self.k_deg_natural = 0.05
        self.k_deg_active = 0.2
        self.k_reaer = 0.4
        self.O2_demand = 3.0
        
        # 생태 성장 관련
        self.growth_rate = 0.5
        self.carrying_capacity = 100.0 
        self.toxicity_threshold = 0.5
        
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
        # 1. Oil Degradation
        delta_oil = self.k_deg_natural * oil_map * dt_days
        next_oil = np.clip(oil_map - delta_oil, 0, None)
        
        # 2. DO Sag & Recovery
        consumption = delta_oil * self.O2_demand
        dampening = 1.0 / (1.0 + 10.0 * oil_map)
        reaeration = self.k_reaer * dampening * (8.0 - do_map) * dt_days
        next_do = np.clip(do_map - consumption + reaeration, 0, 8.0)
        
        # 3. Plankton Recovery (변수명 오류 수정됨)
        tox_inhib = np.clip(oil_map / self.toxicity_threshold, 0, 1.0)
        growth_factor = (1.0 - plankton_map / self.carrying_capacity)
        effective_growth = self.growth_rate * growth_factor * (1.0 - tox_inhib)
        next_plankton = plankton_map + (effective_growth * plankton_map * dt_days)
        
        next_chla = next_plankton * 0.05
        return next_oil, next_do, next_plankton, next_chla
