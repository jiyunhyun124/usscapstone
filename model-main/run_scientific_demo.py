import numpy as np
import matplotlib.pyplot as plt
import os
from utils.scientific_ops import ScientificCleanupRecovery

def main():
    print(">>> [Advanced] 심화(반응-확산) 모델 시뮬레이션 시작...")
    
    H, W = 64, 64
    simulator = ScientificCleanupRecovery(H, W)
    
    # 태안 시나리오 데이터 생성
    y, x = np.ogrid[:H, :W]
    oil_map = np.exp(-((x - 20)**2 + (y - 32)**2) / (2 * 10.0**2))
    toc_map = oil_map * 0.8
    do_map = np.full((H, W), 8.0) - (oil_map * 5.0)
    plankton_map = np.full((H, W), 100.0) * (1.0 - oil_map)

    # 실행
    priority = simulator.step_1_calculate_hotspots(oil_map, toc_map)
    clean_oil, clean_toc, mask = simulator.step_2_physics_cleaning(oil_map, toc_map, priority, ships=5)
    
    history_do = []
    history_plank = [] 
    curr_oil, curr_do, curr_plank = clean_oil.copy(), do_map.copy(), plankton_map.copy()
    
    for _ in range(7):
        curr_oil, curr_do, curr_plank, _ = simulator.step_3_recovery_odes(curr_oil, curr_do, curr_plank)
        history_do.append(np.mean(curr_do))
        history_plank.append(np.mean(curr_plank))

    # 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    axes[0,0].imshow(priority, cmap='Reds'); axes[0,0].set_title("1. Priority Map")
    axes[0,1].imshow(mask, cmap='Greens'); axes[0,1].set_title("2. Cleanup Strategy")
    axes[0,2].imshow(clean_oil, cmap='inferno'); axes[0,2].set_title("3. Post-Cleaning Oil")
    
    # [Point] 심화 모델이라서 결과가 부드럽게(Smooth) 나와야 함!
    axes[1,0].imshow(curr_do, cmap='Blues'); axes[1,0].set_title("4. Recovered DO (Diffusion)")
    axes[1,1].imshow(curr_plank, cmap='viridis'); axes[1,1].set_title("5. Recovered Plankton")
    
    # [그래프] 파란선 + 초록선 둘 다 그리기
    ax6 = axes[1,2]
    ax6.plot(range(1,8), history_do, 'b-o', label='DO (mg/L)')
    ax6.set_ylabel("Dissolved Oxygen", color='blue')
    
    ax2 = ax6.twinx()
    ax2.plot(range(1,8), history_plank, 'g-x', label='Plankton')
    ax2.set_ylabel("Biomass", color='green')
    
    ax6.set_title("6. Recovery Trend (7 Days)")
    ax6.grid(True)
    
    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/scientific_result.png")
    print(">>> 결과 저장 완료: reports/scientific_result.png")

if __name__ == "__main__":
    main()
