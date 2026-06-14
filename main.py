import time
import numpy as np
from ultralytics import YOLO

def main():
    print("=" * 65)
    print(" СИМУЛЯЦІЯ: РОБОТА YOLO НА РC ТА RASPBERRY PI AI KIT (HAILO-8L)")
    print("=" * 65)
    print("Завантаження моделей YOLO (це може зайняти хвилину)...")

    model_nano = YOLO("yolov8n.pt")  
    model_small = YOLO("yolov8s.pt")

    fake_frame = np.zeros((640, 640, 3), dtype=np.uint8)

    print("\n[ОК] Моделі готові. Запуск симуляції інференсу...\n")

    frame_count = 0
    try:
        while True:
            frame_count += 1
            t0 = time.time()
            model_nano(fake_frame, verbose=False)
            t1 = time.time()
            model_small(fake_frame, verbose=False)
            t2 = time.time()

            pc_fps_nano = 1 / (t1 - t0)
            pc_fps_small = 1 / (t2 - t1)

            hailo_fps_nano = 115.0
            hailo_fps_small = 52.0
            combined_hailo_fps = 1 / ((1 / hailo_fps_nano) + (1 / hailo_fps_small))

            print(f" Кадр №{frame_count}")
            print(f"   Хмарний сервер (CPU):")
            print(f"    └─ YOLO Nano : {pc_fps_nano:.2f} FPS")
            print(f"    └─ YOLO Small: {pc_fps_small:.2f} FPS")
            print(f"  Прогноз для Raspberry Pi 5 + AI Kit (Hailo-8L):")
            print(f"    ├─ YOLO Nano (Окремо) : {hailo_fps_nano} FPS")
            print(f"    ├─ YOLO Small (Окремо): {hailo_fps_small} FPS")
            print(f"    └─ КОМБІНОВАНИЙ ЗАПУСК : ~{combined_hailo_fps:.1f} FPS")
            print("-" * 65)
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nСимуляцію зупинено.")

if __name__ == "__main__":
    main()
