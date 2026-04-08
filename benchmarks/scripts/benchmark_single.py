import os
import socket
import subprocess
import time
import h2o
import h2o.backend
from h2o.automl import H2OAutoML
import openml
import pandas as pd
from pathlib import Path

# WSL 전용 H2O 초기화 함수
def init_h2o_wsl(mem_size="6G"):
    def get_wsl_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8', 80))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP

    wsl_ip = get_wsl_ip()
    port = "54321"
    jar_path = os.path.join(os.path.dirname(h2o.backend.__file__), "bin", "h2o.jar")

    try:
        h2o.connect(url=f"http://{wsl_ip}:{port}", silent_connect=True)
        print(f"[!] H2O 클러스터가 이미 실행 중입니다: {wsl_ip}")
    except Exception:
        print(f"[!] [{wsl_ip}] 주소로 H2O 서버를 부팅합니다... (약 5초 대기)")
        cmd = ["java", f"-Xmx{mem_size}", "-jar", jar_path, "-ip", wsl_ip, "-port", port]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)
        h2o.init(url=f"http://{wsl_ip}:{port}")

# 실행부
if __name__ == "__main__":
    init_h2o_wsl("6G") # 초기화 실행
    
    ds_name = "australian"
    print(f"\n>>> 테스트 데이터셋: {ds_name}")
    
    data = openml.datasets.get_dataset(ds_name)
    X, y, _, _ = data.get_data(target=data.default_target_attribute)
    hf = h2o.H2OFrame(pd.concat([X, y], axis=1))
    
    target = data.default_target_attribute
    hf[target] = hf[target].asfactor()
    train, test = hf.split_frame(ratios=[0.8], seed=42)

    # 짧게 1분만 테스트
    aml = H2OAutoML(max_runtime_secs=60, seed=42)
    aml.train(y=target, training_frame=train)
    
    perf = aml.leader.model_performance(test)
    print(f"\n[결과] {ds_name} AUC: {round(perf.auc(), 4)}")