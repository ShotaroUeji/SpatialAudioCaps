# FOA形式のRIRを作成し、モノラル音源と畳み込んでFOA音声を出力するスクリプト（正四面体マイクアレイ配置）
# Spatial LibriSpeechの表A.T.2に準拠したパラメトリック部屋生成と制約付きランダム配置を反映

import numpy as np
import pyroomacoustics as pra
import soundfile as sf
import os
from scipy.signal import fftconvolve

# === 入出力設定 ===
INPUT_MONO_PATH = "Ambulance siren passing by with Doppler effect.wav"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 表A.T.2準拠の部屋生成 ===
def sample_room_from_area():
    area = np.random.uniform(13.3, 277.4)  # m^2（Train/Val/Test 共通）
    aspect_ratio = np.random.uniform(1.0, 2.0)
    width = np.sqrt(area / aspect_ratio)
    depth = width * aspect_ratio
    height = np.random.uniform(2.4, 3.5)
    return [width, depth, height]

room_dim = sample_room_from_area()

# === 音源とマイクのランダム位置（壁との距離、距離範囲を制約） ===
def bounded_position(room, min_dist_to_wall=0.4):
    return [
        np.random.uniform(min_dist_to_wall, room[0] - min_dist_to_wall),
        np.random.uniform(min_dist_to_wall, room[1] - min_dist_to_wall),
        np.random.uniform(1.0, room[2] - 0.3)
    ]

mic_center = np.array(bounded_position(room_dim))

# 音源とマイクの距離を制約（0.9m〜4.0m）
def sample_valid_source(room, mic_center, min_d=0.9, max_d=4.0):
    for _ in range(100):
        src = bounded_position(room)
        dist = np.linalg.norm(np.array(src) - np.array(mic_center))
        if min_d <= dist <= max_d:
            return src
    raise ValueError("適切な音源位置が見つかりませんでした。")

src_pos = sample_valid_source(room_dim, mic_center)
rir_save_path = os.path.join(OUTPUT_DIR, "rir_foa.npy")

# === 正四面体マイクアレイ配置 ===
r = 0.05
offsets = r / np.sqrt(3) * np.array([
    [1,  1,  1],
    [-1, -1,  1],
    [-1,  1, -1],
    [1, -1, -1],
])
mic_locs = (mic_center[None, :] + offsets).T  # shape: (3, 4)

# === 高忠実度RIR計算（反射音・T30含む） ===
absorption = np.random.uniform(0.3, 0.6)
room = pra.ShoeBox(
    room_dim,
    fs=16000,
    absorption=absorption,
    max_order=17,
    ray_tracing=False,
    air_absorption=True
)
room.add_source(src_pos)
room.add_microphone_array(pra.MicrophoneArray(mic_locs, room.fs))
room.compute_rir()

# === RIR保存 ===
rir = [room.rir[i][0] for i in range(4)]
rir = np.array(rir)
np.save(rir_save_path, rir)
print(f"✅ FOA-RIR を保存しました: {rir_save_path}")

# === ミキシング処理 ===
sig, sr = sf.read(INPUT_MONO_PATH)
nonzero = np.where(np.abs(sig) > 1e-4)[0]
if len(nonzero) > 0:
    sig = sig[nonzero[0]:nonzero[-1]]
scaling = 10**((np.random.uniform(85, 100) - 100)/20)
sig = sig * scaling

foa = [fftconvolve(sig, rir[i], mode='full') for i in range(4)]
foa = np.stack(foa, axis=1)
sf.write(os.path.join(OUTPUT_DIR, "example_foa.wav"), foa, sr)
print("✅ FOA音声を保存しました: example_foa.wav")
