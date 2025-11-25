# main_train.py
from models.mert_encoder import MERTEncoder
from models.transformer_mixer import MixerTransformer
from data.build_pairs import build_auto_pairs
from data.dataset import ABTransitionDataset
from train import train_model


def main():
    data_folder = "raw_music"  # 你的音乐文件夹，里面全是 .wav

    # 1) Build song pairs
    pairs = build_auto_pairs(data_folder)
    print(f"Found {len(pairs)} pairs.")

    # 2) MERT encoder
    mert = MERTEncoder()
    d_mert = mert.embedding_dim
    target_frames = 200

    # 3) Dataset
    dataset = ABTransitionDataset(
        pairs=pairs,
        mert_encoder=mert,
        target_frames=target_frames
    )

    # 4) Model
    in_dim = 2 * d_mert + 4  # H_A + H_B + PE_A + PE_B
    model = MixerTransformer(
        in_dim=in_dim,
        d_model=512,
        nhead=8,
        num_layers=4,
        dsp_dim=8
    )

    # 5) Train
    train_model(
        model=model,
        dataset=dataset,
        epochs=10,
        batch_size=2,
        lr=1e-4
    )

    # 6) Save
    import torch
    torch.save(model.state_dict(), "bpm_mixer.pt")
    print("Model saved to bpm_mixer.pt")


if __name__ == "__main__":
    main()
