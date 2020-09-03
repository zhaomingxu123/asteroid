from torch.utils.data import Dataset
import json
import torchaudio
import torch
import numpy as np


class TACDataset(Dataset):
    def __init__(self, json_file, segment=None, samplerate=16000, max_mics=6):

        self.segment = segment
        self.samplerate = samplerate
        self.max_mics = max_mics

        with open(json_file, "r") as f:
            examples = json.load(f)

        if self.segment:
            target_len = int(segment * samplerate)
            self.examples = []
            for ex in examples:
                if ex["mixture"][0]["length"] < target_len:
                    continue
                self.examples.append(ex)
            print(
                "Discarded {} out of {} because too short".format(
                    len(examples) - len(self.examples), len(examples)
                )
            )
        else:
            self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):

        c_ex = self.examples[item]

        mixture = []
        for mic in c_ex["mixture"]:
            if self.segment:
                offset = 0
                if mic["length"] > int(self.segment * self.samplerate):
                    offset = np.random.randint(
                        0, mic["length"] - int(self.segment * self.samplerate)
                    )

                tmp, fs = torchaudio.load(
                    mic["file"],
                    offset=offset,
                    num_frames=int(self.segment * self.samplerate),
                    normalization=False,
                )
            else:
                tmp, fs = torchaudio.load(mic["file"], normalization=False)  # load all

            assert fs == self.samplerate
            mixture.append(tmp)
        mixture = torch.cat(mixture, 0)

        # sources
        speakers = list([k for k in c_ex.keys() if k != "mixture"])

        sources = []
        for spk in speakers:
            c_spk = []
            for mic in range(len(c_ex[spk])):
                if self.segment:
                    offset = 0
                    if c_ex[spk][mic]["length"] > int(self.segment * self.samplerate):
                        offset = np.random.randint(
                            0, c_ex[spk][mic]["length"] - int(self.segment * self.samplerate)
                        )
                    tmp, fs = torchaudio.load(
                        c_ex[spk][mic]["file"],
                        offset=offset,
                        num_frames=int(self.segment * self.samplerate),
                        normalization=False,
                    )
                else:
                    tmp, fs = torchaudio.load(c_ex[spk][mic]["file"], normalization=False)
                assert fs == self.samplerate
                c_spk.append(tmp)
            c_spk = torch.cat(c_spk, 0)
            sources.append(c_spk)
        sources = torch.stack(sources)

        # we pad till max_mic
        valid_mics = mixture.shape[0]
        if mixture.shape[0] < self.max_mics:
            dummy = torch.zeros((self.max_mics - mixture.shape[0], mixture.shape[-1]))
            mixture = torch.cat((mixture, dummy), 0)
            sources = torch.cat((sources, dummy.repeat(sources.shape[0], 1, 1)), 1)

        return mixture, sources, valid_mics


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    data = TACDataset("/media/sam/Data/DPTransformer/asteroid/egs/TAC/data/train.json", segment=2)
    for i in DataLoader(data, batch_size=8, shuffle=True):
        print(i)
