from torch import nn
import torch
from asteroid.masknn.recurrent import DPRNNBlock
import torch.nn.functional as F
from asteroid.masknn import activations, norms


def seq_cos_sim(ref, target, eps=1e-8):  # we may want to move this in DSP
    """
    Cosine similarity between some reference mics and some target mics
    ref: shape (nmic1, L, seg1)
    target: shape (nmic2, L, seg2)
    """

    assert ref.size(1) == target.size(1), "Inputs should have same length."
    assert ref.size(2) >= target.size(
        2
    ), "Reference input should be no smaller than the target input."

    seq_length = ref.size(1)

    larger_ch = ref.size(0)
    if target.size(0) > ref.size(0):
        ref = ref.expand(target.size(0), ref.size(1), ref.size(2)).contiguous()  # nmic2, L, seg1
        larger_ch = target.size(0)
    elif target.size(0) < ref.size(0):
        target = target.expand(
            ref.size(0), target.size(1), target.size(2)
        ).contiguous()  # nmic1, L, seg2

    # L2 norms
    ref_norm = F.conv1d(
        ref.view(1, -1, ref.size(2)).pow(2),
        torch.ones(ref.size(0) * ref.size(1), 1, target.size(2)).type(ref.type()),
        groups=larger_ch * seq_length,
    )  # 1, larger_ch*L, seg1-seg2+1
    ref_norm = ref_norm.sqrt() + eps
    target_norm = target.norm(2, dim=2).view(1, -1, 1) + eps  # 1, larger_ch*L, 1
    # cosine similarity
    cos_sim = F.conv1d(
        ref.view(1, -1, ref.size(2)),
        target.view(-1, 1, target.size(2)),
        groups=larger_ch * seq_length,
    )  # 1, larger_ch*L, seg1-seg2+1
    cos_sim = cos_sim / (ref_norm * target_norm)

    return cos_sim.view(larger_ch, seq_length, -1)


class TAC(nn.Module):
    def __init__(self, input_dim, hidden_dim=384, activation="prelu", norm_type="gLN"):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.input_tf = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), activations.get(activation)()
        )
        self.avg_tf = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), activations.get(activation)()
        )
        self.concat_tf = nn.Sequential(
            nn.Linear(2 * hidden_dim, input_dim), activations.get(activation)()
        )
        self.norm = norms.get(norm_type)(input_dim)

    def forward(self, x, valid_mics):

        batch_size, nmics, channels, chunk_size, n_chunks = x.size()
        output = self.input_tf(
            x.permute(0, 3, 4, 1, 2).reshape(batch_size * nmics * chunk_size * n_chunks, channels)
        ).reshape(batch_size, chunk_size, n_chunks, nmics, self.hidden_dim)

        # mean pooling across channels
        if valid_mics.max() == 0:
            # fixed geometry array
            mics_mean = output.mean(1)
        else:
            # only consider valid channels
            mics_mean = [
                output[b, :, :, : valid_mics[b]].mean(2).unsqueeze(0) for b in range(batch_size)
            ]  # 1, dim1*dim2, H
            mics_mean = torch.cat(mics_mean, 0)  # B*dim1*dim2, H

        mics_mean = self.avg_tf(
            mics_mean.reshape(batch_size * chunk_size * n_chunks, self.hidden_dim)
        )
        mics_mean = (
            mics_mean.reshape(batch_size, chunk_size, n_chunks, self.hidden_dim)
            .unsqueeze(3)
            .expand_as(output)
        )
        output = torch.cat([output, mics_mean], -1)
        output = self.concat_tf(
            output.reshape(batch_size * chunk_size * n_chunks * nmics, -1)
        ).reshape(batch_size, chunk_size, n_chunks, nmics, -1)
        output = self.norm(
            output.permute(0, 3, 4, 1, 2).reshape(batch_size * nmics, -1, chunk_size, n_chunks)
        ).reshape(batch_size, nmics, -1, chunk_size, n_chunks)

        return output + x


class FasNetTAC(nn.Module):
    def __init__(
        self,
        enc_dim,
        feature_dim,
        hidden_dim,
        n_layers=4,
        segment_size=50,
        nspk=2,
        win_len=4,
        stride=None,
        context_len=16,
        sr=16000,
        tac_hidden_dim=384,
        norm_type="gLN",
        chunk_size=50,
        hop_size=25,
        output_dim=64,
    ):
        super().__init__()

        self.chunk_size = chunk_size
        self.hop_size = hop_size

        assert win_len % 2 == 0, "Window length should be even"
        # parameters
        self.window = int(sr * win_len / 1000)
        self.context = int(sr * context_len / 1000)
        if not stride:
            self.stride = self.window // 2
        else:
            self.stride = int(sr * stride / 1000)

        self.filter_dim = self.context * 2 + 1  # length of beamforming filter
        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = self.context * 2 + 1
        self.segment_size = segment_size

        self.n_layers = n_layers
        self.num_spk = nspk
        self.eps = 1e-8

        # waveform encoder
        self.encoder = nn.Conv1d(1, self.enc_dim, self.context * 2 + self.window, bias=False)
        self.enc_LN = nn.GroupNorm(1, self.enc_dim, eps=1e-8)

        # DPRNN here basically + TAC at each layer
        self.bottleneck = nn.Conv1d(self.filter_dim + self.enc_dim, self.feature_dim, 1, bias=False)

        self.DPRNN_TAC = nn.ModuleList([])
        for i in range(self.n_layers):
            self.DPRNN_TAC.append(
                nn.ModuleList(
                    [DPRNNBlock(self.enc_dim, self.hidden_dim), TAC(self.enc_dim, tac_hidden_dim)]
                )
            )

        # DPRNN output layers
        self.conv_2D = nn.Sequential(nn.PReLU(), nn.Conv2d(self.enc_dim, nspk * self.enc_dim, 1))
        self.tanh = nn.Sequential(nn.Conv1d(self.feature_dim, self.output_dim, 1), nn.Tanh())

        self.gate = nn.Sequential(nn.Conv1d(self.feature_dim, self.output_dim, 1), nn.Sigmoid())

    # beamforming output

    @staticmethod
    def windowing_with_context(x, window, context):
        batch_size, nmic, nsample = x.shape

        unfolded = F.unfold(
            x.unsqueeze(-1),
            kernel_size=(window + 2 * context, 1),
            padding=(context + window, 0),
            stride=(window // 2, 1),
        )

        n_chunks = unfolded.size(-1)
        unfolded = unfolded.reshape(batch_size, nmic, window + 2 * context, n_chunks)

        return (
            unfolded[:, :, context : context + window].transpose(2, -1),
            unfolded.transpose(2, -1),
        )

    def forward(self, x, valid_mics):

        n_samples = x.size(-1)
        all_seg, all_mic_context = self.windowing_with_context(x, self.window, self.context)
        batch_size, mics, seq_length, feats = all_mic_context.size()
        # all_seg contains only the central window

        # encoder applies a filter on each all_mic_context feats
        enc_output = (
            self.encoder(all_mic_context.reshape(batch_size * mics * seq_length, 1, feats))
            .reshape(batch_size * mics, seq_length, self.enc_dim)
            .transpose(1, 2)
            .contiguous()
        )  # B*nmic, N, L
        enc_output = self.enc_LN(enc_output).reshape(batch_size, mics, self.enc_dim, seq_length)

        # for each context window cosine similarity is computed
        ref_seg = all_seg[:, 0].contiguous().view(1, -1, self.window)  # 1, B*L, win
        all_context = (
            all_mic_context.transpose(0, 1)
            .contiguous()
            .view(mics, -1, self.context * 2 + self.window)
        )  # 1, B*L, 3*win
        all_cos_sim = seq_cos_sim(all_context, ref_seg)  # nmic, B*L, 2*win+1
        all_cos_sim = (
            all_cos_sim.view(mics, batch_size, seq_length, self.filter_dim)
            .permute(1, 0, 3, 2)
            .contiguous()
        )  # B, nmic, 2*win+1, L

        input_feature = torch.cat([enc_output, all_cos_sim], 2)  # B, nmic, N+2*win+1, L

        # we now apply DPRNN
        input_feature = self.bottleneck(input_feature.reshape(batch_size * mics, -1, seq_length))

        # we unfold the features for dual path processing
        unfolded = F.unfold(
            input_feature.unsqueeze(-1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )
        n_chunks = unfolded.size(-1)
        unfolded = unfolded.reshape(batch_size * mics, self.enc_dim, self.chunk_size, n_chunks)

        for i in range(self.n_layers):
            dprnn, tac = self.DPRNN_TAC[i]
            out_dprnn = dprnn(unfolded)
            b, ch, chunk_size, n_chunks = out_dprnn.size()
            unfolded = unfolded.reshape(-1, mics, ch, chunk_size, n_chunks)
            unfolded = tac(unfolded, valid_mics).reshape(
                batch_size * mics, self.enc_dim, self.chunk_size, n_chunks
            )

        # output
        unfolded = self.conv_2D(unfolded).reshape(
            batch_size * mics * self.num_spk, self.enc_dim * self.chunk_size, n_chunks
        )
        folded = F.fold(
            unfolded,
            (seq_length, 1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )

        folded = folded.squeeze(-1) / (self.chunk_size / self.hop_size)
        folded = self.tanh(folded) * self.gate(folded)
        folded = folded.view(batch_size, mics, self.num_spk, -1, seq_length)

        # beamforming
        # convolving with all mic context

        all_mic_context = all_mic_context.unsqueeze(2).repeat(1, 1, 2, 1, 1)

        all_bf_output = F.conv1d(
            all_mic_context.view(1, -1, self.context * 2 + self.window),
            folded.transpose(3, -1).contiguous().view(-1, 1, self.filter_dim),
            groups=batch_size * mics * self.num_spk * seq_length,
        )  # 1, B*nmic*nspk*L, win
        all_bf_output = all_bf_output.view(
            batch_size, mics, self.num_spk, seq_length, self.window
        )  # B, nmic, nsp# k, L, win
        all_bf_output = F.fold(
            all_bf_output.reshape(
                batch_size * mics * self.num_spk, seq_length, self.window
            ).transpose(1, -1),
            (n_samples, 1),
            kernel_size=(self.window, 1),
            padding=(self.window, 0),
            stride=(self.window // 2, 1),
        )

        bf_signal = all_bf_output.reshape(batch_size, mics, self.num_spk, n_samples)

        if valid_mics.max() == 0:
            bf_signal = bf_signal.mean(1)  # B, nspk, T
        else:
            bf_signal = [
                bf_signal[b, : valid_mics[b]].mean(0).unsqueeze(0) for b in range(batch_size)
            ]  # nspk, T
            bf_signal = torch.cat(bf_signal, 0)  # B, nspk, T

        return bf_signal


if __name__ == "__main__":
    import numpy as np

    fasnet_tac = FasNetTAC(
        enc_dim=64,
        feature_dim=64,
        hidden_dim=128,
        n_layers=4,
        segment_size=50,
        nspk=2,
        win_len=4,
        context_len=16,
        sr=16000,
    )

    x = torch.rand(2, 4, 32000)  # (batch, num_mic, length)
    num_mic = torch.from_numpy(np.array([3, 2])).view(-1,).type(x.type())  # ad-hoc array
    none_mic = torch.zeros(1).type(x.type())  # fixed-array
    y1 = fasnet_tac(x, num_mic.long())
