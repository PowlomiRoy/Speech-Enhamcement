from real_modules import *
import warnings

warnings.filterwarnings("ignore")
torch.manual_seed(9999)
EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)


def param(nnet, Mb=True):
    neles = sum([param.nelement() for param in nnet.parameters()])
    return np.round(neles / 10 ** 6 if Mb else neles, 2)

# Network 2- DCCTN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class DCCTN(torch.nn.Module):
    """
    FTBComplexSkipConvNet + Transformer
    Complex Skip convolution
    It uses only two FTB layers; one in the first layer and one in the last layer
    Instead of using LSTM, it uses transformer
    """

    def __init__(self, L=256, N=256, H=128, Mask=[5, 7], B=24, F_dim=129):
        super().__init__()
        self.name = 'DCCTN'  
        self.f_taps = list(range(-Mask[0] // 2 + 1, Mask[0] // 2 + 1))
        self.t_taps = list(range(-Mask[1] // 2 + 1, Mask[1] // 2 + 1))

        self.stft = STFT(frame_len=L, frame_hop=H, num_fft=N)
        self.istft = iSTFT(frame_len=L, frame_hop=H, num_fft=N)

        self.enc1 = Encoder(1, 1 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True)
        self.FTB1 = RealFTB(math.ceil(F_dim / 2), channels=1 * B)  # First FTB layer
        self.enc2 = Encoder(1 * B, 2 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True)
        self.FTB2 = RealFTB(math.ceil(F_dim / 4), channels=2 * B)  # First FTB layer
        self.enc3 = Encoder(2 * B, 2 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True)
        self.FTB3 = RealFTB(math.ceil(F_dim / 8), channels=2 * B)  # First FTB layer
        self.enc4 = Encoder(2 * B, 3 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True)
        self.FTB4 = RealFTB(math.ceil(F_dim / 16), channels=3 * B)  # First FTB layer
        self.enc5 = Encoder(3 * B, 3 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True)
        self.FTB5 = RealFTB(math.ceil(F_dim / 32), channels=3 * B)  # First FTB layer
        self.enc6 = Encoder(3 * B, 4 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True)
        self.FTB6 = RealFTB(math.ceil(F_dim / 64), channels=4 * B)  # First FTB layer
        self.enc7 = Encoder(4 * B, 4 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True)
        self.FTB7 = RealFTB(math.ceil(F_dim / 128), channels=4 * B)  # First FTB layer
        self.enc8 = Encoder(4 * B, 8 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True)
        self.TB = RealTransformer(nhead=1, num_layer=2)  # d_model = x.shape[3]
        self.GRU = RealGRU(8 * B, 8 * B, num_layers=2)

        self.skip1 = SkipConnection(8 * B, num_convblocks=4)
        self.skip2 = SkipConnection(4 * B, num_convblocks=4)
        self.skip3 = SkipConnection(4 * B, num_convblocks=3)
        self.skip4 = SkipConnection(3 * B, num_convblocks=3)
        self.skip5 = SkipConnection(3 * B, num_convblocks=2)
        self.skip6 = SkipConnection(2 * B, num_convblocks=2)
        self.skip7 = SkipConnection(2 * B, num_convblocks=1)
        self.skip8 = SkipConnection(1 * B, num_convblocks=1)

        self.dec1 = Decoder(16 * B, 8 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True,
                                   output_padding=(1, 0))
        self.dec2 = Decoder(12 * B, 8 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True)
        self.dec3 = Decoder(12 * B, 4 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True)
        self.dec4 = Decoder(7 * B, 3 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True)
        self.dec5 = Decoder(6 * B, 3 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True)
        self.dec6 = Decoder(5 * B, 2 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True)
        self.dec7 = Decoder(4 * B, 2 * B, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=True)
        self.dec8 = Decoder(3 * B, Mask[0] * Mask[1], kernel_size=(3, 3), stride=(2, 1), padding=(1, 1),
                                   bias=True)

    def cat(self, x, y, dim):
        return torch.cat([x, y], dim)

    def deepfiltering(self, deepfilter, cplxInput):
        deepfilter = deepfilter.permute(0, 2, 3, 1)
        real_tf_shift = torch.stack(
            [torch.roll(cplxInput.real, (i, j), dims=(1, 2)) for i in self.f_taps for j in self.t_taps], 3).transpose(
            -1, -2)
        cplxInput_shift = real_tf_shift
        est_complex = einsum('bftd,bfdt->bft', [deepfilter.conj(), cplxInput_shift])
        return est_complex


        # Apply deep filtering via einsum
        est_real = torch.einsum('btfd,btfd->btf', deepfilter, real_tf_shift)
        return est_real

    def forward(self, audio, verbose=False):
        """
        batch: tensor of shape (batch_size x channels x num_samples)
        """
        if verbose: print('*' * 60)
        if verbose: print('Input Audio Shape         : ', audio.shape)
        if verbose: print('*' * 60)

        #real = self.stft(audio)  # Only keep the real part
        #real_input = real  # Shape: (batch_size, freq_bins, time_frames
        real_input = self.stft(audio) [2] # real part
        if verbose: print('STFT Real Spec            : ', real_input.shape)

        if verbose: print('\n' + '-' * 20)
        if verbose: print('Encoder Network')
        if verbose: print('-' * 20)

        enc1 = self.enc1(real_input.unsqueeze(1))  # Add channel dim if needed
        if verbose: print('Encoder-1                 : ', enc1.shape)
        FTB1 = self.FTB1(enc1)
        if verbose: print('FTB-1               : ', FTB1.shape)
        enc2 = self.enc2(FTB1)
        if verbose: print('Encoder-2                 : ', enc2.shape)
        enc3 = self.enc3(enc2)
        if verbose: print('Encoder-3                 : ', enc3.shape)
        enc4 = self.enc4(enc3)
        if verbose: print('Encoder-4                 : ', enc4.shape)
        enc5 = self.enc5(enc4)
        if verbose: print('Encoder-5                 : ', enc5.shape)
        enc6 = self.enc6(enc5)
        if verbose: print('Encoder-6                 : ', enc6.shape)
        enc7 = self.enc7(enc6)
        if verbose: print('Encoder-7                 : ', enc7.shape)
        FTB7 = self.FTB7(enc7)
        if verbose: print('FTB-7               : ', FTB7.shape)
        enc8 = self.enc8(FTB7)
        if verbose: print('Encoder-8                 : ', enc8.shape)


        # +++++++++++++++++++ Expanding Path  +++++++++++++++++++++ #

        MLTB = self.TB(enc8)
        if verbose: print('Transformer-1               : ', MLTB.shape)
        if verbose: print('\n' + '-' * 20)
        if verbose: print('Decoder Network')
        if verbose: print('-' * 20)
        dec = self.dec1(self.cat(MLTB, self.skip1(enc8), 1))
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        if verbose: print('Decoder-1                 : ', dec.shape)
        dec = self.dec2(self.cat(dec, self.skip2(enc7), 1))
        if verbose: print('Decoder-2                 : ', dec.shape)
        dec = self.dec3(self.cat(dec, self.skip3(enc6), 1))
        if verbose: print('Decoder-3                 : ', dec.shape)
        dec = self.dec4(self.cat(dec, self.skip4(enc5), 1))
        if verbose: print('Decoder-4                 : ', dec.shape)
        dec = self.dec5(self.cat(dec, self.skip5(enc4), 1))
        if verbose: print('Decoder-5                 : ', dec.shape)
        dec = self.dec6(self.cat(dec, self.skip6(enc3), 1))
        if verbose: print('Decoder-6                 : ', dec.shape)
        dec = self.dec7(self.cat(dec, self.skip7(enc2), 1))
        if verbose: print('Decoder-7                 : ', dec.shape)
        dec = self.dec8(self.cat(dec, self.skip8(enc1), 1))
        if verbose: print('Decoder-8                 : ', dec.shape)

        deepfilter = RealTensor(dec.real)
        enhanced = self.deepfiltering(deepfilter, real_input)
        # Skip magnitude/phase and directly use enhanced signal for ISTFT
        audio_enh = self.istft(enhanced, squeeze=True)
        if verbose: print('*' * 60)
        if verbose: print('Output Audio Shape        : ', audio_enh.shape)
        if verbose: print('*' * 60)

        return audio_enh
    
if __name__ == '__main__':
    x = torch.randn(2, 64000)  # Batch x 1 x Samples
    model = DCCTN()
    # x = torch.randn(2, 64000).to('cuda')  # Batch x 1 x Samples
    # model = DCCTN().to('cuda')
    # model = DATCFTNET()
    y = model(x, verbose=True)

    print('\n\n--------------------------------- Script Inputs and Outputs :: Summary')
    print('Model Name          : ', model.name)
    print('Model params (M)    : ', param(model, Mb=True))
    print('Input audio Stream  : ', x.shape)
    print('Output audio Stream : ', y.shape)
    print('--------------------------------------------------------------------------\n')
    print('Done!')

