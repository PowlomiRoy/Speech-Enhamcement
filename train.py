import os
import argparse
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import ModelCheckpoint
from dataloader import loaddataset
from auraloss.time import SISDRLoss
from auraloss.freq import STFTLoss

# Remove warnings
import warnings
warnings.filterwarnings("ignore")

epsilon = torch.finfo(torch.float32).eps

class DeepLearningModel(pl.LightningModule):
    def __init__(self, net, batch_size=1):
        super(DeepLearningModel, self).__init__()
        self.model = net
        self.modelname = self.model.name
        self.batch_size = batch_size
        self.si_sdr = SISDRLoss()
        self.freqloss = STFTLoss(fft_size=320, hop_size=80, win_length=320, sample_rate=16000, scale_invariance=False,
                                 w_sc=0.0)
        print('\nUsing Si-SDR + STFT loss function to train the network! ....')

        # Store validation outputs
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)
    
    def loss_function(self, cln_audio, enh_audio):
    # Debugging the shapes of the tensors
        print("Clean audio shape:", cln_audio.shape)
        print("Enhanced audio shape:", enh_audio.shape)
    
    # If the tensors have 2 dimensions (e.g., batch_size, sequence_length), add a channel dimension
        if len(cln_audio.shape) == 2:
            cln_audio = cln_audio.unsqueeze(1)  # Add channel dimension (batch_size, 1, sequence_length)
        if len(enh_audio.shape) == 2:
            enh_audio = enh_audio.unsqueeze(1)  # Add channel dimension (batch_size, 1, sequence_length)
    
    # Now calculate the loss
        loss = self.si_sdr(cln_audio, enh_audio) + 25 * self.freqloss(cln_audio, enh_audio)
        
        return loss


    def training_step(self, batch, batch_nb):
        enh_audio = self(batch['noisy'])
        loss = self.loss_function(batch['clean'], enh_audio)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_nb):
        enh_audio = self(batch['noisy'])
        loss = self.loss_function(batch['clean'], enh_audio)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Store the loss for epoch-end calculation
        self.validation_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        self.log("val_loss", avg_loss, prog_bar=True, logger=True)
        self.validation_step_outputs.clear()  # Clear stored outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, weight_decay=1e-5, betas=(0.5, 0.999))
        scheduler = {'scheduler': ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True),
                     'interval': 'epoch', 'frequency': 1, 'reduce_on_plateau': True, 'monitor': 'val_loss'}
        return [optimizer], [scheduler]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Speech Enhancement using SkipConv Net')
    parser.add_argument('--model', type=str, help='ModelName', default='DCCTN_Modified')
    parser.add_argument('--mode', type=str, help='Choose between "summary", "fast_run" or "train"', default='summary')
    parser.add_argument('--b', type=int, help='Batch Size', default=8)
    parser.add_argument('--e', type=int, help='Epochs', default=50)
    parser.add_argument('--loss', type=str, help='Loss function', default='SISDR+FreqLoss')
    args = parser.parse_args()

    # Force CPU usage
    device = torch.device("cpu")
    print("Training on CPU only")

    # Model selection
    if args.model == 'CFTNet':
        from Network import CFTNet
        curr_model = CFTNet()
    elif args.model == 'DCCTN':
        from Network import DCCTN
        curr_model = DCCTN()
    elif args.model == 'DATCFTNET':
        from Network import DATCFTNET
        curr_model = DATCFTNET()
    elif args.model == 'DATCFTNET_DSC':
        from Network import DATCFTNET_DSC
        curr_model = DATCFTNET_DSC()
    else:
        raise ValueError("Invalid model name")

    print("This process has the PID", os.getpid())
    model = DeepLearningModel(curr_model, args.b)
    print('Training Model:', model.modelname)
    print('Saving model to:', os.getcwd() + '/Saved_Models/' + model.modelname)

    # Checkpoint callback
    callbacks = ModelCheckpoint(monitor='val_loss', dirpath=os.getcwd() + '/Saved_Models/' + model.modelname,
                                filename=model.modelname + '-DADX-IEEE-' + args.loss + '-{epoch:02d}-{val_loss:.2f}',
                                save_top_k=1, mode='min')

    # Load datasets
    TrainData = loaddataset(os.getcwd() + '/Database/Training_Samples/Train')
    trainloader = DataLoader(TrainData, batch_size=args.b, shuffle=True, num_workers=0, pin_memory=True)
    DevData = loaddataset(os.getcwd() + '/Database/Training_Samples/Dev')
    devloader = DataLoader(DevData, batch_size=args.b, shuffle=False, num_workers=0, pin_memory=True)

    # Trainer for CPU only
    trainer = pl.Trainer(max_epochs=args.e, accelerator="cpu", callbacks=[callbacks], gradient_clip_val=10, accumulate_grad_batches=8)
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=devloader)

    print('Training completed!')

# Example: python3 train.py --model CFTNet --b 8 --e 50 --loss SISDR+FreqLoss --gpu '0 1'
# Simple Example: python3 train.py --model CFTNet