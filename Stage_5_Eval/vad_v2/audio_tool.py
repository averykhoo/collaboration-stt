import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torchaudio
import pytorch_lightning as pl
import numpy as np
import warnings

class AudioTool:
    def __init__(self, 
                 model_path='vad_v2/unetVerySmall_mseLoss.ckpt',
                 criterion='MSE',
                 sum_range=[3, 200]) -> None:
        self.model_path = model_path
        self.criterion = criterion
        self.sum_range = sum_range
        self.sample_rate = 8000
        self.win_size = 1024
        self.hop_len = 512
        self.eps = 1e-6

    def load_model(self, device):
        self.device = device
        model_def = Model()
        
        # Load model from checkpoint using the class, not the instance
        self.model = Pl_module.load_from_checkpoint(
            checkpoint_path=self.model_path,
            model=model_def,
            map_location=device
        )

        # Prepare model for inference
        self.model.eval()
        self.model.to(self.device)

    # Old version - delete it.
    # def load_model(self, device):
    #     self.device = device
    #     model_def = Model()
    #     self.model = Pl_module(model_def)
    #     self.model = self.model.load_from_checkpoint(
    #         model=model_def,
    #         checkpoint_path=self.model_path)
    #     self.model = self.model.eval()
    #     self.model = self.model.to(self.device)

    def infer(self, waveform, sample_rate, thresh=8, n_moving=5):
        self.n_moving = n_moving
        self.threshold = thresh
        waveform, num_wav_channels = self.check_audio_file(waveform)
        
        preds_dict = {}
        for ii in range(num_wav_channels):
            audio_signal = waveform[ii, :].squeeze() if num_wav_channels == 2 else waveform
            log_spec, audio_signal = self.preprocess(audio_signal, sample_rate)
            
            if self.device is not None:
                log_spec = log_spec.to(self.device)
            
            vad_pred = self.predict(log_spec)
            preds_dict[ii], vad_pred_sum = self.postprocessing(
                vad_pred, thresh, n_moving, self.sum_range, len(audio_signal))
        return preds_dict, vad_pred_sum
    
    def preprocess(self, waveform, sample_rate):
        # waveform += torch.randn_like(waveform) * 0.001
        if sample_rate != self.sample_rate:
            resample = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resample(waveform)
        
        waveform = waveform / torch.max(torch.abs(waveform))
        waveform = waveform / (torch.std(waveform) + self.eps)

        spectogram = torchaudio.transforms.Spectrogram(
            n_fft=self.win_size, hop_length=self.hop_len)(waveform)
        
        return torch.log10(spectogram + self.eps), waveform
    
    def predict(self, features):
        if len(features.shape) == 2:
            features = features[None, None, :, :]

        with torch.no_grad():
            pred = self.model(features)
        
        if self.criterion in ['BCELoss', 'BCE', 'bce', 'bceloss']:
            pred = torch.sigmoid(pred)
        
        return pred[0, 0, :, :features.shape[-1]].cpu().detach().numpy()
    
    def postprocessing(
            self, vad_pred, thresh, n_moving, sum_range, waveform_len):
        vad_pred_sum = vad_pred[sum_range[0]: sum_range[1], :].sum(axis=0)
        vad_pred_sum = np.convolve(
            vad_pred_sum.squeeze(), np.ones(n_moving) / n_moving, mode='same')
        
        vad_frame = (vad_pred_sum > thresh).astype(int)
        num_frames_pred = np.size(vad_frame)

        vad_in_time = np.zeros(waveform_len)
        for ii in range(num_frames_pred-1):
            vad_in_time[ii * self.hop_len : (ii + 1) * self.hop_len] = vad_frame[ii: ii + 1].max()
        vad_in_time[(num_frames_pred - 1) * self.hop_len:] = vad_frame[num_frames_pred - 1]
        
        return self.create_vad_dict(vad_in_time), vad_pred_sum

    def check_audio_file(self, waveform):
        num_wav_channels = 1      
        # Multi channel audio file: 
        if (waveform.shape[0] == 2):  
            # If both channels are the same - take just one
            if torch.equal(waveform[0, :], waveform[1, :]):
                waveform = waveform[0, :].squeeze()
            else:
                num_wav_channels = 2

        # Single channel audio file:
        elif waveform.shape[0] == 1 and len(waveform.shape) > 1:
            waveform = waveform.squeeze()

        else:
            print(
                f"Error - received an audio file with unknown number of channels, "
                f"file shape is {waveform.shape}")
        
        return waveform, num_wav_channels
    
    def create_vad_dict(self, vad_in_time):
        #see where the changes in pred are
        changes = np.diff(vad_in_time)
        start_list = np.atleast_1d(np.array(np.argwhere(changes==1).squeeze()))
        end_list = np.atleast_1d(np.array(np.argwhere(changes == -1).squeeze()))
        
        #check if pred_starts/stops at 1 (meaning no change)
        if vad_in_time[0] == 1:
            start_list = np.concatenate(([0], start_list))
        
        if vad_in_time[-1] == 1:
            end_list = np.concatenate((end_list, [np.size(vad_in_time)]))
        
        start_list = np.atleast_1d(start_list)
        end_list = np.atleast_1d(end_list)
        num_of_sections = np.size(start_list)
        vad_times_dict = {}
        
        #create the dict
        for section in range(num_of_sections):
            vad_times_dict[section] = [
                start_list[section] / self.sample_rate, 
                end_list[section]/self.sample_rate]
        
        return vad_times_dict


# ------- Model Definition and Parts ---------
    
class Pl_module(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.inc = (DoubleConv(1, 8))
        self.down1 = (Down(8, 16))
        self.down2 = (Down(16, 32))
        self.down3 = (Down(32, 64))
        self.down4 = (Down(64, 128))
        
        self.up1 = (Up(128, 64))
        self.up2 = (Up(64, 32))
        self.up3 = (Up(32, 16))
        self.up4 = (Up(16, 8))
        self.outc = (OutConv(8, 1))
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        logits = 6 * self.tanh(x)
        
        return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

