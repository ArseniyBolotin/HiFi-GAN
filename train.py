from dataclasses import dataclass
from itertools import chain
import random

import librosa

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from collate import LJSpeechCollator
from dataset import LJSpeechDataset
from featurizer import MelSpectrogram, MelSpectrogramConfig
from loss import generator_loss, discriminator_loss, feature_loss
from generator import Generator
from discriminator import MPD, MSD
from utils import inf_loop, set_seed
from wandb_writer import WandbWriter

import google_drive

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class GeneratorConfig:
    h_u = 512
    k_u = [16, 16, 4, 4]
    k_r = [3, 7, 11]
    D_r = [
        [[1, 1], [3, 1], [5, 1]],
        [[1, 1], [3, 1], [5, 1]],
        [[1, 1], [3, 1], [5, 1]]
    ]


if __name__ == '__main__':
    set_seed()
    use_wandb = False
    use_google_drive = False
    # Google drive saving
    # --------------------------------------------------------------
    if use_google_drive:
        google_drive.auth()
    # --------------------------------------------------------------
    try:
        generator = torch.load("./resume/generator.pt").to(device)
        mpd = torch.load("./resume/mpd.pt").to(device)
        msd = torch.load("./resume/msd.pt").to(device)
    except:
        generator = Generator(GeneratorConfig(), MelSpectrogramConfig()).to(device)
        mpd = MPD().to(device)
        msd = MSD().to(device)
    dataloader = DataLoader(LJSpeechDataset('./data/dataset/LJSpeech'), batch_size=16, collate_fn=LJSpeechCollator())
    featurizer = MelSpectrogram(MelSpectrogramConfig()).to(device)
    if use_wandb:
        wandb_writer = WandbWriter()

    # test
    # --------------------------------------------------------------
    TEST_SIZE = 3
    test_texts = [
        'A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest',
        'Massachusetts Institute of Technology may be best known for its math, science and engineering education',
        'Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space'
    ]
    test_wavs = []
    for index in range(1, TEST_SIZE + 1):
        wav, _ = librosa.load('./data/test/audio_' + str(index) + '.wav', sr=22050)
        test_wavs.append(wav)
    # --------------------------------------------------------------

    n_iters = 50000
    save_step = 5000
    output_step = 100

    generator.train()
    mpd.train()
    msd.train()
    g_criterion = nn.L1Loss()

    g_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.8, 0.99), weight_decay=0.01)
    g_scheduler = optim.lr_scheduler.StepLR(g_optimizer, step_size=1, gamma=0.999)

    d_optimizer = optim.Adam(chain(msd.parameters(), mpd.parameters()), lr=2e-4, betas=(0.8, 0.99), weight_decay=0.01)
    d_scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=1, gamma=0.999)

    current_iter = 1
    segment_size = 8192

    for batch in inf_loop(dataloader):
        try:
            waveform = batch.waveform.to(device)
            if waveform.shape[1] - segment_size > 0:
                segment_start = random.randint(0, waveform.shape[1] - segment_size)
                waveform = waveform[:, segment_start:segment_start + segment_size]
            mels = featurizer(waveform)
            waveform = waveform.unsqueeze(1)

            y_preds = generator(mels)
            mel_preds = featurizer(y_preds.squeeze(1))
            preds_common_shape = min(mel_preds.size(2), mels.size(2))
            y_mels = mels[:, :, :preds_common_shape]
            y_pred_mels = mel_preds[:, :, :preds_common_shape]

            d_optimizer.zero_grad()

            real_outputs, pred_outputs, _, _ = mpd(waveform, y_preds.detach())
            mpd_loss = discriminator_loss(real_outputs, pred_outputs)

            real_outputs, pred_outputs, _, _ = msd(waveform, y_preds.detach())
            msd_loss = discriminator_loss(real_outputs, pred_outputs)

            d_loss = mpd_loss + msd_loss

            d_loss.backward()
            d_optimizer.step()

            g_optimizer.zero_grad()
            l1_loss = 45 * g_criterion(y_mels, y_pred_mels)
            _, pred_outputs, real_features, pred_features = mpd(waveform, y_preds)
            generator_loss_mpd = generator_loss(pred_outputs)
            feature_loss_mpd = 2 * feature_loss(real_features, pred_features)
            _, pred_outputs, real_features, pred_features = msd(waveform, y_preds)
            generator_loss_msd = generator_loss(pred_outputs)
            feature_loss_msd = 2 * feature_loss(real_features, pred_features)
            g_loss = generator_loss_msd + generator_loss_mpd + feature_loss_msd + feature_loss_mpd + l1_loss

            g_loss.backward()

            g_optimizer.step()
            if current_iter % output_step == 0 and use_wandb:
                wandb_writer.set_step(current_iter)
                wandb_writer.add_scalar("Loss generator", g_loss.item())
                wandb_writer.add_scalar("Loss discriminator", d_loss.item())
                wandb_writer.add_text("Text sample", batch.transcript[0])
                wandb_writer.add_audio("Ground truth audio sample", batch.waveform[0], sample_rate=22050)
                wandb_writer.add_audio("Generated audio sample", y_preds[0], sample_rate=22050)
                wandb_writer.add_scalar('Learning rate', g_scheduler.get_last_lr()[0])

                generator.eval()
                for index in range(TEST_SIZE):
                    test_wav = torch.Tensor(test_wavs[index])
                    test_wav = test_wav.unsqueeze(0).to(device)
                    test_pred = generator(featurizer(test_wav))
                    wandb_writer.add_audio("Generated audio #" + str(index + 1), test_pred[0], sample_rate=22050)
                    wandb_writer.add_audio("Test audio #" + str(index + 1), test_wav, sample_rate=22050)
                    wandb_writer.add_text("Test text #" + str(index + 1), test_texts[index])
                generator.train()

            if current_iter % save_step == 0:
                for model, name in zip((generator, mpd, msd), ("generator", "mpd", "msd")):
                    model_name = name + '_' + str(current_iter) + '.pt'
                    print("Saving " + model_name)
                    torch.save(model, model_name)
                    if use_google_drive:
                        id = google_drive.save_model(model_name)
                        print('Google drive file ID: {}'.format(id))
        except Exception as e:
            print(e)
        if current_iter % len(dataloader) == 0:
            d_scheduler.step()
            g_scheduler.step()
        current_iter += 1
        if current_iter > n_iters:
            break
