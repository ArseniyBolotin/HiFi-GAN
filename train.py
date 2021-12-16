from dataclasses import dataclass
from itertools import repeat
from itertools import islice
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from collate import LJSpeechCollator
from dataset import LJSpeechDataset
from featurizer import MelSpectrogram, MelSpectrogramConfig
from model import Generator
from wandb_writer import WandbWriter

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


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


if __name__ == '__main__':
    # Google drive saving
    # --------------------------------------------------------------
    google_drive = False
    if google_drive:
        from pydrive.auth import GoogleAuth
        from pydrive.drive import GoogleDrive
        from google.colab import auth
        from oauth2client.client import GoogleCredentials
        from googleapiclient.http import MediaFileUpload
        from googleapiclient.discovery import build
        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        drive = GoogleDrive(gauth)
    # --------------------------------------------------------------

    try:
        generator = torch.load("./resume.pt")
    except:
        generator = Generator(GeneratorConfig(), MelSpectrogramConfig()).to(device)
    dataloader = DataLoader(LJSpeechDataset('./data/dataset/LJSpeech'), batch_size=32, collate_fn=LJSpeechCollator())
    wandb_writer = WandbWriter()
    featurizer = MelSpectrogram(MelSpectrogramConfig())

    n_iters = 2000
    save_step = 2000
    output_step = 10

    generator.train()
    criterion = nn.L1Loss()

    optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.8, 0.99), weight_decay=0.01)
    warmup_steps = 4000
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.999)

    current_iter = 1

    batch = list(islice(dataloader, 1))[0]
    mels = featurizer(batch.waveform).to(device)

    while True:
        preds = generator(mels)
        optimizer.zero_grad()
        loss = criterion(featurizer(preds), mels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if current_iter % output_step == 0:
            wandb_writer.set_step(current_iter)
            wandb_writer.add_scalar("Loss", loss.item())
            wandb_writer.add_text("Text sample", batch.transcript[0])
            wandb_writer.add_audio("Ground truth audio", batch.waveform[0], sample_rate=22050)
            wandb_writer.add_scalar('Learning rate', scheduler.get_last_lr()[0])

        if current_iter % save_step == 0:
            model_name = 'fast_speech_' + str(current_iter) + '.pt'
            torch.save(generator, model_name)
            if google_drive:
                drive_service = build('drive', 'v3')
                file_metadata = {'name': model_name}
                media = MediaFileUpload(model_name, resumable=True)
                created = drive_service.files().create(body=file_metadata,
                                                       media_body=media,
                                                       fields='id').execute()
                print("Iteration : ", current_iter)
                print("Save model")
                print('File ID: {}'.format(created.get('id')))

        current_iter += 1
        if current_iter > n_iters:
            break
