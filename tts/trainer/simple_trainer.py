from tqdm.autonotebook import tqdm
import torch
from tts.logger.wandb import WanDBWriter
from tts.utils.util import set_require_grad
from tts.model.generator import Generator
from tts.model.discriminator import Discriminator
from tts.collate_fn.collate import Batch
from torchvision.transforms import ToTensor
from collections import defaultdict
import numpy as np
import io
import matplotlib.pyplot as plt
from PIL import Image


def plot_spectrogram_to_buf(spectrogram_tensor, name=None):
    plt.figure(figsize=(20, 5))
    plt.imshow(spectrogram_tensor)
    plt.title(name)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf



def log_audios(batch: Batch, logger: WanDBWriter):
    if batch.waveform is not None:
        logger.add_audio("true_audio", batch.waveform[0], sample_rate=22050)

    if batch.waveform_prediction is not None:
        logger.add_audio("pred_audio", batch.waveform_prediction[0], sample_rate=22050)


def train_epoch(model, optimizer, loader, loss_fn, config, featurizer, logger: WanDBWriter, scheduler=None):
    model.train()

    for i, batch in enumerate(tqdm(iter(loader))):
        logger.set_step(logger.step + 1, mode='train')
        batch = batch.to(config['device'])
        batch.melspec = featurizer(batch.waveform)

        optimizer.zero_grad()

        batch = model(batch)

        loss = loss_fn(batch)

        loss.backward()
        optimizer.step()

        np_loss = loss.detach().cpu().numpy()

        logger.add_scalar("waveform_reconstruction_loss", np_loss)

        if i > config.get('len_epoch', 1e9):
            break
            
        if logger is not None and logger.step % config['log_train_step'] == 0:
            log_audios(batch, logger)
            logger.add_image("spectrogram_true", Image.open(plot_spectrogram_to_buf(batch.melspec[0].detach().cpu())))
            logger.add_image("spectrogram_pred", Image.open(plot_spectrogram_to_buf(featurizer(batch.waveform_prediction)[0].detach().cpu())))

        if scheduler is not None:
            scheduler.step()


def gan_train_epoch(
        G: Generator, D: Discriminator, optimizer_G, optimizer_D, loader, loss_fn,
        config, featurizer, scheduler_G=None, scheduler_D=None, logger: WanDBWriter = None
):
    G.train()
    D.train()
    g_steps = 0
    d_steps = 0

    for i, batch in enumerate(tqdm(iter(loader))):
        if logger is not None:
            logger.set_step(logger.step + 1, mode='train')

        batch = batch.to(config['device'], non_blocking=True)
        batch.melspec = featurizer(batch.waveform)

        batch = G(batch)

        set_require_grad(D, True)
        set_require_grad(G, False)

        def d_step(batch):
            nonlocal d_steps
            d_steps += 1
            optimizer_D.zero_grad()

            fake_loss, true_loss = loss_fn(batch, D.mpd, D.mcd, generator_step=False)
            loss = fake_loss + true_loss

            loss.backward()
            optimizer_D.step()

            if logger is not None:
                fake_loss_np = fake_loss.detach().cpu().numpy()
                true_loss_np = true_loss.detach().cpu().numpy()
                total_loss_np = loss.detach().cpu().numpy()

                logger.add_scalar("D_fake_loss", fake_loss_np)
                logger.add_scalar("D_true_loss", true_loss_np)
                logger.add_scalar("D_total_loss", total_loss_np)
                logger.add_scalar("D_steps", d_steps)

            return loss

        loss = d_step(batch)

        if config.get('discriminator_backprop_threshold') is not None:
            j = 0
            while loss.item() > config['discriminator_backprop_threshold']:
                loss = d_step(batch)

                j += 1
                if j > config['max_steps']:
                    break

        set_require_grad(D, False)
        set_require_grad(G, True)

        def g_step(batch):
            nonlocal g_steps
            g_steps += 1
            optimizer_G.zero_grad()

            gan_loss, fm_loss, reconstruction = loss_fn(batch, D.mpd, D.mcd, generator_step=True)
            loss = gan_loss + reconstruction + fm_loss
            loss.backward()
            optimizer_G.step()

            if logger is not None:
                gan_loss_np = gan_loss.detach().cpu().numpy()
                fm_loss_np = fm_loss.detach().cpu().numpy()
                reconstruction_np = reconstruction.detach().cpu().numpy()
                total_loss_np = loss.detach().cpu().numpy()

                logger.add_scalar("G_gan_loss", gan_loss_np)
                logger.add_scalar("G_reconstruction_loss", reconstruction_np)
                logger.add_scalar("G_FM_loss", fm_loss_np)
                logger.add_scalar("G_total_loss", total_loss_np)
                logger.add_scalar("G_steps", g_steps)

            return loss
        
        batch = G(batch)
        loss = g_step(batch)

        if config.get('generator_backprop_threshold') is not None:
            j = 0

            while loss.item() > config['generator_backprop_threshold']:
                batch = G(batch)

                loss = g_step(batch)

                j += 1
                if j > config['max_steps']:
                    break

        if logger is not None and logger.step % config['log_train_step'] == 0:
            log_audios(batch, logger)
            logger.add_image("spectrogram_true", Image.open(plot_spectrogram_to_buf(batch.melspec[0].detach().cpu())))
            logger.add_image("spectrogram_pred", Image.open(plot_spectrogram_to_buf(featurizer(batch.waveform_prediction)[0].detach().cpu())))

        if i > config.get('len_epoch', 1e9):
            break

        if scheduler_G is not None:
            if logger is not None:
                logger.add_scalar("G_learning rate", scheduler_G.get_last_lr()[0])

            scheduler_G.step()            

        if scheduler_D is not None:
            if logger is not None:
                logger.add_scalar("D_learning rate", scheduler_D.get_last_lr()[0])
            
            scheduler_D.step()


@torch.inference_mode()
def evaluate(model, loader, config, loss_fn, featurizer, metric_calculators=(), logger: WanDBWriter = None):
    model.eval()
    metrics = defaultdict(list)
    batches = []

    for i, batch in enumerate(tqdm(iter(loader))):
        if logger is not None:
            logger.set_step(logger.step + 1, mode='val')

        batch = batch.to(config['device'])
        batch.melspec = featurizer(batch.waveform)

        batch = model(batch)

        loss = loss_fn(batch)

        batches.append(batch.to('cpu'))

        metrics['reconstruction_loss'].append(loss.detach().cpu().numpy())

        if logger is not None and logger.step % config['log_val_step'] == 0:
            log_audios(batch, logger)

    for calc in metric_calculators:
        metrics.update(calc.calculate(batches))

    if logger is not None:
        for metric_name, metric_val in metrics.items():
            logger.add_scalar(metric_name, np.mean(metric_val))

    return metrics
