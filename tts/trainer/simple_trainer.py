from tqdm.autonotebook import tqdm
import torch
from tts.logger.wandb import WanDBWriter


def train_epoch(model, optimizer, loader, scheduler, loss_fn, config, featurizer, logger: WanDBWriter):
    model.train()

    for i, batch in enumerate(tqdm(iter(loader))):
        logger.set_step(logger.step + 1, mode='train')
        batch = batch.to(config['device'])
        batch.melspec = featurizer(batch.waveform)

       # batch.melspec_length = batch.melspec.shape[-1] - batch.melspec.eq(-11.5129251)[:, 0, :].sum(dim=-1)

        optimizer.zero_grad()

        batch = model(batch)

        loss = loss_fn(batch)

        loss.backward()
        optimizer.step()

        np_loss = loss.detach().cpu().numpy()

        logger.add_scalar("waveform_reconstruction_loss", np_loss)

        if i > config.get('len_epoch', 1e9):
            break

        scheduler.step()


@torch.no_grad()
def evaluate(model, loader, config, vocoder, logger: WanDBWriter):
    model.eval()

    for batch in tqdm(iter(loader)):
        batch = batch.to(config['device'])

        batch = model(batch)

        for i in range(batch.melspec_prediction.shape[0]):
            logger.set_step(logger.step + 1, "val")

            reconstructed_wav = vocoder.inference(batch.melspec_prediction[i:i + 1].transpose(-1, -2)).cpu()

            logger.add_text("text", batch.transcript[i])
            logger.add_audio("audio", reconstructed_wav, sample_rate=22050)

