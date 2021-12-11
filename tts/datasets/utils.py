from torch.utils.data import DataLoader

from tts.config import Config


def get_data_loader(config: Config, dataset, collate_fn, mode='train'):
    data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        shuffle=mode == 'train',
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=mode == 'train'
    )

    return data_loader
