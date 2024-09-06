import wandb
import logging
import os

import gin
import pytorch_lightning as pl
import torch
from absl import app, flags
from torch.utils import data
import GaMaDHaNi
from GaMaDHaNi.src.protobuf.data_example import AudioExample
from GaMaDHaNi.src.dataset import SequenceDataset
from pytorch_lightning.plugins.environments import SLURMEnvironment

import wandb
import time
import pdb

FLAGS = flags.FLAGS
flags.DEFINE_multi_string("config",
                          default="decoder_only",
                          help="config to parse.")
flags.DEFINE_string("db_path",
                    default=None,
                    help="path to dataset.")
flags.DEFINE_integer("val_size",
                     default=8192,
                     help="size of validation dataset.")
flags.DEFINE_integer("batch_size", default=64, help="batch size.")
flags.DEFINE_string("name", default=None, required=False, help="train name.")
flags.DEFINE_integer("gpu", default=0, help="gpu index.")
flags.DEFINE_integer("workers",
                     default=0,
                     help="num workers during data loading.")
flags.DEFINE_string("pretrained_embedding",
                    default=None,
                    help="use pretrained embeddings from rave.")
flags.DEFINE_multi_string("override",
                          default=[],
                          help="additional gin bindings.")
flags.DEFINE_string("ckpt",
                    default=None,
                    help="checkpoint to resume training from.")
flags.DEFINE_float('ema',
                   default=None,
                   help='Exponential weight averaging factor (optional)')
flags.DEFINE_integer("val_every",
                     default=10,
                     help="validate training every n epochs.")
flags.DEFINE_integer("checkpoint_model_every",
                     default=10,
                     help="checkpoint model every n epochs.")
flags.DEFINE_string("data_art",
                    default=None,
                    help="data artifact name on wandb.")
flags.DEFINE_bool("split",
                    default=False,
                    help="If true, will randomly split the dataset into train and validation sets.")
flags.DEFINE_bool("debug",
                    default=False,
                    help="If true will run in debug mode")
flags.DEFINE_string("group",
                    default="debug",
                    help="wandb group")
flags.DEFINE_integer("max_epochs",
                     default=1000,
                     help="Maximum number of epochs to train for.")
flags.DEFINE_string("id_to_checkpoint_from",
                    default=None,
                    help="SLURM id to continue from. If None, it is ignored.")
flags.DEFINE_string("run_name",
                    default=None,
                    help="Run name to add on wandb. If None, will default to run_id (SLURM JOB ID)")
flags.DEFINE_string("wandb_notes",
                    default=None,
                    help="Notes to add to wandb (optional).")
flags.DEFINE_bool("log_to_wandb",
                    default=None,
                    help="Option to automatically log to wandb. By default is true when debug is false else false.")
flags.DEFINE_string("wandb_id",
                    default=None,
                    help="Option to allow user to explicitly enter wandb id, in the case that wandb id != slurm id.")

def add_ext(config: str):
    if config[-4:] != ".gin":
        config += ".gin"
    return config


def main(argv):
    
    if FLAGS.id_to_checkpoint_from is not None:
        run_id = FLAGS.id_to_checkpoint_from
    else:
        run_id = os.environ.get('SLURM_ARRAY_JOB_ID') if 'SLURM_ARRAY_JOB_ID' in os.environ else os.environ.get('SLURM_JOB_ID')
        if FLAGS.debug:
            run_id += '_' + str(int(time.time()))
    if FLAGS.wandb_id is not None:
        wandb_id = FLAGS.wandb_id
    else:
        wandb_id = run_id
    if FLAGS.run_name is not None:
        run_name = FLAGS.run_name
    else:
        run_name = run_id
    tmp = os.path.join(os.environ.get('SLURM_TMPDIR'), run_id)
    os.makedirs(tmp, exist_ok=True)
    if FLAGS.ckpt is None:
        checkpoint_folder = os.path.join(os.environ.get('SCRATCH'), 'checkpoints', 'msprior', FLAGS.group, run_id)
        os.makedirs(checkpoint_folder, exist_ok=True)
        FLAGS.ckpt = checkpoint_folder  # add checkpoint to config
    elif not os.path.exists(FLAGS.ckpt):
        raise Exception('Checkpoint path does not exist')
    else:
        checkpoint_folder = FLAGS.ckpt

    if FLAGS.log_to_wandb is None:
        if FLAGS.debug:
            log_to_wandb = False
        else:
            log_to_wandb = True
    else:
        log_to_wandb = FLAGS.log_to_wandb

    if log_to_wandb:
        wandb_run = wandb.init(project='msprior', dir=tmp, config={key: value.value for (key, value) in FLAGS.__dict__['__flags'].items()}, job_type='train', resume='allow' if not FLAGS.debug else 'never', group=FLAGS.group, id=wandb_id, name=run_name, notes = FLAGS.wandb_notes)

        # log slurm id to wandb
        if 'SLURM ID' in wandb_run.config.keys():
            slurm_ids = wandb_run.config['SLURM ID']
            if isinstance(slurm_ids, list):
                slurm_ids.append(run_id)
            else:
                slurm_ids = [slurm_ids]
                slurm_ids.append(run_id)
            wandb_run.config.update({
                'SLURM ID': run_id
            }, allow_val_change=True)
        else:
            wandb_run.config.update({
                'SLURM ID': run_id
            })

    overrides = FLAGS.override
    if FLAGS.pretrained_embedding is not None:
        overrides.append(f"PRETRAINED_RAVE='{FLAGS.pretrained_embedding}'")

    logging.info("parsing configuration")
    configs = list(map(add_ext, FLAGS.config))
    gin.parse_config_files_and_bindings(
        configs,
        overrides,
    )
    

    logging.info("loading dataset")
    if FLAGS.data_art is not None:
        dataset = wandb_run.use_artifact(FLAGS.data_art + ":latest").download(root=os.path.join(tmp, 'data'))
        FLAGS.db_path = os.path.join(dataset)
    elif FLAGS.db_path is not None:
        dataset = FLAGS.db_path
    else:
        raise ValueError("Must provide either data_artifact or data")

    if FLAGS.split:
        dataset = SequenceDataset(db_path=FLAGS.db_path)

        if FLAGS.val_size > len(dataset):
            logging.warn(
                r"Dataset too small, using 5% of the train set as the val set")
            FLAGS.val_size = len(dataset) // 20

        train, val = data.random_split(
            dataset,
            (len(dataset) - FLAGS.val_size, FLAGS.val_size),
            generator=torch.Generator().manual_seed(42),
        )
    else:
        train = SequenceDataset(db_path=os.path.join(FLAGS.db_path, 'train'))
        val = SequenceDataset(db_path=os.path.join(FLAGS.db_path, 'val'))
        
        # inps = [ind for ind in range(len(train)) if not train[ind]]
        #this still returned items with dict objects and not None; so figure if anything at all is getting dropped
        # first check if anything at all gets out of range first of all and then see f the original range and dropped range makes sense
        # print(len(inps))

        # train, val have keys encoder_inputs, decoder_inputs, decoder_targets
        # encoder_inputs - empty
        # decoder_inputs - (256, 16): discrete code indices, number of quantizers
        # decoder_targets - (256, 16): target discrete code indices, number of quantizers (decoder_inputs[ind + 1, :] = decoder_targets[ind, :])

    # if not any(map(lambda x: "flattened" in x, FLAGS.config)):
    #     logging.info("quantizer number retrieval")
    #     # import pdb; pdb.set_trace()
    #     with gin.unlock_config():
    #         gin.parse_config(
    #             f"NUM_QUANTIZERS={train[0]['decoder_inputs'].shape[-1]}") # updates NUM_QUANTIZERS based on the data shape :)
    
    # logging.info("building model")
    # # model = Prior()
    # model = XTransformerPrior()

    train_loader = data.DataLoader(
        train,
        batch_size=FLAGS.batch_size,
        shuffle=True, 
        drop_last=True,
        num_workers=FLAGS.workers,
    )
    val_loader = data.DataLoader(
        val,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=FLAGS.workers,
    )
    temp = iter(val_loader)
    batch = next(temp)
    print(batch)
    # # pdb.set_trace()
    # with open(os.path.join(checkpoint_folder, "config.gin"),
    #           "w") as config_out:
    #     config_out.write(gin.config_str())

    # val_check = {}
    # # if len(train_loader) >= FLAGS.val_every:
    # #     val_check["val_check_interval"] = FLAGS.val_every
    # # else:
    # # nepoch = FLAGS.val_every // len(train_loader)
    # val_check["check_val_every_n_epoch"] = FLAGS.val_every

    # last_checkpoint = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(checkpoint_folder, 'models'), 
    # filename="checkpoint-{epoch:02d}-{val_cross_entropy:.2f}-{cross_entropy:.2f}",
    # save_on_train_epoch_end=True,
    # every_n_epochs=FLAGS.checkpoint_model_every,
    # save_last=True,
    # save_top_k=-1)

    # class DebuggingCallback(pl.Callback):
    #     def __init__(self, batch_to_debug):
    #         super().__init__()
    #         self.batch_to_debug = batch_to_debug

    #     def on_train_batch_start(self, trainer, pl_module):
    #         if trainer.batch_idx == self.batch_to_debug:
    #             print(f"Batch {self.batch_to_debug} started. Entering debug mode.")
    #             pdb.set_trace()

    # # Set the batch number you want to debug
    # # batch_to_debug = 7  # Update with the batch that's causing the error
    # # callbacks = [DebuggingCallback(batch_to_debug)]

    # callbacks = [
    #     pl.callbacks.ModelCheckpoint(monitor="val_cross_entropy",
    #                                  filename='best',
    #                                  dirpath=os.path.join(checkpoint_folder, 'models')),
    #     last_checkpoint,
    #     pl.callbacks.EarlyStopping(
    #         "val_cross_entropy",
    #         patience=20,
    #     )
    # ]

    # if FLAGS.ema is not None:
    #     callbacks.append(msprior.utils.EMA(FLAGS.ema))
    
    # logging.info("creating trainer")
    # if log_to_wandb:
    #     callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='step'))
    #     wandb_logger = pl.loggers.WandbLogger(
    #         project="msprior",
    #         save_dir=os.path.join(tmp, f'wandb-{wandb_run.name}'),
    #         offline=False,
    #         id=wandb_run.id
    #     )
    # else:
    #     wandb_logger = None
    # trainer = pl.Trainer(
    #     logger=wandb_logger,
    #     accelerator='gpu',
    #     devices=[FLAGS.gpu],
    #     callbacks=callbacks,
    #     log_every_n_steps=10,
    #     **val_check,
    #     max_epochs=FLAGS.max_epochs,
    #     plugins=[SLURMEnvironment(auto_requeue=True)],
    #     profiler="simple"
    # )

    # torch.backends.cudnn.benchmark = True
    # torch.set_float32_matmul_precision('high')

    # logging.info("launch training")
    # # pdb.set_trace()
    # run = search_for_run(checkpoint_folder)
    # if run is not None:
    #     step = torch.load(run, map_location='cpu')["global_step"]
    #     trainer.fit_loop.epoch_loop._batches_that_stepped = step
        
    
    # trainer.fit(
    #     model,
    #     train_loader,
    #     val_loader,
    #     ckpt_path=run,
    # )

    # if log_to_wandb:
    #     wandb.finish()
if __name__ == "__main__":
    app.run(main)
