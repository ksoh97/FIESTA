import argparse, os, sys, datetime, importlib
os.environ['KMP_DUPLICATE_LIB_OK']='true'
import torch.optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from engine2 import train_warm_up,evaluate,train_one_epoch_SBF, train_one_epoch_SBF_v2,train_one_epoch
# from engine3 import train_warm_up,evaluate,train_one_epoch_SBF,train_one_epoch
from losses import SetCriterion
import numpy as np
import random
from torch.optim import lr_scheduler
import GPUtil
from torch.utils.tensorboard import SummaryWriter

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def seed_everything(seed=None):
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min
    try:
        if seed is None:
            seed = os.environ.get("PL_GLOBAL_SEED", random.randint(min_seed_value, max_seed_value))
        seed = int(seed)
    except (TypeError, ValueError):
        seed = random.randint(min_seed_value, max_seed_value)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f'training seed is {seed}')
    return seed

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-g",
        "--gpu_id",
        type=str,
        default="-1",
        help="GPU id",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=["configs/efficientUnet_SABSCT_to_CHAOS.yaml"], # CT->MRI (abdominal)
        # default=["configs/efficientUnet_CHAOS_to_SABSCT.yaml"], # MRI->CT (abdominal)
        # default=["configs/efficientUnet_bSSFP_to_LEG.yaml"], # bSSFP->LGE (cardiac)
        # default=["configs/efficientUnet_LEG_to_bSSFP.yaml"], # LGE->bSSFP (cardiac)
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,  # 42
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    return parser

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)  # module='main', cls='DataModuleFromConfig'
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls) # https://stackoverflow.com/questions/41678073/import-class-from-module-dynamically

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

class DataModuleFromConfig(torch.nn.Module):  # 요 함수 안에 self.dataset_configs들이 언제 initial되는지 모르겠음..
    def __init__(self, batch_size, train=None, validation=None, test=NoneW,
                 num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)  # k == train, validation, test

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers)

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    seed=seed_everything(opt.seed)
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
    if opt.base:
        cfg_fname = os.path.split(opt.base[0])[-1]
        cfg_name = os.path.splitext(cfg_fname)[0]
        name = "_" + cfg_name
    if opt.gpu_id:
        if not opt.gpu_id == '-1':
            devices = opt.gpu_id
        else:
            devices = "%s" % GPUtil.getFirstAvailable(order="memory")[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
    else:
        name=None
        raise ValueError('no config')

    nowname = now +f'_seed{seed}'+ name + opt.postfix + "_tesesr"
    # nowname = now +f'_seed{seed}'+ name + opt.postfix + "_train_one_epoch_SBF_v2_UG-E_epoch<1000=UGmix"
    # nowname = now +f'_seed{seed}'+ name + opt.postfix + "_train_one_epoch_SBF_v2_SetA_UG-D"
    # nowname = now +f'_seed{seed}'+ name + opt.postfix + "_NEW_FT(sect60)(GLA_SLAugLLA)-case1_set11-UG-blur"
    logdir = os.path.join("logs", nowname)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    visdir = os.path.join(logdir, "visuals")
    for d in [logdir, cfgdir, ckptdir, visdir]:
        os.makedirs(d, exist_ok=True)

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    OmegaConf.save(config,os.path.join(cfgdir, "{}-project.yaml".format(now)))

    model_config = config.pop("model", OmegaConf.create())
    optimizer_config = config.pop('optimizer', OmegaConf.create())

    SBF_config = config.pop('saliency_balancing_fusion', OmegaConf.create())

    model = instantiate_from_config(model_config)
    if torch.cuda.is_available():
        model=model.cuda()

    bs, lr = config.data.params.batch_size, optimizer_config.learning_rate
    param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad], "lr_scale": 1}]

    opt_params = {'lr': lr}
    for k in ['momentum', 'weight_decay']:
        if k in optimizer_config:
            opt_params[k] = optimizer_config[k]

    criterion = SetCriterion()

    print('optimization parameters: ', opt_params)
    opt = eval(optimizer_config['target'])(param_dicts, **opt_params) # Adam으로 optimizer 세팅

    if optimizer_config.lr_scheduler =='lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + 0 - 50) / float(optimizer_config.max_epoch-50 + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(opt, lr_lambda=lambda_rule)
    else:
        scheduler=None
        print('We follow the SSDG learning rate schedule by default, you can add your own schedule by yourself')
        raise NotImplementedError

    assert optimizer_config.max_epoch > 0 or optimizer_config.max_iter > 0
    if optimizer_config.max_iter > 0:
        max_epoch=999
        print('detect identified max iteration, set max_epoch to 999')
    else:
        max_epoch= optimizer_config.max_epoch

    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()

    print(f'The number of training samples: {len(data.datasets["train"])}')
    train_loader=DataLoader(data.datasets["train"], batch_size=data.batch_size,
                          num_workers=data.num_workers, shuffle=True, persistent_workers=True, drop_last=True, pin_memory = True)

    val_loader=DataLoader(data.datasets["validation"], batch_size=data.batch_size,  num_workers=1)

    if data.datasets.get('test') is not None:
        test_loader=DataLoader(data.datasets["test"], batch_size=1, num_workers=1)
        best_test_dice = 0
        test_phase=True
    else:
        test_phase=False

    if getattr(optimizer_config, 'warmup_iter'):
        if optimizer_config.warmup_iter>0:
            train_warm_up(model, criterion, train_loader, opt, torch.device('cuda'), lr, optimizer_config.warmup_iter)

    cur_iter=0
    best_dice=0
    label_name=data.datasets["train"].all_label_names
    train_writer, validation_writer, test_writer = SummaryWriter(logdir+"/Tensorboard/"), SummaryWriter(logdir+"/Tensorboard/"), SummaryWriter(logdir+"/Tensorboard/")

    for cur_epoch in range(max_epoch):
        print("Current setup title: %s" % nowname)
        if SBF_config.usage:
            cur_iter, loss_dict = train_one_epoch_SBF_v2(model, criterion, train_loader, opt, torch.device('cuda'), cur_epoch, cur_iter, optimizer_config.max_iter, SBF_config, visdir)
            train_writer.add_scalar("Train/CE_loss", loss_dict["ce_loss"], cur_epoch)
            train_writer.add_scalar("Train/DICE_loss", loss_dict["dice_loss"], cur_epoch)
        else:
            print("Normal train_one_epoch")
            cur_iter = train_one_epoch(model, criterion, train_loader, opt, torch.device('cuda'), cur_epoch, cur_iter, optimizer_config.max_iter)
        if scheduler is not None:
            scheduler.step()

        # Save latest model
        if (cur_epoch+1)%50==0:
            torch.save({'model': model.state_dict()}, os.path.join(ckptdir,'latest.pth'))

            # Check the saved model in unseen test data
            cur_dice = evaluate(model, test_loader, torch.device('cuda'))
            str=f'Epoch [{cur_epoch}]   '
            for i,d in enumerate(cur_dice):
                str+=f'Class {i}: {d}, '
            str += f'Test DICE {np.mean(cur_dice)}'
            print(str)

        if cur_iter >= optimizer_config.max_iter and optimizer_config.max_iter>0:
            torch.save({'model': model.state_dict()}, os.path.join(ckptdir, 'latest.pth'))
            print(f'End training with iteration {cur_iter}')
            break

    train_writer.close()
    validation_writer.close()
    test_writer.close()