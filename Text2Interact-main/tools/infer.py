import copy
import os.path
import sys
sys.path.append(sys.path[0] + r"/../")
import torch
import lightning as L
import scipy.ndimage.filters as filters

from os.path import join as pjoin
from models import *
from collections import OrderedDict
from configs import get_config
from utils.plot_script import *
from utils.preprocess import *
from utils import paramUtil


class LitGenModel(L.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        # cfg init
        self.cfg = cfg

        self.automatic_optimization = False

        # train model init
        self.model = model

        # others init
        self.normalizer = MotionNormalizer()

    def plot_t2m(self, mp_data, result_path, caption):
        mp_joint = []
        for i, data in enumerate(mp_data):
            if i == 0:
                joint = data[:,:22*3].reshape(-1,22,3)
            else:
                joint = data[:,:22*3].reshape(-1,22,3)

            mp_joint.append(joint)

        plot_3d_motion(result_path, paramUtil.t2m_kinematic_chain, mp_joint, title=caption, fps=30)


    def generate_one_sample(self, prompt, name):
        self.model.eval()

        seed_everything(37)
        batch = OrderedDict()
        batch["motion_lens"] = torch.zeros(1,1).long().cuda()
        batch["prompt"]      = prompt
        seq = self.generate_loop(batch, 210)
        print(f"prompt: {prompt}")

        os.makedirs("results", exist_ok=True)
        save_path = f"results/{name}.mp4"
        self.plot_t2m([seq[0], seq[1]],
                save_path, prompt)
        print(f"saved motion to {save_path}")

    def generate_loop(self, batch, window_size):
        prompt = batch["prompt"]
        batch = copy.deepcopy(batch)
        batch["motion_lens"][:] = window_size

        sequences = [[], []]

        batch["text"] = [prompt]
        batch = self.model.forward_test(batch)
        motion_output_both = batch["output"][0].reshape(batch["output"][0].shape[0], 2, -1)
        motion_output_both = self.normalizer.backward(motion_output_both.cpu().detach().numpy())


        for j in range(2):
            motion_output = motion_output_both[:,j]

            joints3d = motion_output[:,:22*3].reshape(-1,22,3)
            joints3d = filters.gaussian_filter1d(joints3d, 1, axis=0, mode='nearest')
            sequences[j].append(joints3d)


        sequences[0] = np.concatenate(sequences[0], axis=0)
        sequences[1] = np.concatenate(sequences[1], axis=0)
        return sequences

def build_models(cfg):
    if cfg.NAME == "Text2Interact":
        model = Text2Interact(cfg)
    return model


def seed_everything(SEED):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import random
    import numpy as np
    import torch


    import argparse
    parser = argparse.ArgumentParser(description="Text2Interact")
    parser.add_argument("--cfg", type=str, default="configs/model-finetune-0.35.yaml", help="config file")
    args = parser.parse_args()

    model_cfg = get_config(args.cfg)
    infer_cfg = get_config("configs/infer.yaml")

    model = build_models(model_cfg)

    if model_cfg.CHECKPOINT:
        ckpt = torch.load(model_cfg.CHECKPOINT, map_location="cpu")
        for k in list(ckpt["state_dict"].keys()):
            if "model" in k:
                ckpt["state_dict"][k.replace("model.", "")] = ckpt["state_dict"].pop(k)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print("checkpoint state loaded!")

    litmodel = LitGenModel(model, infer_cfg).to(torch.device("cuda:0"))

    prompt_file = "./custom_prompts.txt"
    with open(prompt_file) as f:
        texts = f.readlines()
    texts = [text.strip("\n") for text in texts]

    for text in texts:
        name = text[:48]
        litmodel.generate_one_sample(text, name)

