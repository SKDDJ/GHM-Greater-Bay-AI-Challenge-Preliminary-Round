import os
import json
import glob
import numpy as np
import torch
import clip
from PIL import Image
import argparse
import warnings
from score_utils.face_model import FaceAnalysis

warnings.filterwarnings("ignore")

class Evaluator():
    def __init__(self):
        self.clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.clip_device)
        self.clip_tokenizer = clip.tokenize

        self.face_model = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.face_model.prepare(ctx_id=0, det_size=(640, 640))

    def pil_to_cv2(self, pil_img):
        return np.array(pil_img)[:,:,::-1]
    
    def get_face_embedding(self, img):
        """ get face embedding
        """
        if type(img) is not np.ndarray:
            img = self.pil_to_cv2(img)
            
        faces = self.face_model.get(img, max_num=1) ## only get first face
        if len(faces) <= 0:
            return None
        else:
            emb = torch.Tensor(faces[0]['embedding']).unsqueeze(0)
            emb /= emb.norm(dim=-1, keepdim=True)
            return emb
            

    def sim_face(self, img1, img2):
        """ 
        calcualte face similarity using insightface
        """
        if type(img1) is not np.ndarray:
            img1 = self.pil_to_cv2(img1)
        if type(img2) is not np.ndarray:
            img2 = self.pil_to_cv2(img2)
            
        feat1 = self.get_face_embedding(img1)
        feat2 = self.get_face_embedding(img2)
        
        if feat1 is None or feat2 is None:
            return 0
        else:
            similarity = feat1 @ feat2.T
            return max(0,similarity.item())
        
    def sim_face_emb(self, img1, embs):
        """ 
        calcualte face similarity using insightface
        """
        if type(img1) is not np.ndarray:
            img1 = self.pil_to_cv2(img1)
            
        feat1 = self.get_face_embedding(img1)
        
        if feat1 is None:
            return 0
        else:
            similarity = feat1 @ embs.T
            return max(0,similarity.max().item())
    
    def get_img_embedding(self, img):
        """ 
        get clip image embedding
        """
        x = self.clip_preprocess(img).unsqueeze(0).to(self.clip_device)
        with torch.no_grad():
            feat = self.clip_model.encode_image(x)
        feat /= feat.norm(dim=-1, keepdim=True)
        return feat

    def get_text_embedding(self, text):
        """ 
        get clip image embedding
        """
        x = self.clip_tokenizer([text]).to(self.clip_device)
        with torch.no_grad():
            feat = self.clip_model.encode_text(x)
        feat /= feat.norm(dim=-1, keepdim=True)
        return feat
        
    
    def sim_clip_img(self, img1, img2):
        """ 
        calcualte img img similarity using CLIP
        """
        feat1 = self.get_img_embedding(img1)
        feat2 = self.get_img_embedding(img2)
        similarity = feat1 @ feat2.T
        return max(0,similarity.item())
    
    def sim_clip_imgembs(self, img, embs):
        feat = self.get_img_embedding(img)
        similarity = feat @ embs.T
        return max(0,similarity.max().item())
        
    def sim_clip_text(self, img, text):
        """ 
        calcualte img text similarity using CLIP
        """
        feat1 = self.get_img_embedding(img)
        feat2 = self.get_text_embedding(text)
        similarity = feat1 @ feat2.T
        return max(0,similarity.item())
    
    
    def score1_gen_vs_img_face(self, gen, img, alpha_img=0.5, alpha_face=0.5):
        img_sim = self.sim_clip_img(gen,img)
        face_sim = self.sim_face(gen, img)
        
        return alpha_img * img_sim + alpha_face * face_sim
    
    def score2_gen_vs_img(self, gen, img, alpha_img=1.0):
        img_sim = self.sim_clip_img(gen,img)
                
        return alpha_img * img_sim
    
    def score3_gen_vs_text(self, gen, text, alpha_text=1.0):
        text_sim = self.sim_clip_text(gen,text)
        return alpha_text * text_sim
        
    def score4_gen_vs_text_refimg(self, gen, text, ref, alpha_text=0.5, alpha_img=0.5):
        text_sim = self.sim_clip_text(gen,text)
        img_sim = self.sim_clip_img(gen, ref)
    
        return alpha_text * text_sim + alpha_img * img_sim
    
def read_img_pil(p):
    return Image.open(p)

def score(dataset_base, prompts_base, outputs_base,i):
    eval = Evaluator()
    # DATANAMES = ["boy1", "boy2", "girl1", "girl2"]
    # SIM_TASKNAMES = ['boy1_sim', 'boy2_sim', 'girl1_sim', 'girl2_sim']
    # EDIT_TASKNAMES = ['boy1_edit', 'boy2_edit', 'girl1_edit', 'girl2_edit']
    DATANAMES = ["boy1"]
    SIM_TASKNAMES = ['boy1_sim']
    EDIT_TASKNAMES = ['boy1_edit']
    ## folder check
    for taskname in DATANAMES:
        task_dataset = os.path.join(dataset_base, f'{taskname}')
        assert os.path.exists(task_dataset), f"Missing Dataset folder: {task_dataset}"
    for taskname in SIM_TASKNAMES + EDIT_TASKNAMES:
        task_prompt = os.path.join(prompts_base, f'{taskname}.json')
        assert os.path.exists(task_prompt), f"Missing Prompt file: {task_prompt}"
        if i == -1:
            task_output = os.path.join(outputs_base, f'{taskname}')
        else:
            task_output = os.path.join(outputs_base, f'{taskname}_{i*500}')
        assert os.path.exists(task_output), f"Missing Output folder: {task_output}"
        
    def score_task(sample_folder, dataset_folder, prompt_json):
        ## get prompt, face, and ref image from dataset folder
        refs = glob.glob(os.path.join(dataset_folder, "*.jpg")) + glob.glob(os.path.join(dataset_folder, "*.jpeg"))
        refs_images = [read_img_pil(ref) for ref in refs]
        
        refs_clip = [eval.get_img_embedding(i) for i in refs_images]
        refs_clip = torch.cat(refs_clip)
        #### print(refs_clip.shape)
        
        refs_embs = [eval.get_face_embedding(i) for i in refs_images]
        refs_embs = [emb for emb in refs_embs if emb is not None]
        refs_embs = torch.cat(refs_embs)
        #### print(refs_embs.shape)
        
        #### print("Ref Count: ", len(refs_images))
        #### print("Emb: ", refs_embs.shape)
        
        pompt_scores = []
        prompts = json.load(open(prompt_json, "r"))
        for prompt_index, prompt in enumerate(prompts):
            sample_scores = []
            for idx in range(0,3): ## 3 generation for each prompt
                sample_path = os.path.join(sample_folder,f"{prompt_index}-{idx:03}.jpg") ## for face / target reference
                try:
                    sample = read_img_pil(sample_path)
                    
                    # sample vs ref
                    score_face = eval.sim_face_emb(sample, refs_embs)
                    score_clip = eval.sim_clip_imgembs(sample, refs_clip)
                    # sample vs prompt
                    score_text = eval.sim_clip_text(sample, prompt)
                    sample_score = [score_face, score_clip, score_text]
                    print(sample_path,"score_face",score_face,"score_clip",score_clip)
                except Exception as e:
                    #### print(e)
                    sample_score = [0.0, 0.0, 0.0]
                #### print(f"Score for sample {idx}: ", sample_score)
                sample_scores.append(sample_score)
            pompt_score = np.mean(sample_scores, axis=0)
            #### print(f"Score for prompt {prompt_index}: ", pompt_score)
            pompt_scores.append(pompt_score)
        task_score = np.mean(pompt_scores, axis=0)
        return task_score
    
    ## calculate sim score
    sim_scores = []
    for dataname, taskname in zip(DATANAMES, SIM_TASKNAMES):
        task_dataset = os.path.join(dataset_base, f'{dataname}')
        task_prompt = os.path.join(prompts_base, f'{taskname}.json')
        if i == -1:
            task_output = os.path.join(outputs_base, f'{taskname}')
        else:
            task_output = os.path.join(outputs_base, f'{taskname}_{i*500}')
        score = score_task(task_output, task_dataset, task_prompt)
        print(f"Score for task {taskname}: ", score)
        sim_scores.append(score)
    print(sim_scores)
    sim_ave_score = np.mean(sim_scores, axis=0)
    
    edit_scores = []
    for dataname, taskname in zip(DATANAMES, EDIT_TASKNAMES):
        task_dataset = os.path.join(dataset_base, f'{dataname}')
        if i == -1:
            task_output = os.path.join(outputs_base, f'{taskname}')
        else:
            task_output = os.path.join(outputs_base, f'{taskname}_{i*500}')
     
        score = score_task(task_output, task_dataset, task_prompt)
        print(f"Score for task {taskname}: ", score)
        edit_scores.append(score)
    print(edit_scores)
    edit_ave_score = np.mean(edit_scores, axis=0)
    
    score_dict = {
        "复现功能的人脸相似度": sim_ave_score[0],
        "复现功能的CLIP图片相似度": sim_ave_score[1],
        "编辑功能的人脸相似度": edit_ave_score[0],
        "编辑功能的CLIP图片相似度": edit_ave_score[1],
        "编辑功能的图文匹配度": edit_ave_score[2],
    }
    print(f"\033[91m 最终结果:\n{score_dict}\033[00m")
    return score_dict
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation Script')
    parser.add_argument('--dataset', type=str, default='./train_data/', help='dataset folder')
    parser.add_argument('--prompts', type=str, default='./eval_prompts/', help='prompt folder')
    parser.add_argument('--outputs', type=str, default='./outputs/', help='output folder')

    args = parser.parse_args()
    for i in range(4,9):
        eval_score = score(args.dataset, args.prompts, args.outputs,i)
        print(eval_score)
    
    # eval_score = score(args.dataset, args.prompts, args.outputs,i=5)
    # print(eval_score)