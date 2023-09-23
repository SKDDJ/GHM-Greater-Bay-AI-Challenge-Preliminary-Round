import os
def save_lora(self,path):
        ## save lora weights 
        os.makedirs(path, exist_ok=True)
        lora_state={}
        for name,param in self.nnet.named_parameters():
            if 'lora' in name:
                lora_state[name]=param
        
        torch.save(lora_state,os.path.join(path,'lora.pt.tmp'))
        os.replace(os.path.join(path,'lora.pt.tmp'),os.path.join(path,'lora.pt'))