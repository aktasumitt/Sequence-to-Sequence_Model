import torch
from torch.utils.data import DataLoader,random_split,Dataset


def Create_Custom_Data(max_len,number_size,feature_size):             
    total_list=[]
    for i in range(feature_size):
        input_data=torch.randint(1,number_size,(torch.randint(1,max_len,(1,)).item(),),dtype=torch.int)
        total_data=(input_data,torch.flip(input_data,dims=[-1]))
        total_list.append(total_data)

    return total_list


def Padding(max_len,data):
    out_list_in=[]
    out_list_out=[]
    for in_data,out_data in data:
        pad=torch.zeros((max_len-len(in_data)),dtype=torch.int)
        new_indata=torch.cat([in_data,pad])
        new_outdata=torch.cat([out_data,pad])
        
        out_list_in.append(new_indata)
        out_list_out.append(new_outdata)
    
    return (torch.stack(out_list_in),torch.stack(out_list_out))




class DATASET(Dataset):
    def __init__(self,data):
        super(DATASET,self).__init__()
        self.input_encoder,self.target_decoder=data    
    
    def __len__(self):
        return len(self.input_encoder)
    
    def __getitem__(self,index):
        
        return (self.input_encoder[index],self.target_decoder[index].long())
    
    
def random_split_fn(data,valid_div):
    
    valid_size=int(len(data)*valid_div)
    train,valid,test=random_split(data,(len(data)-valid_size*2,valid_size,valid_size))
    
    return train,valid,test

def Create_dataloader(train=None,valid=None,test=None,batch_size:int=None):
    if train!=None:
        train_dataloader=DataLoader(train,batch_size=batch_size,shuffle=True)
    if valid!=None:
        valid_dataloader=DataLoader(valid,batch_size=batch_size,shuffle=False)
    if test!=None:
        test_dataloader=DataLoader(test,batch_size=batch_size,shuffle=False)
        
    return train_dataloader,valid_dataloader,test_dataloader