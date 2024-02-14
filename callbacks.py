import torch

def save_checkpoint(epoch,optimizers,Model,PATH_CHECKPOİNT="Sign_Lenguage_callbacks.pth.tar"):
    print("Checkpoint is Saving...")
    
    checkpoint={"Epoch":epoch,
                "Optimizer_state_dict": optimizers.state_dict(),
                "Model_state_dict": Model.state_dict()}
    
    torch.save(checkpoint,f=PATH_CHECKPOİNT)
    
    
    
def load_checkpoint(Checkpoint,optimizer,model):
    
    model.load_state_dict(Checkpoint["Model_state_dict"])
    optimizer.load_state_dict(Checkpoint["Optimizer_state_dict"])
    start_epoch=Checkpoint["Epoch"]
    
    return start_epoch