import torch,tqdm

def Testing(test_dataloader,loss_fn,devices,model,number_size,initial_context,initial_decoder_input):
    
    initial_context=initial_context.to(devices)
    initial_decoder_input=initial_decoder_input.to(devices)
    
    with torch.no_grad():
        loss_total=0.0
        total_size=0.0
        true_test=0.0

        
        pb=tqdm.tqdm(range(len(test_dataloader)),"Test Progress")
        for batch,(input_encoder,target_decoder) in enumerate(test_dataloader):
            
            input_encoder=input_encoder.to(devices)
            target_decoder=target_decoder.to(devices)
            
            out_list=model(input_encoder,initial_context,initial_decoder_input)
            
            loss=loss_fn(out_list.reshape(-1,number_size+1),target_decoder.reshape(-1))
            loss_total+=loss.item()
            
            _,pred_test=torch.max(out_list,2)
            total_size+=(target_decoder.size(0)*target_decoder.size(1))
            true_test+=(pred_test==target_decoder).sum().item()

            pb.update(1)
        
        pb.close()
        print(f"\nLoss_Test: {loss_total/(batch+1)}    Acc_Test: {100*(true_test/total_size)}")



def Prediction(tensor,model,batch_size,max_len,
               initial_context,initial_decoder_input):
    
    with torch.no_grad():
        
        model.cpu()
        pad = torch.zeros((max_len-len(tensor),),dtype=int)
        tensor_pad=torch.concat([tensor,pad])
        rep_tensor = tensor_pad.repeat(batch_size, 1)
        
        out_list=model(rep_tensor,initial_context,initial_decoder_input)
        
        _,pred=torch.max(out_list,-1)
        
        print("Tensor: ",tensor.tolist())
        print("Prediction: ",pred[0][:len(tensor)].tolist())
        
    return pred
            