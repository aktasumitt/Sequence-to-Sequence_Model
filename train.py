import torch
import tqdm


def Train(RESUME_EPOCHS,EPOCHS,train_dataloader,valid_dataloader,model,loss_fn,optimizer,number_size,devices,save_callbaks_fn,path_checkpoint,initial_context,initial_decoder_input):
    
    initial_context=initial_context.to(devices)
    initial_decoder_input=initial_decoder_input.to(devices)

    for epoch in range(RESUME_EPOCHS,EPOCHS):
        
        loss_train = 0.0
        true_train = 0.0
        total_train = 0.0

        prog_bar = tqdm.tqdm(range(len(train_dataloader)), desc=f"Epoch {epoch+1}")

        for batch, (input_encoder, input_decoder) in enumerate(train_dataloader):
            input_encoder = input_encoder.to(devices)
            input_decoder = input_decoder.to(devices)

            optimizer.zero_grad()
            out_list = model(input_encoder,initial_context,initial_decoder_input)
            loss = loss_fn(out_list.reshape(-1, number_size+1), input_decoder.reshape(-1))
            loss.backward()
            optimizer.step()

            _,pred_t=torch.max(out_list,2)
            true_train+=(pred_t==input_decoder).sum().item()
            loss_train += loss.item()
            total_train+= (input_decoder.size(0)*input_decoder.size(1))

            # Validation
            if batch % 20 == 19:
            
                loss_valid = 0.0
                true_valid = 0.0
                total_val = 0.0

                with torch.no_grad():
                    for batch_v, (input_encoder_val, input_decoder_val) in enumerate(valid_dataloader):
                        input_encoder_val = input_encoder_val.to(devices)
                        input_decoder_val = input_decoder_val.to(devices)

                        out_list_v = model(input_encoder_val,initial_context,initial_decoder_input)
                        
                        loss_v = loss_fn(out_list_v.reshape(-1, number_size+1), input_decoder_val.reshape(-1))
                        
                        _,pred=torch.max(out_list_v,2)
                        true_valid+=(pred==input_decoder_val).sum().item()
                        total_val+=(input_decoder_val.size(0)*input_decoder_val.size(1))

                        loss_valid += loss_v.item()

                prog_bar.set_postfix_str(f"Loss Train: {loss_train / (batch+1):.4f}, Loss Val: {loss_valid / (batch_v+1):.4f}, Acc Train: {(100*(true_train / total_train)):.4f}, Acc Val: {(100*(true_valid / total_val)):.4f}")
            
            prog_bar.update(1)
            
        prog_bar.close()
        
        save_callbaks_fn(epoch,optimizer,model,PATH_CHECKPOİNT=path_checkpoint)
