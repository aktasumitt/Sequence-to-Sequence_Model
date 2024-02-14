import torch,dataset,config,train,model,test,callbacks

devices=("cuda" if torch.cuda.is_available() else "cpu")


# Create Custom Data
data=dataset.Create_Custom_Data(max_len=config.MAX_LEN,number_size=config.NUMBER_SIZE,feature_size=config.FEATURE_SIZE)


# Padding data to max length
data_pad=dataset.Padding(data=data,max_len=config.MAX_LEN)


# Create Dataset
datasets=dataset.DATASET(data_pad)


# Random_split for valid and test dataset
train_dataset,valid_dataset,test_dataset=dataset.random_split_fn(datasets,valid_div=config.VALID_DIV)


# Dataloader
train_dataloader,valid_dataloader,test_dataloader=dataset.Create_dataloader(train=train_dataset,
                                                                            valid=valid_dataset,
                                                                            test=test_dataset,
                                                                            batch_size=config.BATCH_SIZE)

# Create model, optimizer and loss_fn
Model=model.Seq2Seq(label_size=config.NUMBER_SIZE,
                    hidden_unit=config.HIDDEN_UNIT,
                    max_len=config.MAX_LEN,
                    batch_Size=config.BATCH_SIZE).to(devices)


optimizer=torch.optim.Adam(params=Model.parameters(),lr=config.LEARNING_RATE)
loss_fn=torch.nn.CrossEntropyLoss()


# Loading Checkpoint If You Have
if config.LOAD_CHECKPOINT==True:
    
    checkpoint=torch.load(f=config.CALLBACKS_PATH)
    
    start_epoch=callbacks.load_checkpoint(Checkpoint=checkpoint,
                                optimizer=optimizer,
                                model=Model)
    print("Model is Loaded...")
    
else:
    start_epoch=0
  
# Create initial values for model (i used zero for start)
initial_context=torch.zeros((config.BATCH_SIZE,config.HIDDEN_UNIT))
initial_decoder_input=torch.zeros((config.BATCH_SIZE,1),dtype=torch.int)

# Training
if config.TRAIN==True:
    train.Train(RESUME_EPOCHS=start_epoch,
                EPOCHS=config.EPOCHS,
                train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                model=Model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                number_size=config.NUMBER_SIZE,
                devices=devices,
                save_callbaks_fn=callbacks.save_checkpoint,
                path_checkpoint=config.CALLBACKS_PATH,
                initial_context=initial_context,
                initial_decoder_input=initial_decoder_input
                )
  
    

# Testing
if config.TEST==True:
    
    test.Testing(test_dataloader=test_dataloader,
                 loss_fn=loss_fn,
                 devices=devices,
                 model=Model,
                 number_size=config.NUMBER_SIZE,
                 initial_context=initial_context,
                 initial_decoder_input=initial_decoder_input)
 
   

# Predicting
if config.PREDICTION==True:
    
    tensor=torch.tensor([1,3,9,5,5,2,4,1])
    predict=test.Prediction(tensor=tensor,model=Model,batch_size=config.BATCH_SIZE,
                            max_len=config.MAX_LEN,
                            initial_context=initial_context,
                            initial_decoder_input=initial_decoder_input)

