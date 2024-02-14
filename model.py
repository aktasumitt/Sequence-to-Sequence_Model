import torch
import torch.nn as nn

class BahdanauAttention(nn.Module):
    def __init__(self,len_encoder,len_decoder):
        super(BahdanauAttention,self).__init__()
        
        self.w_encoder=nn.Linear(in_features=len_encoder,out_features=512)
        self.w_decoder=nn.Linear(in_features=len_decoder,out_features=512)
        self.V=nn.Linear(512,1)
        
    def forward(self,encoder_out,decoder_out):
        
        s1=self.w_encoder(encoder_out)
        s2=self.w_decoder(decoder_out.permute(1,0,2))
        
        score=nn.functional.tanh(s1+s2)
        score=self.V(score)
        attn_weights=nn.functional.softmax(score,dim=1)
        
        context_vector=attn_weights*encoder_out
        context_vector=torch.sum(context_vector,dim=1)
        
        return context_vector
        
        
        

class Seq2Seq(nn.Module):
    
    def __init__(self,label_size,hidden_unit,max_len,batch_Size):
        
        super(Seq2Seq,self).__init__()
        
        self.batch_size=batch_Size
        self.max_len=max_len
        self.hidden_unit=hidden_unit

        self.embedding_encoder=nn.Embedding(num_embeddings=label_size+1,embedding_dim=hidden_unit,padding_idx=0)
        self.embedding_decoder=nn.Embedding(num_embeddings=label_size+1,embedding_dim=hidden_unit,padding_idx=0)
        
        self.encoder=self.Encoder(input_size=hidden_unit,hidden_size=hidden_unit)
        self.decoder=self.Decoder(input_size=hidden_unit*2,hidden_size=hidden_unit) # we will concat with context vector
        self.decoder_linear=nn.Linear(in_features=hidden_unit,out_features=label_size+1)
        
        self.Attention=BahdanauAttention(len_encoder=hidden_unit,len_decoder=hidden_unit)
    
    def Encoder(self,input_size,hidden_size):
        
        encoder_block=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=1,batch_first=True)

        return encoder_block
    
    
    def Decoder(self,input_size,hidden_size):
        decoder_block=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=1,batch_first=True)
    
        return decoder_block

    
    def forward(self,input_encoder,initial_context,initial_decoder_input):
        
        context=initial_context
        decoder_input=initial_decoder_input
        
        embed_encoder=self.embedding_encoder(input_encoder)
        encoder_out,(hid_s,cell_s)=self.encoder(embed_encoder)

        out_decoder_list=[]
        
        for i in range(self.max_len):
            
            embed_decoder=self.embedding_decoder(decoder_input)

            input_decoder=torch.cat([embed_decoder,context.unsqueeze(1)],dim=-1)
            
            decoder_out,(hid_state_d,_)=self.decoder(input_decoder,(hid_s,cell_s))
            
            context=self.Attention(encoder_out,hid_state_d)
            
            out=self.decoder_linear(decoder_out.view(-1,1*self.hidden_unit))
            out_decoder_list.append(out)
            
            soft=nn.functional.softmax(out,dim=1)
            _,pred=torch.max(soft,-1)
            
            decoder_input=pred.unsqueeze(1)

        
        out_decoder_list=torch.stack(out_decoder_list).permute(1,0,2)
        return out_decoder_list