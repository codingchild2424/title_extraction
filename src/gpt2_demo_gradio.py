import gradio as gr

import torch
import transformers

from gpt2_define_argparser import define_argparser

# saved_model
def load_model(model_path, config):
    saved_data = torch.load(
        model_path,
        map_location="cpu" if config.gpu_id < 0 else "cuda:%d" % config.gpu_id
    )

    gpt2_best = saved_data["model"]
    train_config = saved_data["config"]
    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(config.pretrained_model_name)

    ## Load weights.
    model = transformers.GPT2LMHeadModel.from_pretrained(config.pretrained_model_name)
    model.resize_token_embeddings( len(tokenizer) )
    model.load_state_dict(gpt2_best)

    return model, tokenizer


# main
def inference(prompt):

    config = define_argparser()
    model_path = config.model_fpath

    model, tokenizer = load_model(
        model_path=model_path, 
        config=config
        )
    #+"1줄요약"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    #input_ids = torch.tensor(input_ids)
    #input_ids = input_ids.unsqueeze(0)
    output = model.generate(
        input_ids,
        max_length=20,
        repetition_penalty=2.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        use_cache=True
        )
    output = tokenizer.decode(output[0], skip_special_tokens=True)    

    return output

demo = gr.Interface(
    fn=inference, 
    inputs="text", 
    outputs="text" #return 값
    ).launch(share=True) # launch(share=True)를 설정하면 외부에서 접속 가능한 링크가 생성됨

demo.launch()