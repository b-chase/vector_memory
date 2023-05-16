from transformers import BertTokenizerFast, EncoderDecoderModel
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizerFast.from_pretrained('mrm8488/bert-mini2bert-mini-finetuned-cnn_daily_mail-summarization',)
model = EncoderDecoderModel.from_pretrained('mrm8488/bert-mini2bert-mini-finetuned-cnn_daily_mail-summarization').to(device)

def generate_summary(text):
    # cut off at BERT max length 512
    inputs = tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    output = model.generate(input_ids, attention_mask=attention_mask)

    return tokenizer.decode(output[0], skip_special_tokens=True)


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

adv_tokenizer = AutoTokenizer.from_pretrained("mrm8488/roberta-med-small2roberta-med-small-finetuned-cnn_daily_mail-summarization")

adv_model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/roberta-med-small2roberta-med-small-finetuned-cnn_daily_mail-summarization")

def big_summarize_text(text):
    fixed_text = text.replace("\n", "  ")
    prompt_text = f'{fixed_text[0:600]}'
    print("...getting tokens...")
    # adv_inputs = adv_tokenizer(prompt_text, return_tensors="pt")
    # adv_input_ids = adv_inputs.input_ids.to(device)
    # attention_mask = adv_inputs.attention_mask.to(device)
    # output = adv_model.generate(input_ids, attention_mask=attention_mask)
    # return adv_tokenizer.decode(output[0], skip_special_tokens=True)

    adv_input_ids = adv_tokenizer(prompt_text, return_tensors="pt").input_ids
    print("...generating outputs...")

    outputs = adv_model.generate(adv_input_ids[0:514])
    print("...decoding outputs...")

    return adv_tokenizer.decode(outputs[0], skip_special_tokens=True)
  

text_list = []

for i in range(6) :
    with open(f'sample_text{i+1}.txt', 'r') as f:
        text = f.read()
        text_list.append(text)
        
        print(f'Summaries for file {f.name}:')
        simple_summary = generate_summary(text)
        print(f'Simple: {simple_summary}')
        fancy_summary = big_summarize_text(text)
        print(f'Advanced: {fancy_summary}')
        print('\n')        
