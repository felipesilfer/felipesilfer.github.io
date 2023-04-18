---
title: "Criando um Chatbot com GPT-3"
description: "Um passo a passo para implementar um chatbot com GPT-3 utilizando a OpenAI API."
date: 2023-04-16T00:33:47-03:00
image: "/images/articles/creating-a-chatbot-gpt3/gpt-3-thumb.png"
---

![](/images/articles/creating-a-chatbot-gpt3/gpt-3-thumb.png#center "OpenAI GPT-3")

## Motivação
Em 2022, a OpenAI lançou o chatbot ChatGPT, que é baseado no modelo de linguagem GPT-3.5, uma versão aprimorada do GPT-3. 

O GPT-3 é um dos modelos de linguagem mais avançados do mundo, com mais de 175 bilhões de parâmetros. A OpenAI disponibiliza publicamente uma API que permite a criação de aplicativos, como completar textos e gerar imagens, usando o poder do GPT-3. Além disso, há um teste gratuito disponível para experimentar a API.

## O projeto

Resolvi tentar utilizar a API da OpenAI para criar um Chatbot pensando num caso de uso simples: Um bot para atendimento de estudantes de uma universidade genérica.

A ideia é que o Chatbot consiga responder dúvidas sobre a universidade, como saber contatos, endereço, cursos disponíveis, procedimentos, entre outras informações.

Nesse artigo você vai aprender a:

1. Configurar o acesso a API GPT-3 da OpenAI;
2. Criar um Chatbot generalista usando o modelo Davici do GPT-3;
3. Tratar os dados para criar um Fine Tunning do GPT-3;
4. Criar um Chatbot para atendimento universitário usando o Modelo criado;

## O que será preciso

Neste artigo presumo que:

* Você tem conhecimento básico de Python.
* Já possiu Python 3 instalado.
* Está em ambiente Linux (ou Windows com WSL2).

Observações:

* Para facilitar a compreenção, este post é uma adaptação de um Jupyter Notebook que eu fiz e disponibilizarei o repositório futuramente.
* Adaptações podem precisar ser feitas. 😉

---

## Passo 1: Criar conta OpenAI

Vá até a página de [**Sing Up da OpenAI**](https://platform.openai.com/signup) e siga as instruções para criar uma nova conta.

![](/images/articles/creating-a-chatbot-gpt3/signup-opneai.png#center)

## Passo 2: Criar uma API Key

Depois que de ter feito login na sua conta OpenAI, clique em '**Personal**', no canto superior direito, em seguida vá em '**View API keys**'.

![](/images/articles/creating-a-chatbot-gpt3/create-api-key.png#center)

Isso vai levá-lo até página de '**API Keys**', onde deve-se clicar no botão '**+ Create new secret key**' que irá gerar sua Key que será usada posteriomente no código.

![](/images/articles/creating-a-chatbot-gpt3/create-new-secret-key.png#center)

Vamos salvar a **API Key**  que foi gerada num *YAML file* com o nome ``openai-apikey.yml`` contendo a API key da sua conta OpenAI. Dessa forma podemos usá-la em nosso código.

 No corpo do arquivo vamos seguir esse formato:

```YAML
API-key:  "xx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

## Passo 3: Implementar Chatbot Genérico

Vamos fazer uso das bibliotecas OpenAI e Pandas no código, portando as mesmas devem ser instaladas:

>Terminal:
```console
pip install openai pandas
```

Outro passo importante é variável de ambiente ``OPENAI_API_KEY```` adicionando a seguinte linha ao seu script de inicialização do shell (por exemplo, .bashrc, zshrc, etc.) ou executando-a na linha de comando antes do comando de no seu terminal:

>Terminal:
```console
export OPENAI_API_KEY="xx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```
Depois desses ajustes deixamos o ambiente pronto para acessar a API.

 Segue agora uma implentação de um Chatbot genérico usando a GPT3 usando a API da OpenAI:

>Python:
```python
import openai, pandas as pd, yaml, json, time

#importando a API-Key
with open('openai-apikey.yml', 'r') as file:
    api_dict = yaml.safe_load(file)

#criando instancia de Chat com a API passando texto de prompt e modelo a ser usado    
def askGPT(text, engine):
    openai.api_key = api_dict['API-key']
    if engine=="text-davinci-003":
        stop = ""
    else:
        stop = "\n"
        
    response = openai.Completion.create(
        engine = engine,
        prompt = text,
        temperature = 0.6,
        max_tokens = 150,
        stop = stop
        
    )
    return print(response.choices[0].text)

#criando interface para o chat
def simple_chat(engine):
    while True:
        print("GPT: Me pergunte algo (digite: #sair para encerrar)\n")
        myQn = input()
        if (myQn=='#sair'):
            break
        askGPT(myQn, engine=engine)
        print('\n')
        
simple_chat("text-davinci-003")
```

```
    GPT: Me pergunte algo (digite: #sair para encerrar)
    
     O que é Jupyter Notebook?

    
    Jupyter Notebook é uma ferramenta open-source que permite a criação e compartilhamento de documentos contendo código, equações, visualizações de dados e texto narrativo. É um ambiente ideal para a computação científica interativa, pois permite a criação de blocos de código reutilizáveis, anotações, visualizações e também a execução de códigos.
    
    
    GPT: Me pergunte algo (digite: #sair para encerrar)
    
     #sair
```


## Passo 4: Preparando Dataset para Fine tunning

Fine tuning é o ajuste fino dado a um modelo para que atenda melhor um determinado fim.

Para fins de estudo, vamos realizar o fine tunning para um chatbot com finalidade de responder perguntas básicas sobre uma universidade.

Para treinar o modelo, será usado o Dataset ``intents.json`` disponível no Kaggle: [**University Chatbot Dataset - by Niralii Vaghani**](https://www.kaggle.com/dsv/5024271 "University Chatbot Dataset - by Niralii vaghani")

### Formatação

Na documentação é informado que o dataset de treinamento deve informar ao GPT-3 o que você gostaria que ele respoda.

Para isso os dados de treinamento devem estar no formato de documento [JSONL](https://jsonlines.org/) onde cada linha representa um par prompt-resposta correspondentes. Conforme exemplo abaixo:

```JSON
{"prompt": "<prompt text> ->", "completion": " <ideal generated text> \n"}
{"prompt": "<prompt text> ->", "completion": " <ideal generated text> \n"}
{"prompt": "<prompt text> ->", "completion": " <ideal generated text> \n"}
...
```

### Tradução e formatação do Dataset
Para traduzir o dataset do inglês para português usando a API, pode ser usado o código a seguir, mas pode demorar um pouco devido o limite de 60 requests / min que a API tem:

>Python:
```python
#criando instancia de Chat com a API passando o prompt para tradução do dataset   
def translateGPT(text):
    openai.api_key = api_dict['API-key']
    task_description = "Translate English to Brazilian Portuguese:"
    to_translate = text
    prompt = task_description+"\n"+to_translate+" ->"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.3,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    response = json.loads(json.dumps(vars(response)))
    time.sleep(1) #a API limita 60 requests / min, necessário 1s de espera por chamada de função
    return response['_previous']['choices'][0]['text']

#reformatando dados do dataset
df = pd.read_json('intents.json')
df_output = pd.DataFrame()
for row in df.index:
    patterns = df.iloc[row].to_dict()['intents']['patterns']
    responses = df.iloc[row].to_dict()['intents']['responses']
    for p in range(0,len(patterns)):
        for r in range(0,len(responses)):
            dict = {}
            dict['prompt'] = translateGPT(patterns[p])+" ->"
            dict['completion'] = " "+translateGPT(responses[r])+".\n"
            df_dict = pd.DataFrame([dict])
            df_output = pd.concat([df_output, df_dict], ignore_index = True)

#salvando dataset formatado para o arquivo 'dataframe.jsonl'            
with open('dataframe.jsonl', 'w', encoding='utf-8') as file:
    file.write(df_output.to_json(orient='records', lines=True, force_ascii=False))
```

Para não gastar crédito do Trial Free da plataforma, foi feita a tradução por fora do dataset original, e o nomeamos como `intents_ptBR.json`.

>Python:
```python
#visualizando amostra de dados do dataset orginal:
df = pd.read_json('intents_ptBR.json')
df.head(5)
```

|   |intents|
| :--- | :--- |
|0|{'tag': 'saudação', 'patterns': ['Oi', 'Como v...|
|1|{'tag': 'despedida', 'patterns': ['Tchau', 'At...|
|2|{'tag': 'criador', 'patterns': ['qual é o nome...|
|3|{'tag': 'nome', 'patterns': ['nome', 'seu nome...|
|4|{'tag': 'horario', 'patterns': ['horário da fa...|

</br>

O código para reformatação do data é similar ao anterior, sendo apenas retirada a tradução pela função ``translateGPT()``:

>Python:
```python
#reformatando dados do dataset
df_output = pd.DataFrame()
for row in df.index:
    patterns = df.iloc[row].to_dict()['intents']['patterns']
    responses = df.iloc[row].to_dict()['intents']['responses']
    for p in range(0,len(patterns)):
        for r in range(0,len(responses)):
            dict = {}
            dict['prompt'] = patterns[p]+" ->"
            dict['completion'] = " "+responses[r]+".\n"
            df_dict = pd.DataFrame([dict])
            df_output = pd.concat([df_output, df_dict], ignore_index = True)

#visualizamdp amostra do dataset formatado corretamente            
df_output.head(5)
```

|   |prompt|completion|
| :--- | :--- | :--- |
|0|Oi -&gt;|Olá!.\n<|
|1|Oi -&gt;|Bom te ver de novo!.\n|
|2|Oi -&gt;|Olá, como posso ajudar?.\n|
|3|Como você está? -&gt;|Olá!.\n|
|4|Como você está? -&gt;|Bom te ver de novo!.\n|

</br>

Após transformado, o dataset deve ser salvo em arquivo JSONL conforme sugere a documentação:

>Python:
```python
#salvando dataset formatado para o arquivo 'dataframe.jsonl'
with open('dataframe.jsonl', 'w', encoding='utf-8') as file:
    file.write(df_output.to_json(orient='records', lines=True, force_ascii=False))
```

### Checando qualidade do dataframe
Ao instalar a lib OpenAI, junto dela vem uma ferramenta para checar a qualidade do dataset antes de realizar o upload do mesmo para o processamento do fine tunning.

O comando ``openai tools fine_tunes.prepare_data -f`` analisa o arquivo JSONL, e para nosso arquivo, passamos os parâmetros ``y`` para remoção de duplicatas, ``n`` dividir o arquivo em sets separados de treino  e validação (devido o dataset não ser grande o bastante pra isso), e ``y``  para salvar as modificações num novo arquivo ``'dataframe_prepared.jsonl'``, o que vai retornar um Prompt parecido com este:

>Terminal:
```console
printf "y\nn\ny" | openai tools fine_tunes.prepare_data -f dataframe.jsonl
```

```console
    Analyzing...
    
    - Your file contains 521 prompt-completion pairs
    - Based on your data it seems like you're trying to fine-tune a model for classification
    - For classification, we recommend you try one of the faster and cheaper models, such as `ada`
    - For classification, you can estimate the expected model performance by keeping a held out dataset, which is not used for training
    - There are 32 duplicated prompt-completion sets. These are rows: [66, 67, 68, 69, 89, 152, 156, 160, 190, 196, 211, 218, 233, 244, 256, 296, 368, 369, 372, 394, 395, 396, 397, 418, 459, 463, 480, 481, 490, 491, 496, 497]
    - All prompts end with suffix ` ->`
    
    Based on the analysis we will perform the following actions:
    - [Recommended] Remove 32 duplicate rows [Y/n]: - [Recommended] Would you like to split into training and validation set? [Y/n]: 
    
    Your data will be written to a new JSONL file. Proceed [Y/n]: 
    Wrote modified file to `dataframe_prepared.jsonl`
    Feel free to take a look!
    
    Now use that file when fine-tuning:
    > openai api fine_tunes.create -t "dataframe_prepared.jsonl"
    
    After you’ve fine-tuned a model, remember that your prompt has to end with the indicator string ` ->` for the model to start generating completions, rather than continuing with the prompt. Make sure to include `stop=[".\n"]` so that the generated texts ends at the expected place.
    Once your model starts training, it'll approximately take 14.07 minutes to train a `curie` model, and less for `ada` and `babbage`. Queue will approximately take half an hour per job ahead of you.
```

Checando o novo dataframe criado:

>Python:
```python
df = pd.read_json('dataframe_prepared.jsonl', lines=True)
df.head(10)
```

|   |prompt|completion|
| :--- | :--- | :--- |
|0|Oi -&gt;|Olá!.\n<|
|1|Oi -&gt;|Bom te ver de novo!.\n|
|2|Oi -&gt;|Olá, como posso ajudar?.\n|
|3|Como você está? -&gt;|Olá!.\n|
|4|Como você está? -&gt;|Bom te ver de novo!.\n|
|5|Como você está? -&gt;|Olá, como posso ajudar?.\n|
|6|Tem alguém aí? -&gt;|Olá!.\n|
|7|Tem alguém aí? -&gt;|Bom te ver de novo!.\n|
|8|Tem alguém aí? -&gt;|Olá, como posso ajudar?.\n|
|9|Olá -&gt;|Olá!.\n|

</br>


Executando o comando pela segunda vez, vai mostrar como deve ser a saída quando um dataframe está formatado da maneira correta, o resultado deve ser um prompt como este:

>Terminal:
```console
printf "n" | openai tools fine_tunes.prepare_data -f dataframe_prepared.jsonl
```

```console
    Analyzing...
    
    - Your file contains 489 prompt-completion pairs
    - Based on your data it seems like you're trying to fine-tune a model for classification
    - For classification, we recommend you try one of the faster and cheaper models, such as `ada`
    - For classification, you can estimate the expected model performance by keeping a held out dataset, which is not used for training
    - All prompts end with suffix ` ->`
    
    No remediations found.
    - [Recommended] Would you like to split into training and validation set? [Y/n]: 
    You can use your file for fine-tuning:
    > openai api fine_tunes.create -t "dataframe_prepared.jsonl"
    
    After you’ve fine-tuned a model, remember that your prompt has to end with the indicator string ` ->` for the model to start generating completions, rather than continuing with the prompt. Make sure to include `stop=[".\n"]` so that the generated texts ends at the expected place.
    Once your model starts training, it'll approximately take 14.07 minutes to train a `curie` model, and less for `ada` and `babbage`. Queue will approximately take half an hour per job ahead of you.
```

## Passo 5: Upload do Dataset para preparar o Fine tunning

O usando o command line tools da OpenAI, ou a lib no exemplo a seguir, preparamos o dataframe que foi tratado para upload.

Com isso ele ganha um id que deve ser salvo para referenciarmos durante a criação do fine tunning:

>Python:
```python
upload_response = openai.File.create(
    file=open('dataframe_prepared.jsonl'),
    purpose='fine-tune'
)
file_id = upload_response.id
upload_response
```

>Terminal:
```console

    <File file id=file-j9xUWUtOagExIj7QiTRmKXbO at 0x7f49aed5bb50> JSON: {
      "bytes": 59864,
      "created_at": 1681604031,
      "filename": "file",
      "id": "file-j9xUWUtOagExIj7QiTRmKXbO",
      "object": "file",
      "purpose": "fine-tune",
      "status": "uploaded",
      "status_details": null
    }
```


### Criando Fine Tunning
Após o upload, podemos treinar o modelo de Fine Tunning usando a lib OpenAI.

Passaremos como parâmetros o ``training_file`` identificado pelo ID coletado anteriormente, e o ``model`` que é o ID do modelo em que o fine tunning vai se basear, que no nosso caso é o ``davinci``.

Devemos ficar atentos ao ID do finetune e salvá-lo para podermos checar o andamento do processamento, conforme exemplo abaixo:

>Python:
```python
fine_tune = openai.FineTune.create(training_file=file_id, model="davinci")
finetune_id = fine_tune.id
fine_tune
```

```console

    <FineTune fine-tune id=ft-iMXzAEiUcFE4nZLDd5kp6tCY at 0x7f49aedecef0> JSON: {
      "created_at": 1681604044,
      "events": [
        {
          "created_at": 1681604044,
          "level": "info",
          "message": "Created fine-tune: ft-iMXzAEiUcFE4nZLDd5kp6tCY",
          "object": "fine-tune-event"
        }
      ],
      "fine_tuned_model": null,
      "hyperparams": {
        "batch_size": null,
        "learning_rate_multiplier": null,
        "n_epochs": 4,
        "prompt_loss_weight": 0.01
      },
      "id": "ft-iMXzAEiUcFE4nZLDd5kp6tCY",
      "model": "davinci",
      "object": "fine-tune",
      "organization_id": "org-gzduqVnnZJVEbUTxiMIKEE43",
      "result_files": [],
      "status": "pending",
      "training_files": [
        {
          "bytes": 59864,
          "created_at": 1681604031,
          "filename": "file",
          "id": "file-j9xUWUtOagExIj7QiTRmKXbO",
          "object": "file",
          "purpose": "fine-tune",
          "status": "processed",
          "status_details": null
        }
      ],
      "updated_at": 1681604044,
      "validation_files": []
    }
```
Visualizando o ID:

>Python:
```python
print(finetune_id)
```

```
ft-iMXzAEiUcFE4nZLDd5kp6tCY
```

Para checar o progresso do treinamento do modelo, podemos usar o command line tool informando o ID do fine tune que salvamos anteriormente e quando o mesmo estiver concluído, trará um prompt parecido com este quando concluir o treinamento (execute mais de uma vez para acompanhar), informando o ID do modelo gerado:

>Terminal:
```console
openai api fine_tunes.follow -i ft-iMXzAEiUcFE4nZLDd5kp6tCY
```

```console
    [2023-04-15 21:14:04] Created fine-tune: ft-iMXzAEiUcFE4nZLDd5kp6tCY
    [2023-04-15 21:14:16] Fine-tune costs $2.01
    [2023-04-15 21:14:16] Fine-tune enqueued. Queue number: 0
    [2023-04-15 21:24:19] Fine-tune started
    [2023-04-15 21:28:15] Completed epoch 1/4
    [2023-04-15 21:30:41] Completed epoch 2/4
    [2023-04-15 21:33:06] Completed epoch 3/4
    [2023-04-15 21:35:30] Completed epoch 4/4
    [2023-04-15 21:36:04] Uploaded model: davinci:ft-personal-2023-04-16-00-36-03
    [2023-04-15 21:36:05] Uploaded result file: file-rAuDVNmEI9EaUYuSvvLGtQCZ
    [2023-04-15 21:36:05] Fine-tune succeeded
    
    Job complete! Status: succeeded 🎉
    Try out your fine-tuned model:
    
    openai api completions.create -m davinci:ft-personal-2023-04-16-00-36-03 -p <YOUR_PROMPT>
```

Recuperando o ``fine_tuned_model`` do último fine tune realizado (assegurar na etapa anterior que o fine tune foi concluído):

>Python:
```python
finetune_list = json.loads(json.dumps(openai.FineTune.list()))
finetuned_model_id = finetune_list['data'][-1]['fine_tuned_model']
print(finetuned_model_id)
```
```
    'davinci:ft-personal-2023-04-16-00-36-03'
```


### Testando o modelo de Fine Tunning treinado

Para testar o modelo, basta criar uma nova instância de chat passando o ``finetuned_model_id`` do fine tunning  como parâmetro da engine.

Segue prompt resultante do teste:

>Python:
```python
simple_chat(finetuned_model_id)
```

```console
    GPT: Me pergunte algo (digite: #sair para encerrar)
    


     Olá, com quem eu falo?


     -> Você pode me chamar de Leitor da Mente..
    
    
    GPT: Me pergunte algo (digite: #sair para encerrar)
    


     Qual o horário de funcionamento da faculdade?


     -> A faculdade está aberta das 8h às 17h, de segunda a sábado!.
    
    
    GPT: Me pergunte algo (digite: #sair para encerrar)
    


     E tem telefone para contato?


     -> Você pode entrar em contato através do número: (xx) xxxx-xxxx.
    
    
    GPT: Me pergunte algo (digite: #sair para encerrar)
    


     Quais cursos são disponíveis?


     -> Nossa universidade oferece Tecnologia da Informação, Engenharia da Computação, Engenharia Mecânica, Engenharia Química, Engenharia Civil e Matemática..
    
    
    GPT: Me pergunte algo (digite: #sair para encerrar)
    


     Tem cantina na faculdade?


     -> Nossa universidade tem uma cantina com uma variedade de alimentos disponíveis..
    
    
    GPT: Me pergunte algo (digite: #sair para encerrar)
    


     O que tem pra comer na cantina?


     -> Servimos Franky, Locho, Alu-puri, Kachori, Khavsa, Thaali e muitos mais no cardápio..
    
    
    GPT: Me pergunte algo (digite: #sair para encerrar)
    


     #sair

```


## Conclusão

Foi possivel criar um chatbot com respostas razoáveis mesmo um Dataset considerado pequeno (aproximadamente 500 entradas), sendo possível concluir que com um dataset mais robusto os resultados podem ser ainda melhores.

### Pontos observados

- A documentação da OpenAI é um pouco falha, misturando instruções da Lib Python com HTTP request methods no mesmo local, dificultando um pouco encontrar referência
- O custo para treinamento do Fine Tunning é realmente barato, e o custo de uso dos modelos prontos ou dos modelos criados é mais baixo ainda
- É dado um saldo de 18 dólares como Free Trial, o que possibilita fazer realmente muita coisa legal dado o baixo custo
- O tratamento do Dataset é onde é necessário dedicar a maior parte do esforço, visto que a má elaboração do Dataset implica negativamente na qualidade das respostas

### Possibilidades futuras

O mais importante para criação de um chatbot é a qualidade dos dados coletados e organizados para montar o Dataset de treinamento.
Seguir as boas práticas sugeridas na documentação é realmente muito importante.
Nesse caso de uso eu não criei prompts mais complexos para melhorar a qualidade das resposta, mas é possível ser feito isso para enriquecer as respostas do chatbot, e eu recomendo.