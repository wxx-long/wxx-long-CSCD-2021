from pytorch_pretrained_bert import BertTokenizer,BertForMaskedLM
import pandas as pd
import math
import csv
import re
# 第二种
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForMaskedLM
import os


#设置使用那块GPU “0”是GPU的id
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # //主要用的是这句，其他是校验
 
# #查看GPU是否可用
# cuda_available=torch.cuda.is_available()   
# print('cuda_available:',cuda_available)

# #查看可用GPU数量
# coun=torch.cuda.device_count()
# print(coun) 

# #返回GPU名字 0 是返回GPU名字的ID
# name=torch.cuda.get_device_name(0)
# print(name)
# #返回当前设备的ID
# num=torch.cuda.current_device()
# print(num)







# Load pre-trained model (weights)
with torch.no_grad():
    model = BertForMaskedLM.from_pretrained('bert-base-chinese')  # 加载模型
    model.eval()
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')   # 加载tokenizer
    


def clean_str(string):
  string = re.sub('[|]+', "", string)
  # 分别是汉字。？！；，...半角数字（常用的那种） 全角数字
  # string = re.sub(u"([^\u4e00-\u9fa5|\u3002|\uff1f|\uff01|\uff1b|\uff0c|\u2026{2,}|\u0030-\u0039])","",string) #去除特殊字符，只保留汉字 。？！
  string = re.sub(r"\s{2,}", " ", string)
  return string


def preprocess(input_file):
  # 传参：测试csv文件，test_sent.pkl，test_data.pkl
  test_sent = []
  keys = ['(A)', '(B)', '(C)', '(D)', '(E)']
  with open(input_file, "r", encoding='utf-8-sig') as f:
      reader = csv.DictReader(f)
      for i, row in enumerate(reader):
          test_instance = {'编号':row['编号'], '候选句子':[]}  # {'编号': '960', 'candidates': []}
          question = row['问题']  # str类型
          choices = [row[x] for x in keys]  # ['负', '竹钩', '地广所', '赫斯想', '猛仅']
          questions = [question.replace('_____', word) for word in choices] # list类型，元素是把每个选项带入问题
          test_instance['候选句子'] = ['' + q for q in questions]  # 字典类型{{'编号': '960', '候选句子': ['BOS至此，中国队在本届世界杯赛中以8战6胜2负的战绩暂居第四名。EOS','...','...','...'}
          test_sent.append(test_instance)
  return test_sent
  
  

input_files = "./test_data/test_data-500.csv"
test_sent = preprocess(input_files)
answer = []
# print(test_sent)
# print(test_sent[0])
choices = ['A', 'B', 'C', 'D', 'E']
answer = []
for i in range(0,len(test_sent)):  # 遍历每个样本
  scores = []
  print(i)
  sentence_list = []
  for sample in test_sent[i]['候选句子']:  # 遍历每个样本中的句子
    sentence_list.append(sample)
    score = []
  for sentence in sentence_list:
    tokenize_input = tokenizer.tokenize(sentence)  # 把句子tokenizer化
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])  # 转化为张量
    sen_len = len(tokenize_input)  # 转换为张量
    sentence_loss = 0.
    for i, word in enumerate(tokenize_input):  # 遍历每个tokenizer
      # 为每个词都加上mask，为接下来的计算做准备
      tokenize_input[i] = '[MASK]'
      # print(tokenize_input)
      mask_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])  # 转换为张量
      output = model(mask_input)  # 传参：转换为的张量
      prediction_scores = output[0]
      softmax = nn.Softmax(dim=0)
      ps = softmax(prediction_scores[0, i]).log()
      word_loss = ps[tensor_input[0, i]]
      sentence_loss += word_loss.item()
      tokenize_input[i] = word # 本单词转换为mask之后再把其转换为原来的单词
    ppl = np.exp(-sentence_loss/sen_len)
    score.append(ppl)
  #   pricnt(ppl)
  # print(sore)
  min_index = score.index(min(score))    # 最小分数下标
  # print(min_index)
  answer.append(choices[min_index])
print(answer)



# 存储预测结果
print("存储预测结果")
out_path = "./prediction_result"
with open(out_path + '/prediction.csv', 'w', encoding='utf-8-sig') as out:
  writer = csv.writer(out, delimiter=',')
  writer.writerow(['编号','答案'])
  for i, ans in enumerate(answer):
      writer.writerow([str(i+1), ans])

print("Saved prediction to {}".format(out_path))
# print("Total run time: {}s".format(time.time() - start))


# 计算准确率
print("计算准确率")
ans = open('./test_data/test_answer-500.csv', 'r')
ans_list = []
for i, row in enumerate(csv.reader(ans)):
	if i == 0:
		continue
	else:
		ans_list.append(row[1])

prd = open("./prediction_result/prediction.csv", 'r')
prd_list = []
for i, row in enumerate(csv.reader(prd)):
	if i == 0:
		continue
	else:
		prd_list.append(row[1])

acc = 0.
for i in range(len(prd_list)):
	if prd_list[i] == ans_list[i]:
		acc += 1.

print ('Accuracy: {}'.format(acc/float(len(prd_list))))
  
    