"""
@uthor: Prakhar Mishra
"""

import os, csv
import argparse
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_cosine_with_hard_restarts_schedule_with_warmup

import warnings
warnings.filterwarnings('ignore')

class MyDataset(Dataset):
	def __init__(self, data_file_name, data_dir='.data/'):
		super().__init__()

		data_path = os.path.join(data_dir, data_file_name)

		self.data_list = []
		self.end_of_text_token = " <|endoftext|> "
		
		with open(data_path, encoding='UTF-8') as csv_file:
			csv_reader = csv.reader(csv_file, delimiter='\t')
			
			for row in csv_reader:
				data_str = f"{row[0]}{self.end_of_text_token}"
				#print(data_str)
				self.data_list.append(data_str)
		
	def __len__(self):
		return len(self.data_list)

	def __getitem__(self, item):
		return self.data_list[item]

def get_data_loader(data_file_name):
	dataset = MyDataset(data_file_name)
	data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
	return data_loader

def train(epochs, data_loader, batch_size, tokenizer, model, device):	
	batch_counter = 0
	sum_loss = 0.0

	for epoch in range(epochs):
		print (f'Running {epoch+1} epoch')

		for idx, txt in enumerate(data_loader):
			txt = torch.tensor(tokenizer.encode(txt[0],add_special_tokens=True,truncation=True,max_length=1024))
			txt = txt.unsqueeze(0).to(device)
			outputs = model(txt, labels=txt)
			loss, _ = outputs[:2]
			loss.backward()
			sum_loss += loss.data

			if idx%batch_size==0:
				batch_counter += 1
				optimizer.step()
				scheduler.step()
				optimizer.zero_grad()
				model.zero_grad()

			if batch_counter == 10:
				print(f"Total Loss is {sum_loss}") #printed after every 10*batch_size
				batch_counter = 0
				sum_loss = 0.0

	return model

def save_model(model, name):
	"""
	Summary:
		Saving model to the Disk
	Parameters:
		model: Trained model object
		name: Name of the model to be saved
	"""
	print ("Saving model to Disk")
	torch.save(model.state_dict(), f"{name}.pt")
	return

def get_init_model():
	"""
	Summary:
		Loading Pre-trained model
	"""
	print ('Loading/Downloading GPT-2 Model')
	tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
	model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
	return tokenizer, model

def load_models_existed(model_name):
	print("---------Getting the trained model---------")
	tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
	model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
	model_path = model_name
	model.load_state_dict(torch.load(model_path))
	print("---------Got---------")
	return tokenizer, model



if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Arguments of training GPT2 model')

	parser.add_argument('--model_name', default='mymodel.pt', type=str, action='store',
						help='The name user wants to save the model call')
	parser.add_argument('--training_data', default='mydata.csv', type=str, action='store',
						help='The name of traing data user wants to use')
	parser.add_argument('--epoches', default=3, type=int, action='store', help='the number of epoch')
	parser.add_argument('--batch_size', default=32, type=int, action='store', help='Batch size')
	parser.add_argument('--learning_rate', default=3e-5, type=float, action='store', help='Learning rate for the model')
	parser.add_argument('--max_len', default=200, type=int, action='store', help='the maximum length of sequence')
	parser.add_argument('--warmup', default=300, type=int, action='store', help='Number of warmup steps')
	parser.add_argument('--isFirst', default="Yes", type=str, action='store', help='Is it is the first tiem to train')
	parser.add_argument('--loadModel', default='mymodel.pt', type=str, action='store',
						help='Is it is the first tiem to train')

	args = parser.parse_args()

	saved_model_name = args.model_name
	training_data = args.training_data
	epoches = args.epoches
	batch_size = args.batch_size
	learning_rate = args.learning_rate
	max_len = args.max_len
	warmup = args.warmup
	isFirst = args.isFirst
	loadModel = args.loadModel
	print(isFirst)
	if isFirst == "Yes":
		print("This is init")
		tokenizer, model = get_init_model()
	# get tokenizer, model
	else:
		print("This is existed")
		tokenizer, model = load_models_existed(loadModel)
	# get processed dataset
	data_processed = get_data_loader(training_data)
	device = 'cpu'
	if torch.cuda.is_available():
		device = 'cuda'

	model_device = model.to(device)

	# call train
	model_device.train()

	optimizer = AdamW(model_device.parameters(), lr=learning_rate)
	scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup,
																   num_training_steps=-1)

	model = train(epoches, data_processed, batch_size, tokenizer, model, device)
	save_model(model, saved_model_name)
