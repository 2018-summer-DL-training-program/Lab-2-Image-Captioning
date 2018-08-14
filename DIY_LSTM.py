import math
import torch
import warnings
import itertools
import numbers

from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence
from torch.nn import init
import torch.nn as nn

class my_LSTM(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, if_bias=True, batch_first=False):
		super(my_LSTM, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.if_bias = if_bias
		self.batch_first = batch_first

		'''
			TO-DO: define each matric multiplication here
		'''

		self.sigmoid = nn.Sigmoid()
		self.tanh = nn.Tanh()

		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1.0 / math.sqrt(self.hidden_size)
		for weight in self.parameters():
			init.uniform_(weight, -stdv, stdv)

	def forward(self, input_seq, hx=None):
		'''
			TO-DO: check if input_seq is a packed sequence. If yes, unpack it.
		'''

		# outputs
		hidden_state_list = []
		cell_state_list = []

		'''
			TO-DO: if hx is None, initialize it.
		'''
		if hx is None:
			pass
		else:
			pass

		'''
			TO-DO: implement LSTM here
		'''

		hidden_state_list = torch.cat(hidden_state_list, 0)
		cell_state_list = torch.cat(cell_state_list, 0)

		return hidden_state_list, (hidden_state_list, cell_state_list)
